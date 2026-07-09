"""
practical_sigma [B] — TWO-STAGE offline calibration of the syndrome→σ map.

  LUT1  syndrome ratio w  →  effective SNR :  invert the steady-state w̄(SNR) of
        PURE BP (no denoiser).  Higher w ⇒ less-converged ⇒ lower effective SNR.
  LUT2  SNR  →  σ*(SNR) :  at each SNR, sweep the denoiser σ over a FULL EP
        [5]×20 decode and pick the σ minimising end-to-end NACK.
        (The single-step pixel-MSE objective was tried and DEGENERATES to σ_min —
        the EDM denoiser does not improve the BP posterior MEAN in MSE; its value
        is LLR-domain mode-selection, visible only end-to-end.  So LUT2 uses NACK.)
  compose  σ(w) = LUT2(LUT1(w)) , saved as a CalibratedLookupScheduler JSON
        (thresholds = ratio bin edges, sigma_levels = σ*), reused at runtime.

σ is a pixel-domain std, the syndrome a code-domain count; they link only through
the SNR — hence two stages.  σ*(SNR) is the denoiser's own best noise level.

GPU recipe: TF on CPU (BP), torch denoiser on GPU.
"""
import json, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no

from decoder import LDPC5GDecoder_soft
from syndrome_sigma_schedule import DEFAULT_SYNDROME_HARD_DECISION
from sigma_experiment import build_decoder, run as ep_run, K, SCHEDULE

CKPT = "checkpoints/denoiser.pt"
NUM_BPS = 1
SEED = 7
OUT_JSON = "results/sigma_2stage_lookup.json"
OUT_PLOT = "results/sigma_2stage.png"


def _bank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    imgs = ds.data.numpy().astype(np.uint8)              # avoid PIL per-item iteration
    return tf.constant(np.unpackbits(imgs.reshape(-1, 784), axis=1), tf.int32)


def build_lut1(ldpc, snr_grid, batch, rounds):
    """Pure-BP steady-state syndrome ratio w̄(SNR)."""
    crc = CRCEncoder("CRC24A")
    mp = Mapper("pam", num_bits_per_symbol=NUM_BPS)
    dm = Demapper("app", "pam", num_bits_per_symbol=NUM_BPS); aw = AWGN()
    dec = LDPC5GDecoder_soft(
        ldpc, k_payload=K, num_iter=sum(SCHEDULE), bp_schedule=SCHEDULE,
        ep_mode=False, adaptive_sigma=False, denoiser_kwargs=dict(device=DEV),
        cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
        hard_out=False, return_infobits=True, llr_max=30.0)
    dec.denoiser.load_weights_pt(CKPT); dec.alpha = 0.0; dec.beta = 0.0
    bank = _bank()
    w_bar = []
    for eb in snr_grid:
        no = ebnodb2no(eb, NUM_BPS, ldpc.coderate)
        ss = []
        for r in range(rounds):
            tf.random.set_seed(SEED + r + int(eb * 1000))
            idx = tf.random.uniform([batch], 0, tf.shape(bank)[0], dtype=tf.int32)
            u = tf.gather(bank, idx)
            y = aw(mp(ldpc(crc(tf.cast(u, ldpc.rdtype)))), no)
            _ = dec(dm(y, no))
            ss.append(dec.last_chunk_diagnostics[-1]["syndrome_ratio"].numpy())
        w_bar.append(float(np.mean(np.concatenate(ss))))
        print(f"  LUT1 SNR={eb:.2f}  w̄_ss={w_bar[-1]:.4f}", flush=True)
    return w_bar


def build_lut2(ldpc, snr_grid, sig_cands, batch, rounds):
    """σ*(SNR) = argmin_σ end-to-end NACK of fixed-σ EP [5]×20.

    One decoder is built once; only the scalar denoiser σ is varied per candidate
    (avoids reloading the checkpoint for every point)."""
    dec = build_decoder(ldpc, "fixed", fixed_sigma=float(sig_cands[0]))
    sigstar, nack_curves = [], []
    for eb in snr_grid:
        nacks = []
        for s in sig_cands:
            dec.denoiser.sigma = float(s)
            nacks.append(ep_run(dec, ldpc, eb, batch, rounds)["nack"])
        i = int(np.argmin(nacks))
        sigstar.append(float(sig_cands[i])); nack_curves.append(nacks)
        print(f"  LUT2 SNR={eb:.2f}  NACK/{batch*rounds} by σ={list(sig_cands)}: {nacks} "
              f"→ σ*={sig_cands[i]}", flush=True)
    return sigstar, nack_curves


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut1-snr", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--lut2-snr", type=float, nargs="+",
                    default=[0.3, 0.45, 0.6, 0.75, 0.9])
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.15, 0.25, 0.35, 0.45, 0.6])
    ap.add_argument("--lut1-batch", type=int, default=128)
    ap.add_argument("--lut1-rounds", type=int, default=2)
    ap.add_argument("--lut2-batch", type=int, default=64)
    ap.add_argument("--lut2-rounds", type=int, default=3)
    a = ap.parse_args()
    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=NUM_BPS)

    print("== LUT1: pure-BP steady-state syndrome w̄(SNR) ==")
    w1 = build_lut1(ldpc, a.lut1_snr, a.lut1_batch, a.lut1_rounds)
    print("== LUT2: σ*(SNR) by end-to-end NACK ==")
    s2, ncur = build_lut2(ldpc, a.lut2_snr, a.sigmas, a.lut2_batch, a.lut2_rounds)

    w1 = np.array(w1); snr1 = np.array(a.lut1_snr)
    s2 = np.array(s2); snr2 = np.array(a.lut2_snr)
    dec_mono = bool(np.all(np.diff(w1) <= 1e-6))
    print(f"\nLUT1 monotone-decreasing: {dec_mono}  (w {w1.min():.4f}..{w1.max():.4f})")

    # compose σ(w): ratio bins over observed w range; each bin → SNR (LUT1 inv) → σ* (LUT2)
    order = np.argsort(w1); w_s, snr_s = w1[order], snr1[order]
    edges = np.round(np.linspace(w1.min(), w1.max(), 6), 4)
    edges[0] = 0.0
    thresholds = [float(x) for x in edges[1:-1]]
    reps = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    sigma_levels = []
    for wr in reps:
        snr_eff = float(np.interp(wr, w_s, snr_s))
        sigma_levels.append(round(float(np.interp(snr_eff, snr2, s2)), 4))
    for i in range(1, len(sigma_levels)):                 # monotone: higher w → σ ≥
        sigma_levels[i] = max(sigma_levels[i], sigma_levels[i - 1])
    # bins are ordered by increasing w; scheduler maps low ratio→sigma_levels[0].
    # Higher w should give higher σ, so reverse-map: sigma_levels already ascending in w.
    print(f"composed σ(w): thresholds={thresholds}  sigma_levels={sigma_levels}")

    os.makedirs("results", exist_ok=True)
    out = {"schema_version": 2, "method": "two_stage_snr_mediated_nack",
           "per_chunk": False, "thresholds": thresholds, "sigma_levels": sigma_levels,
           "sigma_min": float(min(a.sigmas)), "sigma_max": float(max(a.sigmas)),
           "monotonic_sigma": True, "syndrome_hard_decision": DEFAULT_SYNDROME_HARD_DECISION,
           "lut1": {"snr": list(map(float, snr1)), "w_bar": list(map(float, w1))},
           "lut2": {"snr": list(map(float, snr2)), "sigma_star": list(map(float, s2)),
                    "sigma_candidates": a.sigmas, "nack_curves": ncur},
           "per_chunk_lookup": None}
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"wrote {OUT_JSON}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(snr1, w1, "o-"); ax[0].set_xlabel("SNR (Eb/N0 dB)"); ax[0].set_ylabel("w̄_ss")
    ax[0].set_title("LUT1: syndrome w̄(SNR) (pure BP)"); ax[0].grid(alpha=.3)
    ax[1].plot(snr2, s2, "s-", color="tab:red"); ax[1].set_xlabel("SNR (Eb/N0 dB)")
    ax[1].set_ylabel("σ* (min NACK)"); ax[1].set_title("LUT2: σ*(SNR) end-to-end"); ax[1].grid(alpha=.3)
    ax[2].step(list(edges), [sigma_levels[0]] + sigma_levels, where="post", color="tab:green", lw=2)
    ax[2].set_xlabel("syndrome ratio w"); ax[2].set_ylabel("σ")
    ax[2].set_title("composed σ(w) = LUT2(LUT1(w))"); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT_PLOT, dpi=120); print(f"saved {OUT_PLOT}")


if __name__ == "__main__":
    main()
