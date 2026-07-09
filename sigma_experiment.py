"""
practical_sigma — shared experiment runner for denoiser-σ strategies.

Fixes the current-best EP [5]×20 structure (α_ep = ep_source_power = 0.01,
damped_ep) and varies ONLY the denoiser σ strategy (target = the denoiser noise
argument σ; the pixel→bit readout sigma_post stays fixed at 3.0):

  fixed      : FixedSigmaScheduler(σ)               (baseline; σ=0.3)
  annealing  : AnnealingSigmaScheduler (open-loop, chunk-idx → σ, linear/geom)
  adaptive   : CalibratedLookupScheduler (from a 2-stage-LUT JSON; closed-loop)
  handcrafted: HandcraftedLookupScheduler (default ratio→σ lookup)
  combo      : σ_t = min(adaptive(w_t), anneal(t))  (annealing = upper envelope)

GPU recipe (docs/COMPUTE_LESSONS §5): TF on CPU, torch denoiser on GPU.
Logs BLER (Wilson CI), per-round payload BER trajectory, and the per-chunk σ
trajectory (mean σ actually used at each denoiser chunk).
"""
import math
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")          # TF on CPU
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no

from decoder import LDPC5GDecoder_soft
from syndrome_sigma_schedule import (
    FixedSigmaScheduler, AnnealingSigmaScheduler, HandcraftedLookupScheduler,
    CalibratedLookupScheduler, build_calibrated_scheduler_from_json,
)

K = 6272
NUM_BPS = 1
CKPT = "checkpoints/denoiser.pt"
SEED = 42
SCHEDULE = [5] * 20            # current-best budget-100 EP schedule
N_CHUNKS_DENOISE = len(SCHEDULE) - 1     # last chunk has no denoiser


class ComboScheduler:
    """σ_t = min(adaptive(w_t), anneal(t)): annealing gives an upper envelope,
    the syndrome lookup fine-tunes within it."""
    name = "combo"; is_adaptive = True

    def __init__(self, adaptive, anneal):
        self.adaptive = adaptive; self.anneal = anneal

    def select_sigma(self, chunk_idx, ratios):
        return tf.minimum(self.adaptive.select_sigma(chunk_idx, ratios),
                          self.anneal.select_sigma(chunk_idx, ratios))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0., 0., 0.)
    p = k / n; z2 = z * z; d = 1 + z2 / n; c = (p + z2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (p, max(0., c - h), min(1., c + h))


def make_scheduler(strategy, *, fixed_sigma=0.3, anneal=None, json_path=None):
    if strategy == "fixed":
        return None                                    # decoder uses scalar σ
    if strategy == "annealing":
        m, s0, s1 = anneal
        return AnnealingSigmaScheduler(s0, s1, N_CHUNKS_DENOISE, mode=m)
    if strategy == "handcrafted":
        return HandcraftedLookupScheduler()
    if strategy == "adaptive":
        return build_calibrated_scheduler_from_json(json_path)
    if strategy == "combo":
        m, s0, s1 = anneal
        return ComboScheduler(build_calibrated_scheduler_from_json(json_path),
                              AnnealingSigmaScheduler(s0, s1, N_CHUNKS_DENOISE, mode=m))
    raise ValueError(strategy)


def build_decoder(ldpc, strategy, *, fixed_sigma=0.3, anneal=None, json_path=None):
    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    sched = make_scheduler(strategy, fixed_sigma=fixed_sigma, anneal=anneal,
                           json_path=json_path)
    dec = LDPC5GDecoder_soft(
        ldpc, k_payload=K, num_iter=sum(SCHEDULE), bp_schedule=SCHEDULE,
        ep_mode=True, ep_update="damped_ep", ep_source_power=0.01, ep_code_power=1.0,
        denoiser_kwargs=dict(device=DEV),
        sigma_scheduler=sched, adaptive_sigma=(sched is not None), **common)
    dec.denoiser.load_weights_pt(CKPT)
    dec.denoiser.sigma = float(fixed_sigma)            # fixed-path scalar / fallback
    return dec


_BANK_CACHE = None


def _bank():
    global _BANK_CACHE
    if _BANK_CACHE is None:
        import torchvision
        ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
        imgs = ds.data.numpy().astype(np.uint8)          # [N,28,28], no PIL iteration
        _BANK_CACHE = tf.constant(
            np.unpackbits(imgs.reshape(-1, 784), axis=1), tf.int32)
    return _BANK_CACHE


def run(dec, ldpc, ebno, batch, rounds, *, track_traj=False):
    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    mp = Mapper("pam", num_bits_per_symbol=NUM_BPS)
    dm = Demapper("app", "pam", num_bits_per_symbol=NUM_BPS); aw = AWGN()
    bank = _bank()
    no = ebnodb2no(ebno, NUM_BPS, ldpc.coderate)
    if track_traj:
        dec.ep_track_payload_hist = True
    tot_nack = tot_biterr = 0
    chunk_sigma = {}          # chunk_idx -> [mean_sigma per round]
    round_ber = {}            # chunk_idx -> [BER per round]  (per-round trajectory)
    for r in range(rounds):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        idx = tf.random.uniform([batch], 0, tf.shape(bank)[0], dtype=tf.int32)
        u = tf.gather(bank, idx)
        y = aw(mp(ldpc(crc(tf.cast(u, ldpc.rdtype)))), no)
        hat = dec(dm(y, no)); _, cv = crcd(hat)
        tot_nack += batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_biterr += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K] > 0, tf.int32)), tf.int32)).numpy())
        for d in dec.last_chunk_diagnostics:
            ms = float(tf.reduce_mean(tf.cast(d["sigma"], tf.float32)).numpy())
            chunk_sigma.setdefault(int(d["chunk_idx"]), []).append(ms)
        if track_traj:
            for ci, ph in enumerate(dec.last_payload_hist):
                ber = float(tf.reduce_mean(tf.cast(
                    tf.not_equal(u, tf.cast(ph > 0, tf.int32)), tf.float32)).numpy())
                round_ber.setdefault(ci, []).append(ber)
    tot = batch * rounds
    p, lo, hi = wilson(tot_nack, tot)
    traj_sigma = {c: float(np.mean(v)) for c, v in sorted(chunk_sigma.items())}
    traj_ber = {c: float(np.mean(v)) for c, v in sorted(round_ber.items())}
    return {"bler": p, "ci": [lo, hi], "nack": tot_nack, "total": tot,
            "ber": tot_biterr / (tot * K),
            "sigma_traj": traj_sigma, "ber_traj": traj_ber}


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="fixed")
    p.add_argument("--ebno", type=float, default=0.6)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--fixed-sigma", type=float, default=0.3)
    p.add_argument("--anneal", nargs=3, default=None,
                   metavar=("MODE", "S_START", "S_END"))
    p.add_argument("--json", default=None)
    p.add_argument("--track", action="store_true")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    anneal = None if a.anneal is None else (a.anneal[0], float(a.anneal[1]), float(a.anneal[2]))
    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=NUM_BPS)
    dec = build_decoder(ldpc, a.strategy, fixed_sigma=a.fixed_sigma,
                        anneal=anneal, json_path=a.json)
    res = run(dec, ldpc, a.ebno, a.batch, a.rounds, track_traj=a.track)
    print(f"[{a.strategy}{('/'+a.tag) if a.tag else ''}] ebno={a.ebno} "
          f"BLER={res['bler']:.4f} [{res['ci'][0]:.4f},{res['ci'][1]:.4f}] "
          f"BER={res['ber']:.5f} ({res['nack']}/{res['total']})")
    print("  σ trajectory (chunk→mean σ):",
          {c: round(s, 3) for c, s in res["sigma_traj"].items()})
    if a.track:
        print("  BER trajectory (chunk→BER):",
              {c: round(b, 4) for c, b in res["ber_traj"].items()})
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/sigma_{a.strategy}{('_'+a.tag) if a.tag else ''}.json", "w"), indent=2)
