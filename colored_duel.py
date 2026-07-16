"""
[4] Stage-2 duel: legacy vs altproj (+ BP-only, EP controls) over AR(1) COLORED
noise + fast Rayleigh fading + imperfect CSI, with the receiver assuming WHITE
noise (mismatched by design — see channel_models.py; no whitening).

Fairness rules enforced here:
  * ONE channel instance / LLR per batch, handed to ALL four arms.
  * CRC early-stop applied EXTERNALLY and IDENTICALLY to every arm (via the
    mode-agnostic decoder.track_u_hat hook).  The CRC-ES compute saving belongs to
    CRC-ES, not to any one decoder — an arm-asymmetric harness would fake a win.
  * BLER and compute (mean BP iters to capture) are reported as SEPARATE axes.

Arms ([5]x20, warm-on):
  bp      : altproj with delta=0  (bit-exact == continuous BP-100; verified)
  legacy  : ep_mode=False, source_input_mode="bp_post", alpha=beta (default .15)
  altproj : channel anchor + accumulated correction (delta, rho)
  ep      : ep_mode=True, damped_ep, ep_source_power=alpha_ep (fixed control)

Usage:
  python colored_duel.py --ar-coeff 0.5 --sigma-e2 0.1 --ebno 2 3 4 --cw 512
"""
import os, json, time, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")          # TF -> CPU (safe)

from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.utils import ebnodb2no
from decoder import LDPC5GDecoder_soft
from channel_models import ChannelModel, QPSK_FADING

IMG_H = IMG_W = 28; BPP = 8
K = IMG_H * IMG_W * BPP; N = 12600
CKPT = "checkpoints/denoiser.pt"
DEV = "cuda"
SCHEDULE = [5] * 20
BATCH = 128
SEED = 4242
COMMON = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
              hard_out=False, return_infobits=True, llr_max=30.0)


def bank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    flat = ds.data.numpy().astype(np.uint8).reshape(-1, IMG_H * IMG_W)
    return tf.constant(np.unpackbits(flat, axis=1), tf.int32)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (p, max(0.0, c - h), min(1.0, c + h))


def _mk(ldpc, **kw):
    d = LDPC5GDecoder_soft(ldpc, k_payload=K, num_iter=sum(SCHEDULE),
                           bp_schedule=SCHEDULE, adaptive_sigma=False,
                           denoiser_kwargs=dict(device=DEV), **kw, **COMMON)
    d.denoiser.load_weights_pt(CKPT)
    d.denoiser.sigma = 0.3
    d.track_u_hat = True
    return d


def build_arms(ldpc, a):
    arms = {}
    # BP-only: altproj with delta=0 — verified bit-exact to continuous BP-100.
    bp = _mk(ldpc, altproj=True, altproj_delta=0.0, altproj_rho=1.0,
             altproj_warm_start=True)
    arms["bp"] = bp
    lg = _mk(ldpc, ep_mode=False, source_input_mode="bp_post")
    lg.alpha = a.alpha_legacy; lg.beta = a.beta_legacy
    arms["legacy"] = lg
    ap = _mk(ldpc, altproj=True, altproj_delta=a.delta, altproj_rho=a.rho,
             altproj_sigma_den=0.3, altproj_warm_start=True)
    ap.denoiser.sigma_post = a.sigma_post
    arms["altproj"] = ap
    ep = _mk(ldpc, ep_mode=True, ep_update="damped_ep",
             ep_source_power=a.alpha_ep, ep_code_power=1.0)
    arms["ep"] = ep
    return arms


def build_altproj_grid(ldpc, deltas, rhos, sigma_post=3.0):
    """[5] altproj δ/ρ re-search on the colored channel.  The AWGN optimum
    (δ=0.02, ρ=0.9) need not be the colored+imperfect optimum — precedent: legacy
    moved α 0.1 -> 0.15 for fading.  Same CRC-ES treatment as the duel."""
    arms = {}
    for d in deltas:
        for r in rhos:
            k = f"d{d}_r{r}"
            m = _mk(ldpc, altproj=True, altproj_delta=d, altproj_rho=r,
                    altproj_sigma_den=0.3, altproj_warm_start=True)
            m.denoiser.sigma_post = sigma_post
            arms[k] = m
    return arms


def crc_es(hist, crcd, ldpc):
    """External CRC early-stop, identical for every arm.

    hist: per-round list of x_hat[:, :k] (info block logits).
    Returns (captured [B] bool, bp_iters [B] float) — per codeword, the first round
    whose CRC passes is the decision; never-passing codewords consume the full
    schedule and stay failed.
    """
    B = int(hist[0].shape[0])
    captured = tf.zeros([B], tf.bool)
    iters = np.full(B, float(sum(SCHEDULE)))
    for t, uh in enumerate(hist):
        _, cv = crcd(tf.reshape(uh, [-1, ldpc.k]))
        cv = tf.reshape(tf.cast(cv, tf.bool), [-1])
        newly = tf.logical_and(cv, tf.logical_not(captured))
        nz = newly.numpy()
        iters[nz] = float(sum(SCHEDULE[:t + 1]))
        captured = tf.logical_or(captured, cv)
        if bool(tf.reduce_all(captured).numpy()):
            break
    return captured.numpy(), iters


def run_point(arms, ch, ldpc, crc, crcd, bk, ebno, cw):
    no = ebnodb2no(ebno, ch.num_bps, ldpc.coderate)
    n_imgs = tf.shape(bk)[0]
    rounds = max(1, cw // BATCH)
    acc = {k: {"nack": 0, "iters": []} for k in arms}
    tot = 0
    for r in range(rounds):
        tf.random.set_seed(SEED + r)
        idx = tf.random.uniform([BATCH], 0, n_imgs, dtype=tf.int32)
        u = tf.gather(bk, idx)
        # ONE channel realisation per batch -> identical LLR to all four arms.
        llr = ch.transmit(ldpc(crc(tf.cast(u, ldpc.rdtype))), no)
        tot += BATCH
        for name, dec in arms.items():
            dec(llr)
            cap, it = crc_es(dec.last_u_hat_hist, crcd, ldpc)
            acc[name]["nack"] += int((~cap).sum())
            acc[name]["iters"].append(float(np.mean(it)))
    out = {}
    for name in arms:
        p, lo, hi = wilson(acc[name]["nack"], tot)
        out[name] = dict(bler=p, ci_lo=lo, ci_hi=hi, nack=acc[name]["nack"],
                         cw=tot, bp_iters=float(np.mean(acc[name]["iters"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-coeff", type=float, nargs="+", default=[0.0])
    ap.add_argument("--sigma-e2", type=float, nargs="+", default=[0.1])
    ap.add_argument("--ebno", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--cw", type=int, default=512)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--sigma-post", type=float, default=3.0)
    ap.add_argument("--alpha-legacy", type=float, default=0.15)
    ap.add_argument("--beta-legacy", type=float, default=0.15)
    ap.add_argument("--alpha-ep", type=float, default=0.01)
    ap.add_argument("--scan-altproj", action="store_true",
                    help="[5] sweep altproj delta x rho instead of the 4-way duel")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    ldpc = LDPC5GEncoder(K + crc.crc_length, N, num_bits_per_symbol=2)
    bk = bank()
    if a.scan_altproj:
        arms = build_altproj_grid(ldpc, [0.02, 0.05, 0.1], [0.9, 0.7, 0.5],
                                  a.sigma_post)
    else:
        arms = build_arms(ldpc, a)
    names = list(arms)

    print(f"# colored duel: AR(1)+fading+imperfect CSI, WHITE-assumed LLR "
          f"(mismatched by design, no whitening)")
    print(f"# altproj(δ={a.delta},ρ={a.rho},σpost={a.sigma_post})  "
          f"legacy(α=β={a.alpha_legacy})  ep(α_ep={a.alpha_ep})  "
          f"[5]x20 warm-on, CRC-ES on ALL arms, cw={a.cw}")
    hdr = f"{'a':>4} {'se2':>5} {'ebno':>5} |"
    for n in names:
        hdr += f" {n:>9} {'it':>4} |"
    print(hdr)
    rows = []
    for arc in a.ar_coeff:
        ch = ChannelModel(QPSK_FADING, sigma_e2=a.sigma_e2[0] if len(a.sigma_e2) == 1 else 0.1,
                          perfect_csi=False, llr_method="B", ar_coeff=arc)
        for se2 in a.sigma_e2:
            ch.sigma_e2 = float(se2)
            ch.perfect_csi = (se2 == 0.0)
            for eb in a.ebno:
                t0 = time.time()
                res = run_point(arms, ch, ldpc, crc, crcd, bk, eb, a.cw)
                line = f"{arc:>4.1f} {se2:>5.2f} {eb:>5.1f} |"
                for n in names:
                    r = res[n]
                    line += f" {r['bler']:>9.4f} {r['bp_iters']:>4.0f} |"
                print(line + f"  {time.time()-t0:.0f}s", flush=True)
                rows.append(dict(ar=arc, sigma_e2=se2, ebno=eb, res=res,
                                 delta=a.delta, rho=a.rho, sigma_post=a.sigma_post,
                                 alpha_legacy=a.alpha_legacy, alpha_ep=a.alpha_ep))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"# wrote {a.out}")


if __name__ == "__main__":
    main()
