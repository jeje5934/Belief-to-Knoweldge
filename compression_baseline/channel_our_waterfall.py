"""
Waterfall for the denoiser-in-the-loop system (our best), measured fresh at the
same Es/N0 grid as the separation baseline, so the four curves overlay on one
axis.  Reproduces the pure-EP_practical final-table configs EXACTLY:
  legacy [5]x20 : ep_mode=False, source_input_mode="bp_post", sigma=0.3 fixed,
                  alpha=0.1, beta=0.1   (budget-100)
  EP     [5]x20 : ep_mode=True,  ep_update="damped_ep", ep_source_power=0.01,
                  adaptive_sigma=True,  sigma=0.3 load  (budget-100)

pure-EP_practical code is READ-ONLY: we import decoder.LDPC5GDecoder_soft from a
git worktree checkout (path via EP_RO env / --ep_ro) and only call it.

GPU recipe (docs/COMPUTE_LESSONS §5): TF on CPU, torch denoiser on GPU (this
process mixes TF + torch, so TF must not touch CUDA).

Drive by Es/N0 directly: no = 10^(-Es/N0/10)  (unit-energy BPSK, Es=1) -- the
same `no` the baseline uses, i.e. identical total energy / channel.
"""
import argparse
import json
import math
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")          # TF on CPU (torch keeps GPU)
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.channel.awgn import AWGN

K = 6272
NUM_BPS = 1
SEED = 42
CKPT = "/home/LJH/onlyextrinsic_ada_sigma/checkpoints/denoiser.pt"
OUT = "/home/LJH/onlyextrinsic_ada_sigma/compression_baseline/results/our_waterfall.json"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0., 0., 0.)
    p = k / n; z2 = z * z; d = 1 + z2 / n; c = (p + z2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (p, max(0., c - h), min(1., c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_ro", default=os.environ.get("EP_RO", ""),
                    help="path to read-only pure-EP_practical worktree")
    ap.add_argument("--esno", type=float, nargs="+",
                    default=[-2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=50)
    args = ap.parse_args()

    assert args.ep_ro and os.path.isfile(os.path.join(args.ep_ro, "decoder.py")), \
        "need --ep_ro pointing at a pure-EP_practical checkout"
    sys.path.insert(0, args.ep_ro)
    from decoder import LDPC5GDecoder_soft          # read-only import
    print(f"imported decoder from {args.ep_ro}  device={DEV}")

    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    ldpc = LDPC5GEncoder(K + crc.crc_length, 12600, num_bits_per_symbol=NUM_BPS)
    mp = Mapper("pam", num_bits_per_symbol=NUM_BPS)
    dm = Demapper("app", "pam", num_bits_per_symbol=NUM_BPS)
    aw = AWGN()

    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    imgs = np.array([np.array(im) for im, _ in ds], dtype=np.uint8)
    bank = tf.constant(np.unpackbits(imgs.reshape(-1, 784).astype(np.uint8), axis=1),
                       tf.int32)

    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)

    def soft(**kw):
        d = LDPC5GDecoder_soft(ldpc, k_payload=K, num_iter=100,
                               denoiser_kwargs=dict(device=DEV), **kw, **common)
        d.denoiser.load_weights_pt(CKPT); d.denoiser.sigma = 0.3
        return d

    lg100 = soft(bp_schedule=[5] * 20, ep_mode=False,
                 source_input_mode="bp_post", adaptive_sigma=False)
    lg100.alpha = 0.1; lg100.beta = 0.1
    ep100 = soft(bp_schedule=[5] * 20, ep_mode=True, ep_update="damped_ep",
                 ep_source_power=0.01, ep_code_power=1.0, adaptive_sigma=True)

    rows = [("legacy_[5]x20", lg100), ("EP_[5]x20", ep100)]

    def run(dec, no, esno):
        tn = 0; tot = args.batch * args.rounds
        no_t = tf.constant(no, tf.float32)
        for r in range(args.rounds):
            tf.random.set_seed(SEED + r + int(round(esno * 1000)))
            idx = tf.random.uniform([args.batch], 0, tf.shape(bank)[0], dtype=tf.int32)
            u = tf.gather(bank, idx)
            y = aw(mp(ldpc(crc(tf.cast(u, ldpc.rdtype)))), no_t)
            hat = dec(dm(y, no_t))
            _, cv = crcd(hat)
            tn += args.batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        p, lo, hi = wilson(tn, tot)
        return {"bler": p, "ci_lo": lo, "ci_hi": hi, "nack": tn, "total": tot}

    results = {}
    for esno in args.esno:
        no = float(10.0 ** (-esno / 10.0))
        print(f"===== Es/N0={esno:+.3f} dB  no={no:.5f} "
              f"(raw Eb/N0={esno - 10*math.log10(ldpc.coderate):.3f}) =====", flush=True)
        for label, dec in rows:
            r = run(dec, no, esno)
            results[f"{label}@{esno}"] = {"label": label, "esno": esno, "no": no, **r}
            print(f"  {label:16} BLER={r['bler']:.4f} "
                  f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  ({r['nack']}/{r['total']})",
                  flush=True)
            json.dump(results, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
