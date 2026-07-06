"""
Turbo ↔ damped-EP final comparison — same branch, two modes (ep_mode False/True),
same channel/code/harness, ONLY the update rule differs.  No logic change; just
run both modes side by side.  Validates docs/EP_RESEARCH_SUMMARY §A (turbo
extrinsic = incomplete cavity; α = site damping) and §G (turbo↔EP equivalence).

Main comparison (b) — SAME schedule, SAME fixed σ=0.3, update only:
  turbo [2]×15 : ep_mode=False, α=β=0.1  (additive: new_input = channel + β·bp_ext + α·src_ext)
  EP    [2]×15 : ep_mode=True,  damped α_ep=0.02  (cavity/site: src_site EMA)
  (+ both on [5]×6 at 0.6 dB — is the conclusion schedule-independent?)
Secondary comparison (a) — each at its own best (adaptive σ):
  turbo_best : ep_mode=False, α=β=0.1, [2]×15, adaptive σ
  EP_best    : ep_mode=True,  damped α=0.02, [2]×15, adaptive σ
Baselines: BP-30 (same budget), BP-100 (ceiling).

0.5/0.6/0.7 dB, 3200 cw, Wilson CI, BLER + BER.

Usage:
  CUDA_VISIBLE_DEVICES="" python practical_turbo_vs_ep.py --batch 64 --rounds 50
"""
import os
import argparse
import json
import csv
import math

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no

from decoder import LDPC5GDecoder_soft

IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP
N_CODEWORD = 12600
NUM_BPS = 1
BP_BUDGET = 30
CKPT = "checkpoints/denoiser.pt"
FIXED_SIGMA = 0.3
SEED = 42


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; z2 = z * z; d = 1 + z2 / n
    c = (p + z2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (p, max(0.0, c - h), min(1.0, c + h))


def bitbank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=False,
                                           download=True)
    imgs = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = imgs.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    return tf.constant(np.unpackbits(flat, axis=1), dtype=tf.int32)


def make_common():
    crc_enc = CRCEncoder("CRC24A"); crc_dec = CRCDecoder(crc_enc)
    ldpc = LDPC5GEncoder(K_PAYLOAD + crc_enc.crc_length, N_CODEWORD,
                         num_bits_per_symbol=NUM_BPS)
    mp = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    dm = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    return crc_enc, crc_dec, ldpc, mp, dm, AWGN()


def run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank, ebno, batch, rounds):
    no = ebnodb2no(ebno, NUM_BPS, ldpc.coderate)
    nimg = tf.shape(bank)[0]
    tot_nack = tot_be = 0; total = batch * rounds
    for r in range(rounds):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        idx = tf.random.uniform([batch], 0, nimg, dtype=tf.int32)
        u = tf.gather(bank, idx)
        y = awgn(mp(ldpc(crc_enc(tf.cast(u, ldpc.rdtype)))), no)
        hat = dec(dm(y, no))
        _, cv = crc_dec(hat)
        tot_nack += batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_be += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())
    p, lo, hi = wilson(tot_nack, total)
    return {"bler": p, "ci_lo": lo, "ci_hi": hi,
            "ber": tot_be / (total * K_PAYLOAD), "nack": tot_nack, "total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--out-prefix", default="results/turbo_vs_ep")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    crc_enc, crc_dec, ldpc, mp, dm, awgn = make_common()
    bank = bitbank()
    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    bp30 = LDPC5GDecoder(ldpc, num_iter=BP_BUDGET, **common)
    bp100 = LDPC5GDecoder(ldpc, num_iter=100, **common)

    def soft(ep_mode, adaptive):
        d = LDPC5GDecoder_soft(ldpc, bp_schedule=[2]*15, k_payload=K_PAYLOAD,
                               ep_mode=ep_mode, ep_update="damped_ep",
                               ep_code_power=1.0, adaptive_sigma=adaptive,
                               num_iter=BP_BUDGET, **common)
        d.denoiser.load_weights_pt(CKPT)
        d.denoiser.sigma = FIXED_SIGMA
        return d
    turbo_fix = soft(False, False)   # fixed σ=0.3
    ep_fix    = soft(True,  False)
    turbo_ad  = soft(False, True)    # adaptive σ (each-best)
    ep_ad     = soft(True,  True)

    ALL = [0.5, 0.6, 0.7]
    # (label, decoder, schedule, setter, ebnos, group)
    CFG = [
        ("BP-30",  bp30,  None,   None, ALL, "base"),
        ("BP-100", bp100, None,   None, ALL, "base"),
        # main (b): same schedule [2]×15, fixed σ=0.3, update only
        ("turbo_[2]x15_σ.3", turbo_fix, [2]*15,
         lambda d: (setattr(d, "alpha", 0.1), setattr(d, "beta", 0.1)), ALL, "b"),
        ("EP_[2]x15_σ.3",    ep_fix,    [2]*15,
         lambda d: setattr(d, "ep_source_power", 0.02), ALL, "b"),
        # main (b2): [5]×6 at 0.6 dB (schedule-independence)
        ("turbo_[5]x6_σ.3",  turbo_fix, [5]*6,
         lambda d: (setattr(d, "alpha", 0.1), setattr(d, "beta", 0.1)), [0.6], "b2"),
        ("EP_[5]x6_σ.3",     ep_fix,    [5]*6,
         lambda d: setattr(d, "ep_source_power", 0.02), [0.6], "b2"),
        # secondary (a): each best, adaptive σ
        ("turbo_best_adapt", turbo_ad, [2]*15,
         lambda d: (setattr(d, "alpha", 0.1), setattr(d, "beta", 0.1)), ALL, "a"),
        ("EP_best_adapt",    ep_ad,    [2]*15,
         lambda d: setattr(d, "ep_source_power", 0.02), ALL, "a"),
    ]

    print(f"turbo↔EP @ {args.batch}×{args.rounds}={args.batch*args.rounds} cw, "
          f"fixed σ={FIXED_SIGMA}\n", flush=True)
    results = {}
    csv_f = open(args.out_prefix + ".csv", "w", newline="")
    cw = csv.writer(csv_f)
    cw.writerow(["label", "group", "ebno", "schedule", "bler", "ci_lo", "ci_hi",
                 "ber", "nack", "total"])
    csv_f.flush()
    for label, dec, sched, setter, ebnos, group in CFG:
        for ebno in ebnos:
            if sched is not None:
                dec.bp_schedule = sched
            if setter is not None:
                setter(dec)
            r = run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank,
                    ebno, args.batch, args.rounds)
            results[f"{label}@{ebno}"] = {"label": label, "group": group,
                                          "ebno": ebno, "schedule": sched, **r}
            print(f"  [{group}] {label:18} {ebno}dB  BLER={r['bler']:.4f} "
                  f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  BER={r['ber']:.4f} "
                  f"({r['nack']}/{r['total']})", flush=True)
            cw.writerow([label, group, ebno, sched, f"{r['bler']:.6g}",
                         f"{r['ci_lo']:.6g}", f"{r['ci_hi']:.6g}",
                         f"{r['ber']:.6g}", r["nack"], r["total"]])
            csv_f.flush()
        print(flush=True)
    csv_f.close()
    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"fixed_sigma": FIXED_SIGMA, "results": results}, f, indent=2)

    # equivalence verdict for main (b) at each SNR
    print("--- main (b) turbo vs EP, [2]×15 fixed σ=0.3 (CI overlap = equivalent) ---")
    for e in ALL:
        t = results[f"turbo_[2]x15_σ.3@{e}"]; p = results[f"EP_[2]x15_σ.3@{e}"]
        overlap = not (t["ci_hi"] < p["ci_lo"] or p["ci_hi"] < t["ci_lo"])
        print(f"  {e}dB: turbo {t['bler']:.4f} [{t['ci_lo']:.4f},{t['ci_hi']:.4f}]"
              f"  vs  EP {p['bler']:.4f} [{p['ci_lo']:.4f},{p['ci_hi']:.4f}]"
              f"  → {'OVERLAP (equivalent)' if overlap else 'DISJOINT'}")
    _plot(results, ALL, args.out_prefix + ".png")
    print(f"saved: {args.out_prefix}.csv/.json/.png")


def _plot(results, ebnos, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = {"BP-30": "s--", "BP-100": "^:", "turbo_[2]x15_σ.3": "o-",
              "EP_[2]x15_σ.3": "D-", "turbo_best_adapt": "o:", "EP_best_adapt": "D:"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, st in series.items():
        xs = [e for e in ebnos if f"{name}@{e}" in results]
        bl = [results[f"{name}@{e}"]["bler"] for e in xs]
        be = [results[f"{name}@{e}"]["ber"] for e in xs]
        ax1.semilogy(xs, [max(b, 1e-4) for b in bl], st, label=name, ms=6)
        ax2.semilogy(xs, [max(b, 1e-5) for b in be], st, label=name, ms=6)
    for ax, t in ((ax1, "BLER"), (ax2, "BER")):
        ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel(t); ax.grid(True, which="both", alpha=.3)
        ax.legend(fontsize=7)
    ax1.set_title("turbo vs damped EP — BLER (same schedule, σ=0.3)")
    ax2.set_title("BER")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
