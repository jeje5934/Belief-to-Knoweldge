"""
True-turbo schedule/α search — BP budget ≤ 100.  Find the theoretical optimum of
the exact-turbo structure (fresh BP each chunk, source-free cavity, single carried
extrinsic).  true turbo is structurally weak on finely-sliced schedules ([2]×15 →
BER 0.125) because each chunk does a FRESH short BP; the turbo rhythm is "enough
BP per chunk + coarse exchange".

Stage 1: α=0.1 fixed, sweep schedules (total BP ≤ 100): iters-per-chunk × #chunks.
Stage 2: top-2 schedules × α ∈ {0.05,0.1,0.2,0.5,1.0} (α=1 = classic full-extrinsic
         turbo — does the non-accumulating structure tolerate full trust?).
References: BP-30, BP-100 (same-budget baseline & ceiling), legacy turbo-like
[2]×15 (α=β=0.1, warm-start).  0.6 dB, 1024 cw, Wilson CI, per-round BER.

Fixed σ=0.3 (adaptive_sigma off), sp=3.0.  CPU-only.  No code change / no commit.

Usage:
  CUDA_VISIBLE_DEVICES="" python practical_true_turbo_search.py --batch 64 --rounds 16
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
CKPT = "checkpoints/denoiser.pt"
FIXED_SIGMA = 0.3
SEED = 42

SCHEDULES = {  # label -> schedule (total BP)
    "10x10": [10]*10, "20x5": [20]*5, "25x4": [25]*4, "33x3": [33]*3,
    "50x2": [50]*2, "20x4": [20]*4, "10x5": [10]*5,
}
ALPHAS = [0.05, 0.1, 0.2, 0.5, 1.0]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; z2 = z * z; d = 1 + z2 / n
    c = (p + z2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (p, max(0.0, c - h), min(1.0, c + h))


def bitbank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=False, download=True)
    imgs = np.array([np.array(im) for im, _ in ds], dtype=np.uint8)
    return tf.constant(np.unpackbits(imgs.reshape(-1, IMG_H*IMG_W).astype(np.uint8), axis=1), tf.int32)


def make_common():
    crc_enc = CRCEncoder("CRC24A"); crc_dec = CRCDecoder(crc_enc)
    ldpc = LDPC5GEncoder(K_PAYLOAD + crc_enc.crc_length, N_CODEWORD, num_bits_per_symbol=NUM_BPS)
    mp = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    dm = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    return crc_enc, crc_dec, ldpc, mp, dm, AWGN()


def run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank, ebno, batch, rounds, track=False):
    no = ebnodb2no(ebno, NUM_BPS, ldpc.coderate)
    nimg = tf.shape(bank)[0]
    tot_nack = tot_be = 0; total = batch * rounds; pr = None
    for r in range(rounds):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        idx = tf.random.uniform([batch], 0, nimg, dtype=tf.int32)
        u = tf.gather(bank, idx)
        y = awgn(mp(ldpc(crc_enc(tf.cast(u, ldpc.rdtype)))), no)
        hat = dec(dm(y, no))
        _, cv = crc_dec(hat)
        tot_nack += batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_be += int(tf.reduce_sum(tf.cast(tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)), tf.int32)).numpy())
        if track and getattr(dec, "last_payload_hist", None):
            step = [float(tf.reduce_mean(tf.cast(tf.not_equal(u, tf.cast(lg[:, :K_PAYLOAD] > 0, tf.int32)), tf.float64)).numpy()) for lg in dec.last_payload_hist]
            pr = step if pr is None else [a + b for a, b in zip(pr, step)]
    p, lo, hi = wilson(tot_nack, total)
    out = {"bler": p, "ci_lo": lo, "ci_hi": hi, "ber": tot_be/(total*K_PAYLOAD),
           "nack": tot_nack, "total": total}
    if pr is not None:
        out["per_round_ber"] = [v/rounds for v in pr]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebno", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--out-prefix", default="results/true_turbo_search")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    crc_enc, crc_dec, ldpc, mp, dm, awgn = make_common()
    bank = bitbank()
    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    bp30 = LDPC5GDecoder(ldpc, num_iter=30, **common)
    bp100 = LDPC5GDecoder(ldpc, num_iter=100, **common)
    tt = LDPC5GDecoder_soft(ldpc, bp_schedule=[20]*5, k_payload=K_PAYLOAD,
                            true_turbo=True, adaptive_sigma=False,
                            ep_track_payload_hist=True, num_iter=100, **common)
    tt.denoiser.load_weights_pt(CKPT); tt.denoiser.sigma = FIXED_SIGMA
    legacy = LDPC5GDecoder_soft(ldpc, bp_schedule=[2]*15, k_payload=K_PAYLOAD,
                                ep_mode=False, adaptive_sigma=False, num_iter=30, **common)
    legacy.denoiser.load_weights_pt(CKPT); legacy.denoiser.sigma = FIXED_SIGMA

    E = args.ebno
    results = {}
    csv_f = open(args.out_prefix + ".csv", "w", newline="")
    cw = csv.writer(csv_f)
    cw.writerow(["stage", "label", "schedule", "alpha", "bp_total", "bler", "ci_lo", "ci_hi", "ber", "nack", "total"])
    csv_f.flush()

    def record(stage, label, sched, alpha, r):
        results[label] = {"stage": stage, "schedule": sched, "alpha": alpha, **r}
        cw.writerow([stage, label, sched, alpha, sum(sched) if sched else "",
                     f"{r['bler']:.6g}", f"{r['ci_lo']:.6g}", f"{r['ci_hi']:.6g}",
                     f"{r['ber']:.6g}", r["nack"], r["total"]]); csv_f.flush()
        extra = ""
        if "per_round_ber" in r:
            pr = r["per_round_ber"]; extra = f"  per-round: c0={pr[0]:.3f} c1={pr[1] if len(pr)>1 else pr[0]:.3f} clast={pr[-1]:.3f}"
        print(f"  [{stage}] {label:16} BLER={r['bler']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] BER={r['ber']:.4f} ({r['nack']}/{r['total']}){extra}", flush=True)

    print(f"true-turbo search @ {E} dB, {args.batch}×{args.rounds}={args.batch*args.rounds} cw, σ={FIXED_SIGMA}\n", flush=True)
    # references
    record("ref", "BP-30", [2]*15, None, run(bp30, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank, E, args.batch, args.rounds))
    record("ref", "BP-100", None, None, run(bp100, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank, E, args.batch, args.rounds))
    legacy.alpha = 0.1; legacy.beta = 0.1
    record("ref", "legacy_[2]x15", [2]*15, 0.1, run(legacy, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank, E, args.batch, args.rounds, track=True))
    print(flush=True)

    # stage 1: α=0.1, sweep schedules
    print("--- stage 1: α=0.1, schedule sweep ---", flush=True)
    for lbl, sched in SCHEDULES.items():
        tt.bp_schedule = sched; tt.alpha = 0.1
        record("s1", f"tt_{lbl}_a.1", sched, 0.1,
               run(tt, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank, E, args.batch, args.rounds, track=True))
    print(flush=True)

    # pick top-2 schedules by BLER
    s1 = [(k, v) for k, v in results.items() if v.get("stage") == "s1"]
    top = sorted(s1, key=lambda kv: kv[1]["bler"])[:2]
    top_scheds = [(v["schedule"],) for _, v in top]
    print(f"--- stage 2: α sweep on top-2 {[k for k,_ in top]} ---", flush=True)
    for (_, v) in top:
        sched = v["schedule"]; slbl = "x".join([str(sched[0]), str(len(sched))])
        for a in ALPHAS:
            tt.bp_schedule = sched; tt.alpha = a
            record("s2", f"tt_{slbl}_a{a}", sched, a,
                   run(tt, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank, E, args.batch, args.rounds, track=True))
    csv_f.close()
    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"ebno": E, "sigma": FIXED_SIGMA, "results": results}, f, indent=2)

    eps = {k: v for k, v in results.items() if v.get("stage") in ("s1", "s2")}
    best = min(eps.items(), key=lambda kv: kv[1]["bler"])
    print(f"\n--- verdict @ {E} dB ---", flush=True)
    print(f"  BP-30={results['BP-30']['bler']:.4f}  BP-100={results['BP-100']['bler']:.4f}  legacy_[2]x15={results['legacy_[2]x15']['bler']:.4f}")
    print(f"  true-turbo BEST: {best[0]} sched={best[1]['schedule']} α={best[1]['alpha']} "
          f"BLER={best[1]['bler']:.4f} [{best[1]['ci_lo']:.4f},{best[1]['ci_hi']:.4f}] BER={best[1]['ber']:.4f}")
    print(f"  beats BP-100? {'YES' if best[1]['bler'] < results['BP-100']['bler'] else 'no'}")
    print(f"saved: {args.out_prefix}.csv/.json")


if __name__ == "__main__":
    main()
