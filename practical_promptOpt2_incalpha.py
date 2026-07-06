"""
Prompt Opt-2 — INCREASING α_ep schedule (low early, high late).

Opt-1 flipped the expectation: harm is early over-injection, not late.  Theory:
early BP cavity is poorly converged → denoiser sees a bad, more-multimodal cavity
→ worse correlated hallucination → high early α injects the worst poison.  Late,
the cavity is good → denoiser trustworthy.  So α should RISE with chunk index:
distrust the source early (low α), exploit it late (high α).  Constant α=0.02
over-suppresses the reliable late information.

Anchor: const α=0.02 → BLER 0.126 [.115,.138] at 0.6 dB; BP-100 ceiling 0.245.
[2]×15, sigma_post=3.0, adaptive σ, 14 source-injection chunks (NSRC=14).

Configs:
  const_0.02/0.03/0.05     : constant controls (separate "increasing" from
                             "just higher constant / higher late α")
  inc_a-b                  : linear α from a→b over the 14 chunks (start/end grid)
  step_.01x7_.03x7         : low-early / high-late step
  inc_0-.03                : near-zero start, ramp to 0.03
  earlyoff3_.01-.04        : first 3 chunks α=0 (source ~off), then ramp .01→.04

Key: does an increasing schedule beat const 0.02?  per-round BER should show
early BER staying stable (less early poison) and a faster late descent (more late
gain) — a direct test of the theory.

CPU-only.  0.6 dB, Wilson CI, 3200 cw/config.

Usage:
  CUDA_VISIBLE_DEVICES="" python practical_promptOpt2_incalpha.py \
      --ebno 0.6 --batch 64 --rounds 50
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
EP_SCHEDULE = [2] * 15
NSRC = len(EP_SCHEDULE) - 1
BP_BUDGET = sum(EP_SCHEDULE)
CKPT = "checkpoints/denoiser.pt"
SEED = 42
ANCHOR = 0.126


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; z2 = z * z; d = 1 + z2 / n
    c = (p + z2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (p, max(0.0, c - h), min(1.0, c + h))


def lin(a, b):
    return [float(x) for x in np.linspace(a, b, NSRC)]


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


def run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank, ebno, batch, rounds,
        track=False):
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
        tot_be += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())
        if track and getattr(dec, "last_payload_hist", None):
            step = [float(tf.reduce_mean(tf.cast(tf.not_equal(
                u, tf.cast(lg[:, :K_PAYLOAD] > 0, tf.int32)), tf.float64)).numpy())
                for lg in dec.last_payload_hist]
            pr = step if pr is None else [a + b for a, b in zip(pr, step)]
    p, lo, hi = wilson(tot_nack, total)
    out = {"bler": p, "ci_lo": lo, "ci_hi": hi,
           "ber": tot_be / (total * K_PAYLOAD), "nack": tot_nack, "total": total}
    if pr is not None:
        out["per_round_ber"] = [v / rounds for v in pr]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebno", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--out-prefix", default="results/promptOpt2_incalpha")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    crc_enc, crc_dec, ldpc, mp, dm, awgn = make_common()
    bank = bitbank()
    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    bp30 = LDPC5GDecoder(ldpc, num_iter=BP_BUDGET, **common)
    bp100 = LDPC5GDecoder(ldpc, num_iter=100, **common)
    ep = LDPC5GDecoder_soft(ldpc, bp_schedule=EP_SCHEDULE, k_payload=K_PAYLOAD,
                            ep_mode=True, ep_update="damped_ep", ep_code_power=1.0,
                            adaptive_sigma=True, ep_track_payload_hist=True,
                            num_iter=BP_BUDGET, **common)
    ep.denoiser.load_weights_pt(CKPT)

    # (label, kind, alpha_scalar, alpha_sched)
    CFG = [
        ("bp30",  "bp",   None, None),
        ("bp100", "bp100", None, None),
        ("const_0.02", "ep", 0.02, None),
        ("const_0.03", "ep", 0.03, None),
        ("const_0.05", "ep", 0.05, None),
        ("inc_.005-.03", "ep", None, lin(0.005, 0.03)),
        ("inc_.005-.05", "ep", None, lin(0.005, 0.05)),
        ("inc_.01-.03",  "ep", None, lin(0.01, 0.03)),
        ("inc_.01-.04",  "ep", None, lin(0.01, 0.04)),
        ("inc_0-.03",    "ep", None, lin(0.0, 0.03)),
        ("step_.01x7_.03x7", "ep", None, [0.01]*7 + [0.03]*7),
        ("earlyoff3_.01-.04", "ep", None, [0.0]*3 + lin(0.01, 0.04)[3:]),
    ]

    print(f"Opt-2 increasing α @ {args.ebno} dB, {args.batch}×{args.rounds}"
          f"={args.batch*args.rounds} cw, anchor const_0.02≈{ANCHOR}\n", flush=True)
    decs = {"bp": bp30, "bp100": bp100, "ep": ep}
    results = {}
    csv_f = open(args.out_prefix + ".csv", "w", newline="")
    cw = csv.writer(csv_f)
    cw.writerow(["label", "bler", "ci_lo", "ci_hi", "ber", "nack", "total", "alpha_sched"])
    csv_f.flush()
    for label, kind, a_sc, a_sched in CFG:
        dec = decs["bp" if kind == "bp" else ("bp100" if kind == "bp100" else "ep")]
        track = False
        if kind == "ep":
            ep.ep_source_power = a_sc if a_sc is not None else 0.02
            ep.ep_source_power_schedule = a_sched
            ep.ep_source_off_tail = 0
            track = True
        r = run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank,
                args.ebno, args.batch, args.rounds, track=track)
        results[label] = {"alpha_sched": a_sched, **r}
        mark = ""
        if kind == "ep" and label != "const_0.02":
            mark = f"  Δvs0.02={r['bler']-ANCHOR:+.4f}"
        print(f"  {label:20} BLER={r['bler']:.4f} "
              f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  BER={r['ber']:.4f} "
              f"({r['nack']}/{r['total']}){mark}", flush=True)
        if "per_round_ber" in r:
            pr = r["per_round_ber"]
            print(f"                       per-round BER: c0={pr[0]:.4f} "
                  f"c1={pr[1]:.4f} c7={pr[7]:.4f} c14={pr[-1]:.4f}", flush=True)
        cw.writerow([label, f"{r['bler']:.6g}", f"{r['ci_lo']:.6g}",
                     f"{r['ci_hi']:.6g}", f"{r['ber']:.6g}", r["nack"], r["total"],
                     a_sched])
        csv_f.flush()
    csv_f.close()
    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"ebno": args.ebno, "schedule": EP_SCHEDULE, "nsrc": NSRC,
                   "anchor_bler": ANCHOR, "results": results}, f, indent=2)

    base = results["const_0.02"]["bler"]
    eps = {k: v for k, v in results.items() if k not in ("bp30", "bp100")}
    best = min(eps.items(), key=lambda kv: kv[1]["bler"])
    print(f"\n--- vs const_0.02 (BLER {base:.4f}) ---")
    print(f"  best: {best[0]} BLER {best[1]['bler']:.4f} "
          f"[{best[1]['ci_lo']:.4f},{best[1]['ci_hi']:.4f}] Δ={best[1]['bler']-base:+.4f}")
    print(f"  BP-30={results['bp30']['bler']:.4f}  BP-100={results['bp100']['bler']:.4f}")
    print(f"saved: {args.out_prefix}.csv/.json")


if __name__ == "__main__":
    main()
