"""
Prompt D, reduced steps 5-6 — the single damped-path comparison.

full EP is BLER 1.0 (steps 2-4) so its performance curve is meaningless and is
excluded.  In the DAMPED path (α_ep=0.02, [2]×15, adaptive σ), one comparison on
Trouser images at 0.6 dB:

  C_repro : Trouser denoiser, sigma_post=3.0, damped α=0.02  (reproduce prompt C
            combo C, which was BLER 0.386)
  D_tw    : Trouser denoiser, diagonal Tweedie ON, damped α=0.02

Judge only whether the diagonal Tweedie (INPUT-DEPENDENT per-pixel posterior
variance) beats the fixed sigma_post=3.0 in the damped path.  (Prompt C's ε was a
GLOBAL error scale and hurt; diagonal Tweedie is a different axis, so this
comparison is still meaningful.)  BP-30 (same budget) and BP-100 (ceiling) are
reference lines.

CPU-only default (stable; Tweedie adds ~5× net forwards).

Usage:
  CUDA_VISIBLE_DEVICES="" python practical_promptD_damped.py --ebno 0.6 \
      --batch 64 --rounds 16
"""
import os
import argparse
import json
import csv
import math

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception as _e:
        print(f"[GPU] set_memory_growth skipped: {_e}")

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
TROUSER_LABEL = 1
EP_SCHEDULE = [2] * 15
BP_BUDGET = sum(EP_SCHEDULE)
CKPT_TROUSER = "checkpoints/denoiser_trouser.pt"
SEED = 42
PROMPTC_C_REF = 0.386     # prompt C combo C (Trouser, sp=3.0, damped α=0.02) @0.6dB


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; z2 = z * z; d = 1 + z2 / n
    c = (p + z2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (p, max(0.0, c - h), min(1.0, c + h))


def trouser_bank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=False,
                                           download=True)
    imgs = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    imgs = imgs[np.array(ds.targets) == TROUSER_LABEL]
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
    ap.add_argument("--ebno", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--out-prefix", default="results/promptD_damped")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    crc_enc, crc_dec, ldpc, mp, dm, awgn = make_common()
    bank = trouser_bank()
    print(f"Trouser images={int(tf.shape(bank)[0])}  Eb/N0={args.ebno} dB  "
          f"{args.batch}×{args.rounds}={args.batch*args.rounds} cw\n")

    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    bp30 = LDPC5GDecoder(ldpc, num_iter=BP_BUDGET, **common)
    bp100 = LDPC5GDecoder(ldpc, num_iter=100, **common)
    ep = LDPC5GDecoder_soft(ldpc, bp_schedule=EP_SCHEDULE, k_payload=K_PAYLOAD,
                            ep_mode=True, ep_update="damped_ep", ep_source_power=0.02,
                            ep_code_power=1.0, adaptive_sigma=True,
                            num_iter=BP_BUDGET, **common)
    ep.denoiser.load_weights_pt(CKPT_TROUSER)

    results = {}
    # (label, decoder, tweedie, note)
    specs = [
        ("bp30",   bp30,  None,  "BP-30 (same budget)"),
        ("bp100",  bp100, None,  "BP-100 (ceiling)"),
        ("C_repro", ep,   False, "Trouser + sigma_post=3.0 + damped α=0.02"),
        ("D_tw",    ep,   True,  "Trouser + diagonal Tweedie + damped α=0.02"),
    ]
    for label, dec, tw, note in specs:
        if tw is not None:
            dec.denoiser.tweedie_precision = bool(tw)
        r = run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank,
                args.ebno, args.batch, args.rounds)
        results[label] = {"note": note, "tweedie": tw, **r}
        print(f"[{label:8}] {note}")
        print(f"           BLER={r['bler']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]"
              f"  BER={r['ber']:.4f}  ({r['nack']}/{r['total']})", flush=True)

    c = results["C_repro"]["bler"]; d = results["D_tw"]["bler"]
    print(f"\n--- verdict: diagonal Tweedie vs sigma_post=3.0 (damped α=0.02) ---")
    print(f"  sigma_post=3.0 : BLER={c:.4f}  (prompt-C combo C ref={PROMPTC_C_REF})")
    print(f"  diag Tweedie   : BLER={d:.4f}")
    print(f"  ΔBLER (D−C)    : {d - c:+.4f}  "
          f"=> Tweedie {'BETTER' if d < c else ('WORSE' if d > c else 'equal')} "
          f"(CIs {'overlap' if not (results['D_tw']['ci_hi'] < results['C_repro']['ci_lo'] or results['C_repro']['ci_hi'] < results['D_tw']['ci_lo']) else 'disjoint'})")

    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"ebno": args.ebno, "promptC_C_ref": PROMPTC_C_REF,
                   "results": results}, f, indent=2)
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "note", "tweedie", "bler", "ci_lo", "ci_hi",
                    "ber", "nack", "total"])
        for k, v in results.items():
            w.writerow([k, v["note"], v["tweedie"], f"{v['bler']:.6g}",
                        f"{v['ci_lo']:.6g}", f"{v['ci_hi']:.6g}",
                        f"{v['ber']:.6g}", v["nack"], v["total"]])
    print(f"\nsaved: {args.out_prefix}.csv/.json")


if __name__ == "__main__":
    main()
