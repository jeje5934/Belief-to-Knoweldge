"""
Discriminate H1 vs H2 for the source-purity degradation (D=minus 0.114 ≫
C=bp_post 0.055 at β=0).

H1 (state-dependent self-damping): the contaminated input (BP_post, with the
    source's own prior claim) makes the denoiser emit only a DIFFERENCE — a
    directional damping.  Purifying removes it → more aggressive source; a scalar
    α cannot compensate.  Prediction: minus never reaches C for any α.
H2 (input-distribution shift): the purified input (P − src_feedback) is less
    image-like (source contribution removed) → OOD → the score model emits a
    WEAKER (smaller) output → less effective injection.  A magnitude problem, so
    α CAN compensate.  Prediction: larger α brings minus toward C; and minus's
    |src_ext| is systematically SMALLER than bp_post's (esp. late chunks).

Experiments (β=0, [2]×15, σ_in=0.3, sigma_post=3.0, 0.6 dB, 1024 cw):
  #1 α sweep (minus): α ∈ {0.1(=D), 0.2, 0.3, 0.5} vs C (bp_post, α=0.1) & BP-100.
  #3 direct: per-chunk mean|src_ext| for bp_post vs minus on the SAME batches.

Verdict is data-driven; if #1 and #3 disagree, report both.  No commit.
GPU recipe (docs/COMPUTE_LESSONS §5): TF on CPU, torch denoiser on GPU.
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
tf.config.set_visible_devices([], "GPU")     # TF on CPU
import torch
DENOISER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
C_REF = 0.055     # bp_post, β=0, α=0.1 (source_purity)


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


def build(ldpc, mode):
    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    d = LDPC5GDecoder_soft(ldpc, bp_schedule=[2]*15, k_payload=K_PAYLOAD,
                           ep_mode=False, source_input_mode=mode, adaptive_sigma=False,
                           ep_track_payload_hist=True, num_iter=30,
                           denoiser_kwargs=dict(device=DENOISER_DEVICE), **common)
    d.denoiser.load_weights_pt(CKPT); d.denoiser.sigma = FIXED_SIGMA
    return d


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
        tot_be += int(tf.reduce_sum(tf.cast(tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)), tf.int32)).numpy())
    p, lo, hi = wilson(tot_nack, total)
    return {"bler": p, "ci_lo": lo, "ci_hi": hi, "ber": tot_be/(total*K_PAYLOAD), "nack": tot_nack, "total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebno", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--out-prefix", default="results/purity_discriminate")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    crc_enc, crc_dec, ldpc, mp, dm, awgn = make_common()
    bank = bitbank()
    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    bp100 = LDPC5GDecoder(ldpc, num_iter=100, **common)
    bp = build(ldpc, "bp_post"); mn = build(ldpc, "minus_source_feedback")
    E = args.ebno
    results = {}

    print(f"H1/H2 discriminate @ {E} dB, {args.batch}×{args.rounds}={args.batch*args.rounds} cw, "
          f"denoiser={DENOISER_DEVICE}, C_ref(bp_post β0 α.1)≈{C_REF}\n", flush=True)

    # #3 direct diagnostic: per-chunk mean|src_ext|, bp_post vs minus, same batches
    print("=== #3 direct: per-chunk mean|src_ext| (bp_post vs minus, β=0, α=0.1) ===", flush=True)
    bp.alpha = 0.1; bp.beta = 0.0; mn.alpha = 0.1; mn.beta = 0.0
    no = ebnodb2no(E, NUM_BPS, ldpc.coderate)
    mag_bp = None; mag_mn = None; nb = 4
    for r in range(nb):
        tf.random.set_seed(SEED + r + int(E*1000))
        idx = tf.random.uniform([args.batch], 0, tf.shape(bank)[0], dtype=tf.int32)
        u = tf.gather(bank, idx)
        y = awgn(mp(ldpc(crc_enc(tf.cast(u, ldpc.rdtype)))), no); llr = dm(y, no)
        _ = bp(llr); _ = mn(llr)
        a = np.array(bp.last_src_ext_mag); b = np.array(mn.last_src_ext_mag)
        mag_bp = a if mag_bp is None else mag_bp + a
        mag_mn = b if mag_mn is None else mag_mn + b
    mag_bp /= nb; mag_mn /= nb
    ratio = (mag_mn / np.maximum(mag_bp, 1e-9))
    print("  chunk :   " + " ".join(f"{i:5d}" for i in range(len(mag_bp))), flush=True)
    print("  bp_post:  " + " ".join(f"{v:5.2f}" for v in mag_bp), flush=True)
    print("  minus  :  " + " ".join(f"{v:5.2f}" for v in mag_mn), flush=True)
    print("  minus/bp: " + " ".join(f"{v:5.2f}" for v in ratio), flush=True)
    print(f"  overall mean|src_ext|: bp_post={mag_bp.mean():.3f}  minus={mag_mn.mean():.3f}  "
          f"ratio={mag_mn.mean()/mag_bp.mean():.3f}\n", flush=True)
    results["src_ext_diag"] = {"bp_post": mag_bp.tolist(), "minus": mag_mn.tolist(),
                               "ratio": ratio.tolist()}

    # #1 α sweep (minus, β=0) + references
    print("=== #1 α sweep (minus, β=0): does larger α reach C? ===", flush=True)
    csv_f = open(args.out_prefix + ".csv", "w", newline="")
    cw = csv.writer(csv_f); cw.writerow(["label","mode","alpha","bler","ci_lo","ci_hi","ber","nack","total"]); csv_f.flush()
    def rec(label, dec, mode, a):
        r = run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank, E, args.batch, args.rounds)
        results[label] = {"mode": mode, "alpha": a, **r}
        cw.writerow([label, mode, a, f"{r['bler']:.6g}", f"{r['ci_lo']:.6g}", f"{r['ci_hi']:.6g}", f"{r['ber']:.6g}", r["nack"], r["total"]]); csv_f.flush()
        print(f"  {label:16} BLER={r['bler']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] BER={r['ber']:.4f} ({r['nack']}/{r['total']})", flush=True)
        return r
    bp.alpha = 0.1; bp.beta = 0.0
    C = rec("C_bp_post_a.1", bp, "bp_post", 0.1)
    rec("BP-100", bp100, "-", None)
    for a in [0.1, 0.2, 0.3, 0.5]:
        mn.alpha = a; mn.beta = 0.0
        rec(f"minus_a{a}", mn, "minus", a)
    csv_f.close()
    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"ebno": E, "C_ref": C_REF, "results": results}, f, indent=2)

    # verdict
    print(f"\n--- verdict ---", flush=True)
    Cb = C["bler"]; Clo, Chi = C["ci_lo"], C["ci_hi"]
    best_minus = min([(results[f"minus_a{a}"]["bler"], a) for a in [0.1,0.2,0.3,0.5]])
    bm_lo = results[f"minus_a{best_minus[1]}"]["ci_lo"]
    reaches = bm_lo <= Chi
    print(f"  C (bp_post) = {Cb:.4f} [{Clo:.4f},{Chi:.4f}]")
    print(f"  best minus  = {best_minus[0]:.4f} @ α={best_minus[1]}")
    print(f"  minus reaches C (CI overlap)? {'YES → H2 (magnitude, α-compensable)' if reaches else 'NO → H1 (directional self-damping)'}")
    print(f"  |src_ext| minus/bp overall ratio = {mag_mn.mean()/mag_bp.mean():.3f} "
          f"({'minus SMALLER → supports H2' if mag_mn.mean()<mag_bp.mean()*0.9 else 'not clearly smaller → not H2-magnitude'})")
    print(f"saved: {args.out_prefix}.csv/.json")


if __name__ == "__main__":
    main()
