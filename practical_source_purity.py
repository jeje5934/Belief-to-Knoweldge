"""
Legacy-turbo source-input purity: does removing the EXPLICIT injected source
feedback from the denoiser input help, hurt, or do nothing?  warm-start ALWAYS
kept.  Tests the "incomplete cavity = implicit self-damping" hypothesis.

Verification (spec §4): chunk-0/1 equivalence between modes, source_in assert.
Experiment (spec §5), 0.6 dB, [2]×15, σ=0.3, warm-start:
  A: bp_post,               α=β=0.1   (legacy best repro ≈0.010)
  B: minus_source_feedback, α=β=0.1   (key comparison)
  C: bp_post,               α=0.1,β=0 (β-free baseline)
  D: minus_source_feedback, α=0.1,β=0
  + B α-sweep {0.05,0.1,0.2}.  References BP-30, BP-100.

Verdict: B≈A → explicit contamination harmless (hypothesis rejected); B>A →
purity helps; B<A → the contamination was protective (implicit self-damping).

CPU-only.  No commit.
Usage: CUDA_VISIBLE_DEVICES="" python practical_source_purity.py --batch 64 --rounds 50
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
# GPU recipe (see docs/COMPUTE_LESSONS.md §5): put the torch DENOISER on GPU
# (~16× faster/call) but keep TF on CPU — mixing TF-CUDA and torch-CUDA in one
# process segfaults on this host.  Do NOT build cpu- and cuda-denoisers in the
# same process either.  Run with CUDA_VISIBLE_DEVICES=0; falls back to CPU if
# unset.
tf.config.set_visible_devices([], "GPU")     # TF stays on CPU
import torch
DENOISER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
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


def build_legacy(ldpc, mode):
    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    d = LDPC5GDecoder_soft(ldpc, bp_schedule=[2]*15, k_payload=K_PAYLOAD,
                           ep_mode=False, source_input_mode=mode, adaptive_sigma=False,
                           ep_track_payload_hist=True, num_iter=30,
                           denoiser_kwargs=dict(device=DENOISER_DEVICE), **common)
    d.denoiser.load_weights_pt(CKPT); d.denoiser.sigma = FIXED_SIGMA
    return d


def get_llr(ldpc, crc_enc, mp, dm, awgn, bk, ebno, batch, seed):
    no = ebnodb2no(ebno, NUM_BPS, ldpc.coderate)
    tf.random.set_seed(seed)
    idx = tf.random.uniform([batch], 0, tf.shape(bk)[0], dtype=tf.int32)
    u = tf.gather(bk, idx)
    y = awgn(mp(ldpc(crc_enc(tf.cast(u, ldpc.rdtype)))), no)
    return dm(y, no), u


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
        _, cv = hard_crc_decode(crc_dec, hat)
        tot_nack += batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_be += int(tf.reduce_sum(tf.cast(tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)), tf.int32)).numpy())
        if track and getattr(dec, "last_payload_hist", None):
            step = [float(tf.reduce_mean(tf.cast(tf.not_equal(u, tf.cast(lg[:, :K_PAYLOAD] > 0, tf.int32)), tf.float64)).numpy()) for lg in dec.last_payload_hist]
            pr = step if pr is None else [a + b for a, b in zip(pr, step)]
    p, lo, hi = wilson(tot_nack, total)
    out = {"bler": p, "ci_lo": lo, "ci_hi": hi, "ber": tot_be/(total*K_PAYLOAD), "nack": tot_nack, "total": total}
    if pr is not None:
        out["per_round_ber"] = [v/rounds for v in pr]
    return out


def verify(ldpc, crc_enc, mp, dm, awgn, bk):
    print("=== verification ===", flush=True)
    llr, _ = get_llr(ldpc, crc_enc, mp, dm, awgn, bk, 0.6, 16, SEED)
    A = build_legacy(ldpc, "bp_post"); A.alpha = 0.1; A.beta = 0.1
    B = build_legacy(ldpc, "minus_source_feedback"); B.alpha = 0.1; B.beta = 0.1
    _ = A(llr); _ = B(llr)
    ha = [h.numpy() for h in A.last_payload_hist]; hb = [h.numpy() for h in B.last_payload_hist]
    d0 = float(np.max(np.abs(ha[0]-hb[0]))); d1 = float(np.max(np.abs(ha[1]-hb[1])))
    d2 = float(np.max(np.abs(ha[2]-hb[2])))
    print(f"  [1] chunk-0 payload_hist max|Δ| = {d0:.2e} → {'EQUAL ✓' if d0<1e-4 else '✗'} "
          f"(chunk-0 denoiser input identical: src_feedback=0)")
    print(f"      chunk-1 max|Δ| = {d1:.2e} → {'EQUAL ✓' if d1<1e-4 else '✗'} (chunk-1 prior identical)")
    print(f"      chunk-2 max|Δ| = {d2:.2e} → {'DIVERGES ✓' if d2>1e-4 else 'same?!'} (modes differ from chunk-2)")
    print(f"  [2] invariant payload_intr=payload0+bp_feedback+src_feedback: by construction")
    print(f"  [3] source_in+src_feedback=P assert ran inline during B decode (no raise → holds)\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebno", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--out-prefix", default="results/source_purity")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    crc_enc, crc_dec, ldpc, mp, dm, awgn = make_common()
    bk = bitbank()
    verify(ldpc, crc_enc, mp, dm, awgn, bk)

    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    bp30 = LDPC5GDecoder(ldpc, num_iter=30, **common)
    bp100 = LDPC5GDecoder(ldpc, num_iter=100, **common)
    bp = build_legacy(ldpc, "bp_post")
    mn = build_legacy(ldpc, "minus_source_feedback")

    # (label, decoder, mode-note, alpha, beta)
    CFG = [
        ("BP-30",  bp30,  "-", None, None),
        ("BP-100", bp100, "-", None, None),
        ("A_bp_post_ab.1", bp, "bp_post",  0.1, 0.1),
        ("B_minus_ab.1",   mn, "minus",    0.1, 0.1),
        ("C_bp_post_b0",   bp, "bp_post",  0.1, 0.0),
        ("D_minus_b0",     mn, "minus",    0.1, 0.0),
        ("B_minus_a.05",   mn, "minus",    0.05, 0.1),
        ("B_minus_a.2",    mn, "minus",    0.2, 0.1),
    ]
    E = args.ebno
    print(f"=== experiment @ {E} dB, {args.batch}×{args.rounds}={args.batch*args.rounds} cw ===", flush=True)
    results = {}
    csv_f = open(args.out_prefix + ".csv", "w", newline="")
    cw = csv.writer(csv_f); cw.writerow(["label","mode","alpha","beta","bler","ci_lo","ci_hi","ber","nack","total"]); csv_f.flush()
    for label, dec, mode, a, bta in CFG:
        track = a is not None
        if a is not None:
            dec.alpha = a; dec.beta = bta
        r = run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, bk, E, args.batch, args.rounds, track=track)
        results[label] = {"mode": mode, "alpha": a, "beta": bta, **r}
        print(f"  {label:18} BLER={r['bler']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] BER={r['ber']:.4f} ({r['nack']}/{r['total']})", flush=True)
        if "per_round_ber" in r:
            pr = r["per_round_ber"]; print(f"                     per-round BER: c0={pr[0]:.4f} c1={pr[1]:.4f} clast={pr[-1]:.4f}", flush=True)
        cw.writerow([label, mode, a, bta, f"{r['bler']:.6g}", f"{r['ci_lo']:.6g}", f"{r['ci_hi']:.6g}", f"{r['ber']:.6g}", r["nack"], r["total"]]); csv_f.flush()
    csv_f.close()
    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"ebno": E, "sigma": FIXED_SIGMA, "results": results}, f, indent=2)

    A = results["A_bp_post_ab.1"]["bler"]; B = results["B_minus_ab.1"]["bler"]
    C = results["C_bp_post_b0"]["bler"]; D = results["D_minus_b0"]["bler"]
    print(f"\n--- verdict ---", flush=True)
    def cmp(x, y, xl, yl):
        ov = not (results[xl]["ci_hi"] < results[yl]["ci_lo"] or results[yl]["ci_hi"] < results[xl]["ci_lo"])
        return f"{'≈ (CI overlap)' if ov else ('>' if x>y else '<')}"
    print(f"  B vs A (α=β=0.1): B={B:.4f} {cmp(B,A,'B_minus_ab.1','A_bp_post_ab.1')} A={A:.4f}")
    print(f"  D vs C (β=0):     D={D:.4f} {cmp(D,C,'D_minus_b0','C_bp_post_b0')} C={C:.4f}")
    print(f"  BP-30={results['BP-30']['bler']:.4f} BP-100={results['BP-100']['bler']:.4f}")
    print(f"saved: {args.out_prefix}.csv/.json")


if __name__ == "__main__":
    main()
