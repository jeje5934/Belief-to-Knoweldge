"""
Prompt B — full-EP diagnostic: does the per-round explosion vanish with a
single-category (unimodal) denoiser?

Hypothesis: multi-category full EP breaks (BLER 1.0, BER 0.45) because the
denoiser produces a CORRELATED hallucination, rooted in the cross-category
multimodal posterior.  Trouser (one category) has no cross-category
multimodality, so full EP should work / the explosion should vanish.

Design — TRANSMIT TROUSER IMAGES; isolate the denoiser as the only variable:
  * bp         : pure BP reference, source OFF (ep_mode=False, α=β=0). Same
                 [2]×15 schedule, warm-started → per-round BER should DECREASE.
  * ep_multi   : full EP (ep_mode=True, ep_update=full_ep, α_ep=1) with the
                 MULTI-category denoiser.pt — replicate the multi-cat failure
                 on Trouser images.
  * ep_trouser : full EP with the single-category denoiser_trouser.pt — the test.

Key metric — per-round (per-chunk) BER trajectory (ep_track_payload_hist):
  chunk 0 = pure BP (no source yet), chunk 1 = after first source injection.
  Multi-cat reference: 0.13 → 0.46 explosion, then stays ~0.45.

One SNR (default 0.6 dB), fast.  CPU-only by default (bridge-bound; and stable —
see docs/COMPUTE_LESSONS.md re: GPU faults on long runs; this is short so GPU ok).

Usage:
  CUDA_VISIBLE_DEVICES=0 python practical_promptB_fullep_diag.py \
      --ebno 0.6 --batch 64 --rounds 8
"""
import os
import argparse
import json
import csv

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
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no

from decoder import LDPC5GDecoder_soft

IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP
N_CODEWORD = 12600
NUM_BPS = 1
TROUSER_LABEL = 1
EP_SCHEDULE = [2] * 15
CKPT_MULTI = "checkpoints/denoiser.pt"
CKPT_TROUSER = "checkpoints/denoiser_trouser.pt"
SEED = 42

# Multi-category full-EP reference (documented earlier), on multi-cat images:
MULTI_REF = {"bler": 1.0, "ber": 0.45,
             "per_round_ber_note": "0.13 (chunk0) -> 0.46 (chunk1), stays ~0.45"}


def build_trouser_bitbank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=False,
                                           download=True)
    imgs = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    labels = np.array(ds.targets)
    imgs = imgs[labels == TROUSER_LABEL]           # Trouser test images only
    flat = imgs.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    return tf.constant(np.unpackbits(flat, axis=1), dtype=tf.int32)


def make_common():
    crc_enc = CRCEncoder("CRC24A")
    crc_dec = CRCDecoder(crc_enc)
    ldpc_enc = LDPC5GEncoder(K_PAYLOAD + crc_enc.crc_length, N_CODEWORD,
                             num_bits_per_symbol=NUM_BPS)
    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam",
                        num_bits_per_symbol=NUM_BPS)
    return crc_enc, crc_dec, ldpc_enc, mapper, demapper, AWGN()


def build_decoders(ldpc_enc):
    common = dict(cn_update="boxplus-phi", vn_update="sum",
                  cn_schedule="flooding", hard_out=False,
                  return_infobits=True, llr_max=30.0)
    # pure BP: source OFF (legacy path, α=β=0), same schedule, track per-round.
    bp = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=EP_SCHEDULE, k_payload=K_PAYLOAD,
        ep_mode=False, alpha=0.0, beta=0.0,
        adaptive_sigma=True, ep_track_payload_hist=True,
        num_iter=sum(EP_SCHEDULE), **common)
    # full EP with the multi-category denoiser (replicate failure).
    ep_multi = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=EP_SCHEDULE, k_payload=K_PAYLOAD,
        ep_mode=True, ep_update="full_ep", adaptive_sigma=True,
        ep_track_payload_hist=True, num_iter=sum(EP_SCHEDULE), **common)
    # full EP with the single-category (Trouser) denoiser (the test).
    ep_trouser = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=EP_SCHEDULE, k_payload=K_PAYLOAD,
        ep_mode=True, ep_update="full_ep", adaptive_sigma=True,
        ep_track_payload_hist=True, num_iter=sum(EP_SCHEDULE), **common)
    for name, dec, ck in [("bp", bp, CKPT_MULTI),
                          ("ep_multi", ep_multi, CKPT_MULTI),
                          ("ep_trouser", ep_trouser, CKPT_TROUSER)]:
        if os.path.isfile(ck):
            dec.denoiser.load_weights_pt(ck)
        else:
            print(f"[WARN] {name}: checkpoint missing {ck}")
    return {"bp": bp, "ep_multi": ep_multi, "ep_trouser": ep_trouser}


def per_round_ber(dec, u_true):
    """From the decoder's payload history, BER of each chunk's BP posterior."""
    hist = dec.last_payload_hist          # list of [B, k_payload] LLR tensors
    out = []
    for logit in hist:
        hard = tf.cast(logit[:, :K_PAYLOAD] > 0.0, tf.int32)
        be = tf.reduce_mean(tf.cast(tf.not_equal(u_true, hard), tf.float64))
        out.append(float(be.numpy()))
    return out


def run(dec, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
        bit_bank, ebno, batch, rounds):
    no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
    n_imgs = tf.shape(bit_bank)[0]
    tot_nack = tot_biterr = 0
    total = batch * rounds
    pr_accum = None
    for r in range(rounds):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        idx = tf.random.uniform([batch], 0, n_imgs, dtype=tf.int32)
        u = tf.gather(bit_bank, idx)
        u_crc = crc_enc(tf.cast(u, ldpc_enc.rdtype))
        y = awgn(mapper(ldpc_enc(u_crc)), no)
        llr = demapper(y, no)
        hat = dec(llr)
        _, cv = hard_crc_decode(crc_dec, hat)
        ack = int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_nack += (batch - ack)
        tot_biterr += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())
        pr = per_round_ber(dec, u)
        pr_accum = pr if pr_accum is None else [a + b for a, b in zip(pr_accum, pr)]
    pr_mean = [v / rounds for v in pr_accum] if pr_accum else []
    return {"bler": tot_nack / total, "ber": tot_biterr / (total * K_PAYLOAD),
            "nack": tot_nack, "total": total, "per_round_ber": pr_mean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebno", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out-prefix", default="results/promptB_fullep_diag")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    crc_enc, crc_dec, ldpc_enc, mapper, demapper, awgn = make_common()
    bit_bank = build_trouser_bitbank()
    print(f"Trouser test bit-bank: {int(tf.shape(bit_bank)[0])} images, "
          f"Eb/N0={args.ebno} dB, {args.batch}×{args.rounds} codewords/decoder\n")
    decs = build_decoders(ldpc_enc)

    results = {}
    for name in ("bp", "ep_multi", "ep_trouser"):
        r = run(decs[name], ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
                bit_bank, args.ebno, args.batch, args.rounds)
        results[name] = r
        pr = r["per_round_ber"]
        print(f"[{name}] BLER={r['bler']:.4f}  BER={r['ber']:.4f}  "
              f"({r['nack']}/{r['total']} NACK)")
        print(f"    per-round BER: chunk0={pr[0]:.4f}  chunk1={pr[1]:.4f}  "
              f"... chunk{len(pr)-1}={pr[-1]:.4f}")
        print(f"    full trajectory: {[round(v,4) for v in pr]}\n")

    # explosion metric: chunk1/chunk0 ratio (first source injection)
    for name in ("ep_multi", "ep_trouser"):
        pr = results[name]["per_round_ber"]
        results[name]["explosion_ratio_c1_over_c0"] = pr[1] / max(pr[0], 1e-9)

    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"ebno": args.ebno, "schedule": EP_SCHEDULE,
                   "multi_cat_reference": MULTI_REF, "results": results},
                  f, indent=2)
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decoder", "chunk", "per_round_ber"])
        for name in results:
            for c, v in enumerate(results[name]["per_round_ber"]):
                w.writerow([name, c, f"{v:.6g}"])

    _plot(results, args.ebno, args.out_prefix + ".png")
    print(f"saved: {args.out_prefix}.png/.csv/.json")
    print(f"\nMulti-cat full-EP reference (multi-cat images): BLER "
          f"{MULTI_REF['bler']}, BER {MULTI_REF['ber']}, "
          f"per-round {MULTI_REF['per_round_ber_note']}")


def _plot(results, ebno, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"bp": "pure BP (source OFF)",
              "ep_multi": "full EP + multi-cat denoiser",
              "ep_trouser": "full EP + Trouser denoiser"}
    styles = {"bp": "s--", "ep_multi": "o-", "ep_trouser": "^-"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("bp", "ep_multi", "ep_trouser"):
        pr = results[name]["per_round_ber"]
        ax.plot(range(len(pr)), pr, styles[name], label=labels[name],
                linewidth=2, markersize=6)
    ax.set_xlabel("chunk (round); source injected from chunk 1 on")
    ax.set_ylabel("per-round payload BER")
    ax.set_title(f"Per-round BER — full-EP correlated-hallucination probe "
                 f"(Trouser images, {ebno} dB)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
