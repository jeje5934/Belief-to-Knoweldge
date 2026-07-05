"""
Prompt 0 — Low-SNR baseline for pure-EP_practical.

Goal: at the SNR region where BP struggles (0.4-0.7 dB), establish the three
reference curves under a COMMON BP budget so the ONLY difference is "source
knowledge on/off":

  * bp{K}   — Baseline BP, K iterations, NO source prior.  K = total BP budget
              of the EP schedule ([2]*15 => 30 BP iters => bp30).  [same budget]
  * ep      — Working EP decoder: bp_schedule=[2]*15, ep_mode=True,
              ep_update="damped_ep", ep_source_power=0.02 (α_ep), adaptive
              syndrome-ratio σ.  (NOT full_ep — that decodes at BLER 1.0.)
  * bp100   — Baseline BP, 100 iterations.  BP-limit ceiling reference line.

For every (SNR, decoder) we report:
  * pooled BLER + Wilson score 95% CI
  * pooled BER
  * per-round BLER (list) — mean / min / max, to gauge stability

CPU-only by default (the denoiser bridge is CPU-bound anyway).

Usage:
  CUDA_VISIBLE_DEVICES="" python practical_lowsnr_baseline.py \
      --ebno-list 0.4 0.5 0.6 0.7 --batch 64 --rounds 50
"""
import os
import sys
import csv
import json
import math
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf

# GPU safety: when a GPU is visible (CUDA_VISIBLE_DEVICES != ""), enable memory
# growth so TF does not grab all VRAM up-front (torch/denoiser shares the device).
# Prevents the host GPU lockups noted in README/HANDOFF.  CPU-only run is a no-op.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception as _e:  # already initialized, etc.
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
CKPT = "checkpoints/denoiser.pt"

# Working EP config (ada-sigma best): damped EP, α_ep=0.02, [2]*15.
EP_SCHEDULE = [2] * 15
EP_SOURCE_POWER = 0.02
BP_BUDGET = sum(EP_SCHEDULE)     # 30 => baseline "bp30" matches EP BP budget
SEED = 42


# ────────────────────────── statistics ──────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (p, max(0.0, center - half), min(1.0, center + half))


# ────────────────────────── harness ──────────────────────────

def build_test_bitbank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=False,
                                           download=True)
    images = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = images.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
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
    bpK = LDPC5GDecoder(ldpc_enc, num_iter=BP_BUDGET, **common)
    bp100 = LDPC5GDecoder(ldpc_enc, num_iter=100, **common)
    ep = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=EP_SCHEDULE, k_payload=K_PAYLOAD,
        ep_mode=True, ep_update="damped_ep", ep_source_power=EP_SOURCE_POWER,
        ep_code_power=1.0, adaptive_sigma=True,
        num_iter=sum(EP_SCHEDULE), **common)
    if os.path.isfile(CKPT):
        ep.denoiser.load_weights_pt(CKPT)
    else:
        print(f"[WARN] checkpoint missing ({CKPT}); denoiser weights random.")
    return {f"bp{BP_BUDGET}": bpK, "ep": ep, "bp100": bp100}


def run_point(dec, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
              bit_bank, ebno, batch, rounds):
    """Return dict with pooled + per-round stats for one (dec, ebno)."""
    no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
    n_imgs = tf.shape(bit_bank)[0]
    tot_nack = tot_biterr = 0
    total = batch * rounds
    per_round_bler = []
    for r in range(rounds):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        idx = tf.random.uniform([batch], 0, n_imgs, dtype=tf.int32)
        u = tf.gather(bit_bank, idx)
        u_crc = crc_enc(tf.cast(u, ldpc_enc.rdtype))
        y = awgn(mapper(ldpc_enc(u_crc)), no)
        llr = demapper(y, no)
        hat = dec(llr)
        _, cv = crc_dec(hat)
        ack = int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())   # cv=1 => ACK
        nack = batch - ack
        tot_nack += nack
        per_round_bler.append(nack / batch)
        tot_biterr += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())
    p, lo, hi = wilson_ci(tot_nack, total)
    return {
        "bler": p, "bler_ci_lo": lo, "bler_ci_hi": hi,
        "ber": tot_biterr / (total * K_PAYLOAD),
        "nack": tot_nack, "total": total, "biterr": tot_biterr,
        "per_round_bler_mean": float(np.mean(per_round_bler)),
        "per_round_bler_min": float(np.min(per_round_bler)),
        "per_round_bler_max": float(np.max(per_round_bler)),
        "per_round_bler": per_round_bler,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ebno-list", type=float, nargs="+",
                   default=[0.4, 0.5, 0.6, 0.7])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--out-prefix", default="results/practical_lowsnr_baseline")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    crc_enc, crc_dec, ldpc_enc, mapper, demapper, awgn = make_common()
    bit_bank = build_test_bitbank()
    decs = build_decoders(ldpc_enc)
    names = list(decs.keys())            # [bp30, ep, bp100]

    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["ebno_db", "decoder", "bler", "bler_ci_lo", "bler_ci_hi",
             "ber", "nack", "total", "biterr",
             "pr_bler_mean", "pr_bler_min", "pr_bler_max"])

    total_cw = args.batch * args.rounds
    results = {name: {} for name in names}
    print(f"Low-SNR baseline: {total_cw} codewords/point "
          f"(batch {args.batch} × rounds {args.rounds}), "
          f"schedule={EP_SCHEDULE}, α_ep={EP_SOURCE_POWER}, "
          f"Eb/N0={args.ebno_list}\n")
    print(f"{'Eb/N0':>6} {'decoder':>7} {'BLER':>8}  {'95% CI':>17}  "
          f"{'BER':>9}  {'NACK/tot':>10}")
    for ebno in args.ebno_list:
        for name in names:
            r = run_point(decs[name], ldpc_enc, crc_enc, crc_dec, mapper,
                          demapper, awgn, bit_bank, ebno, args.batch, args.rounds)
            results[name][ebno] = r
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [ebno, name, f"{r['bler']:.6g}", f"{r['bler_ci_lo']:.6g}",
                     f"{r['bler_ci_hi']:.6g}", f"{r['ber']:.6g}",
                     r["nack"], r["total"], r["biterr"],
                     f"{r['per_round_bler_mean']:.6g}",
                     f"{r['per_round_bler_min']:.6g}",
                     f"{r['per_round_bler_max']:.6g}"])
            print(f"{ebno:>6} {name:>7} {r['bler']:>8.4f}  "
                  f"[{r['bler_ci_lo']:.4f},{r['bler_ci_hi']:.4f}]  "
                  f"{r['ber']:>9.2e}  {r['nack']:>4}/{r['total']:<5}")
        # per-SNR source-gain readout
        bpK = results[f"bp{BP_BUDGET}"][ebno]["bler"]
        epb = results["ep"][ebno]["bler"]
        bp100b = results["bp100"][ebno]["bler"]
        print(f"       └ source gain vs bp{BP_BUDGET}: "
              f"ΔBLER={bpK - epb:+.4f}  |  EP below BP-100 ceiling: "
              f"{'YES' if epb < bp100b else 'no'} "
              f"(EP {epb:.4f} vs ceil {bp100b:.4f})\n")

    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"schedule": EP_SCHEDULE, "ep_source_power": EP_SOURCE_POWER,
                   "bp_budget": BP_BUDGET, "codewords_per_point": total_cw,
                   "ebno_list": args.ebno_list, "results": results}, f, indent=2)

    _plot(results, names, args.ebno_list, args.out_prefix + ".png")
    print(f"saved: {csv_path}, {args.out_prefix}.png, {args.out_prefix}.json")


def _plot(results, names, ebnos, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {f"bp{BP_BUDGET}": f"Baseline BP ({BP_BUDGET} it, no prior)",
              "ep": f"EP damped ([2]×15, α_ep={EP_SOURCE_POWER}, adapt σ)",
              "bp100": "Baseline BP (100 it, ceiling)"}
    styles = {f"bp{BP_BUDGET}": "s--", "ep": "o-", "bp100": "^:"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for name in names:
        ber = [results[name][e]["ber"] for e in ebnos]
        bler = [results[name][e]["bler"] for e in ebnos]
        lo = [results[name][e]["bler_ci_lo"] for e in ebnos]
        hi = [results[name][e]["bler_ci_hi"] for e in ebnos]
        ax1.semilogy(ebnos, [max(b, 1e-6) for b in ber],
                     styles[name], label=labels[name], linewidth=2, markersize=6)
        yerr = [[max(results[name][e]["bler"] - l, 0) for e, l in zip(ebnos, lo)],
                [max(h - results[name][e]["bler"], 0) for e, h in zip(ebnos, hi)]]
        ax2.errorbar(ebnos, [max(b, 1e-4) for b in bler], yerr=yerr,
                     fmt=styles[name], label=labels[name], linewidth=2,
                     markersize=6, capsize=3)
    ax2.set_yscale("log")
    ax1.set_ylabel("BER"); ax1.grid(True, which="both", alpha=0.3); ax1.legend()
    ax1.set_title("Low-SNR baseline — EP (source) vs BP, common BP budget")
    ax2.set_ylabel("BLER (Wilson 95% CI)"); ax2.set_xlabel("Eb/N0 (dB)")
    ax2.grid(True, which="both", alpha=0.3); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
