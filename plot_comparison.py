"""
Eb/N0 sweep: baseline BP vs onlyextrinsic BP+Denoiser (alpha/beta).

[onlyextrinsic variant — independent alpha/beta]
  new_input = channel + beta * bp_ext + alpha * src_ext

Comparison modes (preserved for backward compatibility):

  * default                : baseline + fixed σ
  * --adaptive-sigma       : baseline + adaptive σ only
  * --compare-adaptive     : baseline + fixed σ + hand-crafted adaptive σ
  * --compare-four-modes   : baseline + fixed σ + hand-crafted adaptive σ
                             + calibrated adaptive σ (requires --sigma-lookup-json)
  * --monotonic-sigma      : repair active lookup so larger ratio bins ⇒
                             non-smaller σ (applies to hand-crafted and
                             calibrated branches)

Saves plot to ``--output`` (default ``results/comparison.png``) and a CSV
row per Eb/N0 to ``--append-csv`` if requested.

Usage examples:
  CUDA_VISIBLE_DEVICES=0 python plot_comparison.py
  CUDA_VISIBLE_DEVICES=0 python plot_comparison.py --alpha 0.1 --beta 0.1
  CUDA_VISIBLE_DEVICES=0 python plot_comparison.py --compare-four-modes \\
      --sigma-lookup-json results/sigma_lookup_tail_nack.json --monotonic-sigma
"""
import argparse
import csv
import datetime
import gc
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from cli_common import (
    add_adaptive_sigma_args,
    add_runtime_args,
    decoder_sigma_kwargs,
    early_gpu_mb_argv,
)

_mb = early_gpu_mb_argv(sys.argv)
if _mb is not None and _mb > 0:
    os.environ["FMNIST_GPU_MEM_MB"] = str(_mb)

import numpy as np
import tensorflow as tf
from gpu_limits import setup_tensorflow_gpu

setup_tensorflow_gpu()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no
from decoder import LDPC5GDecoder_soft

import torchvision

IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP  # 6272

CRC_DEGREE = "CRC24A"
N_CODEWORD = 12600
NUM_BPS = 1

BEST_ALPHA = 0.1
BEST_BETA = 0.0
BEST_SIGMA = 0.3
BP_SCHEDULE = [10, 10, 10]
BATCH = 200
ROUNDS = 5
SEED = 42
CKPT = "checkpoints/denoiser.pt"

EBNO_LIST = [0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]


def build_test_bitbank():
    """Fashion-MNIST test set (10,000 images), grayscale 28x28 → 6272 bits each."""
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=True)
    images = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = images.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    bits = np.unpackbits(flat, axis=1)
    return tf.constant(bits, dtype=tf.int32)


def _build_decoder(ldpc_enc, alpha, beta, *, scheduler_kwargs=None,
                   fixed_sigma=None):
    """Construct an LDPC5GDecoder_soft using a sigma_scheduler kwargs bundle."""
    extras = scheduler_kwargs or {}
    dec = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=BP_SCHEDULE,
        alpha=alpha, beta=beta, k_payload=K_PAYLOAD,
        cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(BP_SCHEDULE), llr_max=30.0,
        **extras,
    )
    if fixed_sigma is not None:
        dec.denoiser.sigma = float(fixed_sigma)
    return dec


def run_sweep(ebno_list, dec_base, decoders_named, ldpc_enc,
              crc_enc, crc_dec, mapper, demapper, awgn, bit_bank,
              batch, rounds):
    """Run an Eb/N0 sweep across a baseline and a dict of named decoders."""
    keys = list(decoders_named.keys())
    results = {"ebno": [], "nack_base": [], "err_base": []}
    for k in keys:
        results[f"nack_{k}"] = []
        results[f"err_{k}"] = []

    for ebno in ebno_list:
        no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
        ack_b = err_b = 0
        ack = {k: 0 for k in keys}
        err = {k: 0 for k in keys}

        for r in range(rounds):
            tf.random.set_seed(SEED + r + int(ebno * 1000))
            n_imgs = tf.shape(bit_bank)[0]
            idx = tf.random.uniform([batch], 0, n_imgs, dtype=tf.int32)
            u = tf.gather(bit_bank, idx)
            u_crc = crc_enc(tf.cast(u, ldpc_enc.rdtype))
            c = ldpc_enc(u_crc)
            x = mapper(c)
            y = awgn(x, no)
            llr_ch = demapper(y, no)

            hat_b = dec_base(llr_ch)
            _, cv_b = crc_dec(hat_b)
            ack_b += int(tf.reduce_sum(tf.cast(cv_b, tf.int32)).numpy())
            err_b += int(tf.reduce_sum(tf.cast(
                tf.not_equal(u, tf.cast(hat_b[:, :K_PAYLOAD] > 0, tf.int32)),
                tf.int32)).numpy())

            for k, dec in decoders_named.items():
                hat = dec(llr_ch)
                _, cv = crc_dec(hat)
                ack[k] += int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
                err[k] += int(tf.reduce_sum(tf.cast(
                    tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
                    tf.int32)).numpy())

        total = batch * rounds
        results["ebno"].append(ebno)
        results["nack_base"].append((total - ack_b) / total)
        results["err_base"].append(err_b / (total * K_PAYLOAD))
        for k in keys:
            results[f"nack_{k}"].append((total - ack[k]) / total)
            results[f"err_{k}"].append(err[k] / (total * K_PAYLOAD))

        msg = (f"Eb/N0={ebno:+.1f} dB  |  "
               f"baseline NACK={total-ack_b:>4}/{total} "
               f"BER={err_b/(total*K_PAYLOAD):.2e}")
        for k in keys:
            msg += (f"  |  {k:>10s} NACK={total-ack[k]:>4}/{total} "
                    f"BER={err[k]/(total*K_PAYLOAD):.2e}")
        print(msg)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=BEST_ALPHA)
    p.add_argument("--beta", type=float, default=BEST_BETA)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--ebno", type=float, nargs="+", default=EBNO_LIST)
    add_adaptive_sigma_args(p, fixed_sigma_default=BEST_SIGMA)
    add_runtime_args(p, batch_default=BATCH, rounds_default=ROUNDS)
    p.add_argument("--compare-adaptive", action="store_true",
                   help="Plot baseline + fixed σ + hand-crafted adaptive σ.")
    p.add_argument("--compare-four-modes", action="store_true",
                   help="Plot baseline + fixed + hand-crafted + calibrated.")
    p.add_argument("--output", default="results/comparison.png",
                   help="Output PNG path.")
    p.add_argument("--no-save-plot", action="store_true",
                   help="Skip matplotlib figure (memory/stability).")
    p.add_argument("--append-csv", default=None,
                   help="Append scalar results to this CSV (one row per Eb/N0).")
    p.add_argument("--sweep-tag", default="",
                   help="Tag stamped into --append-csv rows.")
    args = p.parse_args()

    # Build baseline (pure BP) and the fixed/adaptive/calibrated branches.
    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    k_ldpc = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)

    dec_base = LDPC5GDecoder(
        ldpc_enc, cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(BP_SCHEDULE), llr_max=30.0)

    # Active scheduler from CLI (for "adaptive only" or "compare adaptive").
    active_kwargs, active_info = decoder_sigma_kwargs(
        args, force_adaptive=args.adaptive_sigma)

    # Always available: a fixed-σ branch and a hand-crafted adaptive branch.
    fixed_kwargs, _ = decoder_sigma_kwargs(args, force_adaptive=False)
    handcrafted_args = argparse.Namespace(**vars(args))
    handcrafted_args.sigma_lookup_json = None  # force handcrafted lookup
    hand_kwargs, hand_info = decoder_sigma_kwargs(
        handcrafted_args, force_adaptive=True)

    cal_kwargs = None
    cal_info = None
    if args.sigma_lookup_json:
        # Calibrated branch (only used when --compare-four-modes or
        # --adaptive-sigma --sigma-lookup-json).
        cal_kwargs, cal_info = decoder_sigma_kwargs(args, force_adaptive=True)

    dec_fixed = _build_decoder(ldpc_enc, args.alpha, args.beta,
                               scheduler_kwargs=fixed_kwargs,
                               fixed_sigma=args.sigma)
    dec_adaptive = _build_decoder(ldpc_enc, args.alpha, args.beta,
                                  scheduler_kwargs=hand_kwargs)
    dec_calibrated = (
        _build_decoder(ldpc_enc, args.alpha, args.beta,
                       scheduler_kwargs=cal_kwargs)
        if cal_kwargs is not None else None
    )

    if os.path.isfile(args.ckpt):
        dec_fixed.denoiser.load_weights_pt(args.ckpt)
        dec_adaptive.denoiser.load_weights_pt(args.ckpt)
        if dec_calibrated is not None:
            dec_calibrated.denoiser.load_weights_pt(args.ckpt)
        print(f"[INFO] Loaded {args.ckpt}")
    else:
        print(f"[WARN] {args.ckpt} not found")

    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam",
                        num_bits_per_symbol=NUM_BPS)
    awgn = AWGN()

    bit_bank = build_test_bitbank()
    print(f"Test bitbank: {bit_bank.shape} (Fashion-MNIST test, "
          f"grayscale {IMG_H}x{IMG_W}, {BPP}-bit → {K_PAYLOAD} bits/img)")

    use_fixed = (not args.adaptive_sigma) or args.compare_adaptive or args.compare_four_modes
    use_adaptive = args.adaptive_sigma or args.compare_adaptive or args.compare_four_modes
    use_calibrated = (args.compare_four_modes
                      and dec_calibrated is not None)
    if args.adaptive_sigma and args.sigma_lookup_json and dec_calibrated is not None and not args.compare_four_modes:
        # If only adaptive+calibrated requested without compare flags, use the
        # calibrated branch as "adaptive".
        dec_adaptive = dec_calibrated
        use_adaptive = True
        use_calibrated = False

    decoders_named = {}
    if use_fixed:
        decoders_named["fixed"] = dec_fixed
    if use_adaptive:
        decoders_named["adaptive"] = dec_adaptive
    if use_calibrated:
        decoders_named["calibrated"] = dec_calibrated

    sig_note_parts = []
    if use_fixed:
        sig_note_parts.append(f"fixed σ={args.sigma}")
    if use_adaptive:
        if args.sigma_lookup_json and not use_calibrated:
            sig_note_parts.append(
                f"adaptive (calibrated:{os.path.basename(args.sigma_lookup_json)})")
        else:
            sig_note_parts.append("adaptive (hand-crafted)")
    if use_calibrated:
        sig_note_parts.append(
            f"calibrated ({os.path.basename(args.sigma_lookup_json)})")
    sig_note = " + ".join(sig_note_parts) or "baseline only"

    print(f"α={args.alpha}  β={args.beta}  {sig_note}  "
          f"schedule={BP_SCHEDULE}  batch={args.batch}  rounds={args.rounds}  "
          f"monotonic_sigma={args.monotonic_sigma}")
    print(f"  syndrome_hard_decision={args.syndrome_hard_decision}  "
          f"sign_debug={args.syndrome_sign_debug}")

    res = run_sweep(args.ebno, dec_base, decoders_named, ldpc_enc,
                    crc_enc, crc_dec, mapper, demapper, awgn, bit_bank,
                    args.batch, args.rounds)

    # ── CSV append (scalar only) ─────────────────────────────────────
    if args.append_csv:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if args.compare_four_modes:
            mode_tag = "four_modes"
        elif args.compare_adaptive:
            mode_tag = "compare_adaptive"
        elif args.adaptive_sigma:
            mode_tag = "adaptive_only"
        else:
            mode_tag = "baseline_fixed"
        path = args.append_csv
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = [
            "ts_utc", "sweep_tag", "mode", "alpha", "beta", "sigma_fixed",
            "monotonic_sigma", "syndrome_hard_decision",
            "batch", "rounds", "ebno_db",
            "bler_baseline", "ber_baseline",
            "bler_fixed", "ber_fixed",
            "bler_adaptive", "ber_adaptive",
            "bler_calibrated", "ber_calibrated",
        ]
        write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for i, eb in enumerate(res["ebno"]):
                row = {
                    "ts_utc": ts,
                    "sweep_tag": args.sweep_tag,
                    "mode": mode_tag,
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "sigma_fixed": args.sigma,
                    "monotonic_sigma": int(args.monotonic_sigma),
                    "syndrome_hard_decision": args.syndrome_hard_decision,
                    "batch": args.batch,
                    "rounds": args.rounds,
                    "ebno_db": eb,
                    "bler_baseline": res["nack_base"][i],
                    "ber_baseline": res["err_base"][i],
                    "bler_fixed": res.get("nack_fixed", [""] * len(res["ebno"]))[i],
                    "ber_fixed": res.get("err_fixed", [""] * len(res["ebno"]))[i],
                    "bler_adaptive": res.get("nack_adaptive",
                                             [""] * len(res["ebno"]))[i],
                    "ber_adaptive": res.get("err_adaptive",
                                            [""] * len(res["ebno"]))[i],
                    "bler_calibrated": res.get("nack_calibrated",
                                               [""] * len(res["ebno"]))[i],
                    "ber_calibrated": res.get("err_calibrated",
                                              [""] * len(res["ebno"]))[i],
                }
                w.writerow(row)
        print(f"[scalar] Appended rows to {path}")

    # ── Plot ─────────────────────────────────────────────────────────
    if not args.no_save_plot:
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        labels = {
            "fixed": f"BP + Denoiser Fixed (σ={args.sigma})",
            "adaptive": "BP + Denoiser Adaptive (hand-crafted)",
            "calibrated": "BP + Denoiser Calibrated Adaptive",
        }
        markers = {"fixed": "s", "adaptive": "^", "calibrated": "d"}
        colors = {"fixed": "#1f77b4", "adaptive": "#2ca02c",
                  "calibrated": "#9467bd"}

        ax1.semilogy(res["ebno"], res["nack_base"], "o-", color="#d62728",
                     linewidth=2, markersize=7,
                     label=f"Baseline BP ({sum(BP_SCHEDULE)} iter)")
        for k in ("fixed", "adaptive", "calibrated"):
            key = f"nack_{k}"
            if key in res:
                ax1.semilogy(res["ebno"], res[key],
                             marker=markers[k], linestyle="-", color=colors[k],
                             linewidth=2, markersize=7, label=labels[k])

        ax1.set_xlabel("Eb/N0 (dB)", fontsize=12)
        ax1.set_ylabel("NACK Rate (BLER)", fontsize=12)
        ax1.set_title("Block Error Rate", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, which="both", alpha=0.3)
        ax1.set_ylim(bottom=5e-4)

        # BER (mask out zeros for log scale)
        def _mask_pos(xs, ys):
            pairs = [(e, v) for e, v in zip(xs, ys) if v > 0]
            return zip(*pairs) if pairs else ([], [])

        eb_b, er_b = _mask_pos(res["ebno"], res["err_base"])
        if eb_b:
            ax2.semilogy(eb_b, er_b, "o-", color="#d62728",
                         linewidth=2, markersize=7, label="Baseline BP")
        for k in ("fixed", "adaptive", "calibrated"):
            key = f"err_{k}"
            if key in res:
                ebk, erk = _mask_pos(res["ebno"], res[key])
                if ebk:
                    ax2.semilogy(ebk, erk, marker=markers[k], linestyle="-",
                                 color=colors[k], linewidth=2, markersize=7,
                                 label=labels[k])
        ax2.set_xlabel("Eb/N0 (dB)", fontsize=12)
        ax2.set_ylabel("Bit Error Rate", fontsize=12)
        ax2.set_title("Bit Error Rate", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, which="both", alpha=0.3)

        fig.suptitle(
            "Fashion-MNIST over AWGN — Baseline BP vs onlyextrinsic BP+Denoiser\n"
            f"new_input = ch + β·bp_ext + α·src_ext  |  "
            f"K={K_PAYLOAD}, N={N_CODEWORD}, BPSK, schedule={BP_SCHEDULE}",
            fontsize=11, y=1.02)
        fig.tight_layout()
        out = args.output
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nPlot saved → {out}")

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    main()
