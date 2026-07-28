"""
Collect syndrome statistics per BP chunk across Eb/N0 values.

Outputs:
  - results/syndrome_diagnostics.json
  - results/syndrome_diagnostics.csv

Modes always evaluated:
  baseline                    : pure BP, no denoiser
  fixed_sigma                 : BP + denoiser with fixed σ (--sigma)
  adaptive_handcrafted        : BP + denoiser with hand-crafted lookup
  adaptive_calibrated         : BP + denoiser with --sigma-lookup-json (if given)

The CLI flags come from ``cli_common`` so they match other scripts.
"""
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

import argparse
import csv
import json

import numpy as np
import tensorflow as tf
from gpu_limits import setup_tensorflow_gpu

setup_tensorflow_gpu()

import torchvision

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no

from decoder import LDPC5GDecoder_soft
from syndrome_sigma_schedule import summarize_chunk_diagnostics


IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP
CRC_DEGREE = "CRC24A"
N_CODEWORD = 12600
NUM_BPS = 1
SCHEDULE = [10, 10, 10]
SEED = 42
CKPT = "checkpoints/denoiser.pt"


def build_test_bitbank():
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=True)
    images = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = images.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    bits = np.unpackbits(flat, axis=1)
    return tf.constant(bits, dtype=tf.int32)


def make_decoder(mode, ldpc_enc, args):
    if mode == "baseline":
        return LDPC5GDecoder(
            ldpc_enc, cn_update="boxplus-phi", vn_update="sum",
            cn_schedule="flooding", hard_out=False, return_infobits=True,
            num_iter=sum(SCHEDULE), llr_max=30.0,
        )

    # Build a per-mode argparse copy so we can switch the lookup branch
    # without leaking flags between branches.
    mode_args = argparse.Namespace(**vars(args))
    if mode == "fixed":
        kwargs, _ = decoder_sigma_kwargs(mode_args, force_adaptive=False)
    elif mode == "adaptive_handcrafted":
        mode_args.sigma_lookup_json = None
        kwargs, _ = decoder_sigma_kwargs(mode_args, force_adaptive=True)
    elif mode == "adaptive_calibrated":
        kwargs, _ = decoder_sigma_kwargs(mode_args, force_adaptive=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    dec = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=SCHEDULE,
        alpha=args.alpha, beta=args.beta, k_payload=K_PAYLOAD,
        cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(SCHEDULE), llr_max=30.0,
        **kwargs,
    )
    if os.path.isfile(args.ckpt):
        dec.denoiser.load_weights_pt(args.ckpt)
    if not getattr(dec, "adaptive_sigma", False):
        dec.denoiser.sigma = args.sigma
    return dec


def run_mode(dec, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
             bit_bank, ebno, batch, rounds):
    no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
    ack = err = 0
    all_chunk_diags = []
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
        hat = dec(llr_ch)
        _, cv = hard_crc_decode(crc_dec, hat)
        ack += int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        err += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())
        if hasattr(dec, "last_chunk_diagnostics"):
            all_chunk_diags.extend(dec.last_chunk_diagnostics)
    total = batch * rounds
    return {
        "bler": float((total - ack) / total),
        "ber": float(err / (total * K_PAYLOAD)),
        "chunk_stats": summarize_chunk_diagnostics(all_chunk_diags),
        "blocks": int(total),
    }


def flatten_csv_rows(summary):
    rows = []
    for eb in summary["ebno_results"]:
        ebno = eb["ebno_db"]
        for mode, mres in eb["modes"].items():
            if "chunk_stats" not in mres:
                rows.append({
                    "ebno_db": ebno, "mode": mode, "chunk_idx": -1,
                    "bler": mres["bler"], "ber": mres["ber"],
                    "mean_syndrome_weight": "", "mean_syndrome_ratio": "",
                    "std_syndrome_ratio": "", "min_syndrome_ratio": "",
                    "max_syndrome_ratio": "", "mean_sigma": "",
                    "sigma_distribution": "",
                })
                continue
            for c in mres["chunk_stats"]:
                rows.append({
                    "ebno_db": ebno, "mode": mode,
                    "chunk_idx": c["chunk_idx"],
                    "bler": mres["bler"], "ber": mres["ber"],
                    "mean_syndrome_weight": c["mean_syndrome_weight"],
                    "mean_syndrome_ratio": c["mean_syndrome_ratio"],
                    "std_syndrome_ratio": c["std_syndrome_ratio"],
                    "min_syndrome_ratio": c["min_syndrome_ratio"],
                    "max_syndrome_ratio": c["max_syndrome_ratio"],
                    "mean_sigma": c["mean_sigma"],
                    "sigma_distribution": json.dumps(
                        c["sigma_distribution"], ensure_ascii=False),
                })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--ebno", type=float, nargs="+",
                   default=[0.6, 0.7, 0.8, 0.9, 1.0])
    add_adaptive_sigma_args(p, fixed_sigma_default=0.3,
                            include_adaptive_flag=False)
    add_runtime_args(p, batch_default=100, rounds_default=5)
    p.add_argument("--output-prefix", default="results/syndrome_diagnostics")
    args = p.parse_args()

    use_calibrated = bool(args.sigma_lookup_json)

    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    k_ldpc = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)
    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam",
                        num_bits_per_symbol=NUM_BPS)
    awgn = AWGN()
    bit_bank = build_test_bitbank()

    summary = {
        "alpha": args.alpha,
        "beta": args.beta,
        "sigma_fixed": args.sigma,
        "ebno_list": list(args.ebno),
        "batch": args.batch,
        "rounds": args.rounds,
        "syndrome_thresholds": list(args.syndrome_thresholds),
        "adaptive_sigmas": list(args.adaptive_sigmas),
        "monotonic_sigma": bool(args.monotonic_sigma),
        "sigma_lookup_json": args.sigma_lookup_json,
        "syndrome_hard_decision": args.syndrome_hard_decision,
        "syndrome_sign_debug": bool(args.syndrome_sign_debug),
        "ebno_results": [],
    }

    for eb in args.ebno:
        print(f"\n=== Eb/N0 {eb:.2f} dB ===")
        dec_base = make_decoder("baseline", ldpc_enc, args)
        dec_fixed = make_decoder("fixed", ldpc_enc, args)
        dec_hand = make_decoder("adaptive_handcrafted", ldpc_enc, args)
        dec_cal = (make_decoder("adaptive_calibrated", ldpc_enc, args)
                   if use_calibrated else None)

        modes = {}
        modes["baseline"] = run_mode(
            dec_base, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
            bit_bank, eb, args.batch, args.rounds)
        modes["fixed_sigma"] = run_mode(
            dec_fixed, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
            bit_bank, eb, args.batch, args.rounds)
        modes["adaptive_handcrafted"] = run_mode(
            dec_hand, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
            bit_bank, eb, args.batch, args.rounds)
        if dec_cal is not None:
            modes["adaptive_calibrated"] = run_mode(
                dec_cal, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
                bit_bank, eb, args.batch, args.rounds)

        for name, res in modes.items():
            print(f"{name:>22s}: BLER={res['bler']:.4f}, BER={res['ber']:.2e}")
            for c in res.get("chunk_stats", []):
                sd = c.get("sign_debug")
                if sd:
                    winner = "gt0✓" if sd["primary_better"] else "lt0✓"
                    print(" " * 24
                          + f"chunk{c['chunk_idx']}: ratio={c['mean_syndrome_ratio']:.4f}"
                          + f"±{c['std_syndrome_ratio']:.4f}, sigma_mean={c['mean_sigma']:.3f}"
                          + f"  |  sign_debug: gt0={sd['primary_mean_ratio']:.4f}"
                          + f"  lt0={sd['alt_mean_ratio']:.4f}  [{winner}]")
                else:
                    print(" " * 24
                          + f"chunk{c['chunk_idx']}: ratio={c['mean_syndrome_ratio']:.4f}"
                          + f"±{c['std_syndrome_ratio']:.4f}, sigma_mean={c['mean_sigma']:.3f}")

        summary["ebno_results"].append({
            "ebno_db": float(eb),
            "modes": modes,
        })

    os.makedirs("results", exist_ok=True)
    out_json = f"{args.output_prefix}.json"
    out_csv = f"{args.output_prefix}.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    rows = flatten_csv_rows(summary)
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved JSON -> {out_json}")
    print(f"Saved CSV  -> {out_csv}")


if __name__ == "__main__":
    main()
