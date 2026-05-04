"""
Alpha sweep with β = 0 (Fashion-MNIST test bitbank).

  new_input = channel + beta * bp_ext + alpha * src_ext

Supports fixed denoiser σ (legacy), syndrome-driven adaptive σ
(hand-crafted lookup), and calibrated adaptive σ from a JSON lookup
(``--sigma-lookup-json``).  Shared adaptive-σ flags come from
``cli_common`` for consistency with ``plot_comparison.py`` and
``syndrome_diagnostics.py``.

Comparison modes:
  * default                : fixed σ alpha sweep
  * --adaptive-sigma       : adaptive σ alpha sweep
  * --compare-sigma        : both fixed and adaptive on the same plot
  * --compare-sigma --compare-calibrated --sigma-lookup-json …
                           : adds a calibrated branch
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
import json

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
from syndrome_sigma_schedule import summarize_chunk_diagnostics

import torchvision

IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP  # 6272

CRC_DEGREE = "CRC24A"
N_CODEWORD = 12600
NUM_BPS = 1
SCHEDULE = [10, 10, 10]
CKPT = "checkpoints/denoiser.pt"
DEFAULT_BATCH = 200
DEFAULT_ROUNDS = 5
SEED = 42


def build_test_bitbank():
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=True)
    images = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = images.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    bits = np.unpackbits(flat, axis=1)
    return tf.constant(bits, dtype=tf.int32)


def run_one(dec, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
            bit_bank, ebno, batch, rounds):
    no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
    tot_ack = tot_err = 0
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
        if hasattr(dec, "last_chunk_diagnostics"):
            dlist = dec.last_chunk_diagnostics
            if dlist:
                all_chunk_diags.extend(dlist)
        _, cv = crc_dec(hat)
        tot_ack += int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_err += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())
    total = batch * rounds
    diag_summary = summarize_chunk_diagnostics(all_chunk_diags)
    return tot_ack, total - tot_ack, tot_err, diag_summary


def alpha_sweep(alphas, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
                bit_bank, ebno, ckpt, fixed_sigma, sched_kwargs,
                batch, rounds, *, mode_label):
    """Returns dict alpha -> dict metrics."""
    results = {}
    for alpha in alphas:
        dec = LDPC5GDecoder_soft(
            ldpc_enc, bp_schedule=SCHEDULE,
            alpha=alpha, beta=0.0, k_payload=K_PAYLOAD,
            cn_update="boxplus-phi", vn_update="sum",
            cn_schedule="flooding", hard_out=False, return_infobits=True,
            num_iter=sum(SCHEDULE), llr_max=30.0,
            **sched_kwargs)
        if os.path.isfile(ckpt):
            dec.denoiser.load_weights_pt(ckpt)
        # Fixed branch: configure denoiser scalar sigma.
        if not getattr(dec, "adaptive_sigma", False):
            dec.denoiser.sigma = fixed_sigma

        ack, nack, err, diag = run_one(
            dec, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
            bit_bank, ebno, batch, rounds)
        results[float(alpha)] = {
            "ack": ack,
            "nack": nack,
            "bit_errs": err,
            "chunk_diagnostics": diag,
        }
        tot = batch * rounds
        print(f"  α={alpha:<6}  ({mode_label})  "
              f"ACK={ack:>4}  NACK={nack:>4}  BLER={nack/tot:.4f}")
        for d in diag:
            print("      "
                  f"chunk{d['chunk_idx']}: w={d['mean_syndrome_weight']:.1f}, "
                  f"ratio={d['mean_syndrome_ratio']:.4f}±{d['std_syndrome_ratio']:.4f}, "
                  f"range=[{d['min_syndrome_ratio']:.4f},{d['max_syndrome_ratio']:.4f}], "
                  f"sigma_mean={d['mean_sigma']:.3f}, "
                  f"sigma_dist={d['sigma_distribution']}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ebno", type=float, default=0.8)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--compare-sigma", action="store_true",
                   help="Run both fixed-σ and adaptive-σ alpha sweeps.")
    p.add_argument("--compare-calibrated", action="store_true",
                   help="With --compare-sigma + --sigma-lookup-json: also "
                        "run a calibrated adaptive sweep.")
    add_adaptive_sigma_args(p, fixed_sigma_default=0.3)
    add_runtime_args(p, batch_default=DEFAULT_BATCH,
                     rounds_default=DEFAULT_ROUNDS)
    args = p.parse_args()

    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    k_ldpc = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)

    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam",
                        num_bits_per_symbol=NUM_BPS)
    awgn = AWGN()
    bit_bank = build_test_bitbank()

    dec_base = LDPC5GDecoder(
        ldpc_enc, cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(SCHEDULE), llr_max=30.0)

    batch, rounds = args.batch, args.rounds
    total = batch * rounds
    base_ack, base_nack, base_err, _ = run_one(
        dec_base, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
        bit_bank, args.ebno, batch, rounds)
    print(f"Eb/N0={args.ebno} dB  schedule={SCHEDULE}  β=0  "
          f"alphas={list(args.alphas)}  monotonic_sigma={args.monotonic_sigma}")
    print(f"Baseline BP: ACK={base_ack}/{total}  NACK={base_nack}  "
          f"BLER={base_nack/total:.4f}  BER={base_err/(total*K_PAYLOAD):.2e}\n")

    runs = []
    fixed_kwargs, _ = decoder_sigma_kwargs(args, force_adaptive=False)
    handcrafted_args = argparse.Namespace(**vars(args))
    handcrafted_args.sigma_lookup_json = None
    hand_kwargs, hand_info = decoder_sigma_kwargs(
        handcrafted_args, force_adaptive=True)
    cal_kwargs, cal_info = (None, None)
    if args.sigma_lookup_json:
        cal_kwargs, cal_info = decoder_sigma_kwargs(args, force_adaptive=True)

    if args.compare_sigma:
        print("--- Fixed σ sweep ---")
        fixed_res = alpha_sweep(
            args.alphas, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
            bit_bank, args.ebno, args.ckpt, args.sigma, fixed_kwargs,
            batch=batch, rounds=rounds,
            mode_label=f"fixed-σ={args.sigma}")
        runs.append(("fixed", args.sigma, fixed_res))
        print("\n--- Adaptive σ sweep (hand-crafted) ---")
        ad_res = alpha_sweep(
            args.alphas, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
            bit_bank, args.ebno, args.ckpt, args.sigma, hand_kwargs,
            batch=batch, rounds=rounds,
            mode_label="adaptive-σ (hand-crafted)")
        runs.append(("adaptive", None, ad_res))
        if args.compare_calibrated and cal_kwargs is not None:
            print("\n--- Adaptive σ sweep (calibrated) ---")
            ad_cal = alpha_sweep(
                args.alphas, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
                bit_bank, args.ebno, args.ckpt, args.sigma, cal_kwargs,
                batch=batch, rounds=rounds,
                mode_label="adaptive-σ (calibrated)")
            runs.append(("adaptive_calibrated", None, ad_cal))
    elif args.adaptive_sigma:
        kwargs = cal_kwargs if cal_kwargs is not None else hand_kwargs
        label = ("adaptive-σ (calibrated)" if cal_kwargs is not None
                 else "adaptive-σ (hand-crafted)")
        print(f"--- {label} sweep ---")
        ad_res = alpha_sweep(
            args.alphas, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
            bit_bank, args.ebno, args.ckpt, args.sigma, kwargs,
            batch=batch, rounds=rounds, mode_label=label)
        runs.append(("adaptive" if cal_kwargs is None else "adaptive_calibrated",
                     None, ad_res))
    else:
        print(f"--- Fixed σ={args.sigma} sweep ---")
        fixed_res = alpha_sweep(
            args.alphas, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
            bit_bank, args.ebno, args.ckpt, args.sigma, fixed_kwargs,
            batch=batch, rounds=rounds, mode_label=f"fixed-σ={args.sigma}")
        runs.append(("fixed", args.sigma, fixed_res))

    os.makedirs("results", exist_ok=True)
    out_json = f"results/alpha_sweep_beta0_ebno{args.ebno}.json"
    serial = []
    for name, sig, res in runs:
        serial.append({
            "mode": name,
            "fixed_sigma": sig,
            "alphas": {str(k): {"ack": v["ack"], "nack": v["nack"],
                                "bit_errs": v["bit_errs"],
                                "bler": v["nack"] / total,
                                "chunk_diagnostics": v["chunk_diagnostics"]}
                       for k, v in res.items()},
        })
    ckpt_loaded = os.path.isfile(args.ckpt)
    if not ckpt_loaded:
        print(f"\n[WARN] Checkpoint missing ({args.ckpt}); "
              "denoiser weights are random.")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "ebno_db": args.ebno,
            "beta": 0.0,
            "batch": batch,
            "rounds": rounds,
            "checkpoint": args.ckpt,
            "checkpoint_loaded": ckpt_loaded,
            "baseline_bler": base_nack / total,
            "adaptive_sigma_lookup": {
                "syndrome_metric": "normalized_unsatisfied_parity_ratio",
                "monotonic_sigma": bool(args.monotonic_sigma),
                "syndrome_hard_decision": args.syndrome_hard_decision,
                "syndrome_sign_debug": args.syndrome_sign_debug,
                "handcrafted_thresholds": list(hand_info["thresholds"]),
                "handcrafted_sigma_levels": list(hand_info["sigma_levels"]),
                "calibrated_lookup_json": args.sigma_lookup_json,
                "calibrated_per_chunk": (
                    cal_info["per_chunk_calibrated"]
                    if cal_info is not None else None),
                "sigma_min": args.sigma_min,
                "sigma_max": args.sigma_max,
            },
            "runs": serial,
        }, f, indent=2)
    print(f"\nJSON saved → {out_json}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(base_nack / total, color="#888", linestyle="--",
               label=f"Baseline BP (BLER={base_nack/total:.3f})")
    for name, sig, res in runs:
        xs = sorted(res.keys())
        ys = [res[a]["nack"] / total for a in xs]
        if name == "adaptive":
            lab = "Adaptive σ (hand-crafted)"
        elif name == "adaptive_calibrated":
            lab = "Adaptive σ (calibrated)"
        else:
            lab = f"Fixed σ={sig}"
        ax.plot(xs, ys, "o-", linewidth=2, markersize=7, label=lab)
    ax.set_xlabel("α (source extrinsic weight)", fontsize=11)
    ax.set_ylabel("BLER (NACK rate)", fontsize=11)
    ax.set_title(
        f"Fashion-MNIST — α sweep at Eb/N0={args.ebno} dB, β=0\n"
        f"new_input = ch + α·src_ext",
        fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out_png = f"results/alpha_sweep_beta0_ebno{args.ebno}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_png}")


if __name__ == "__main__":
    main()
