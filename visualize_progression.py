"""
BP + denoiser 반복 과정에서 이미지 복원 시각화.

[onlyextrinsic variant — independent alpha/beta, turbo-style extrinsic]
  new_input = channel + beta * bp_ext + alpha * src_ext

  bp_ext  = BP_post  - payload_intr   (BP extrinsic = posterior - a priori input)
  src_ext = src_post - BP_post        (source extrinsic = posterior - denoiser input)

Usage:
  CUDA_VISIBLE_DEVICES=0 python visualize_progression.py --ebno 0.6
  CUDA_VISIBLE_DEVICES=0 python visualize_progression.py --img_idx 3 --ebno 0.55 --alpha 0.1 --beta 0
"""
import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


from cli_common import (
    add_adaptive_sigma_args,
    decoder_sigma_kwargs,
    early_gpu_mb_argv,
)

_mb = early_gpu_mb_argv(sys.argv)
if _mb is not None and _mb > 0:
    os.environ["FMNIST_GPU_MEM_MB"] = str(_mb)

import numpy as np
import torch
import tensorflow as tf
from gpu_limits import setup_tensorflow_gpu

setup_tensorflow_gpu()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPCBPDecoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no

from decoder import LDPC5GDecoder_soft
from syndrome_sigma_schedule import compute_syndrome_weight_and_ratio

import torchvision

IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP  # 6272

CRC_DEGREE = "CRC24A"
N_CODEWORD = 12600
NUM_BPS = 1


def llr_to_image_np(llr_tf, k_payload=K_PAYLOAD):
    llr = llr_tf[0, :k_payload].numpy()
    p = 1.0 / (1.0 + np.exp(-np.clip(llr, -20, 20)))
    weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.float32)
    p = p.reshape(-1, BPP)
    pixel = (p * weights).sum(axis=1) / 255.0
    return pixel.reshape(IMG_H, IMG_W)


def compute_ber(llr_tf, orig_bits, k_payload=K_PAYLOAD):
    dec_bits = (llr_tf[0, :k_payload].numpy() > 0).astype(np.int32)
    return np.mean(dec_bits != orig_bits[:k_payload])


def compute_mse(soft_img, orig_img):
    return float(np.mean((soft_img - orig_img) ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_idx", type=int, default=3)
    p.add_argument("--ebno", type=float, default=0.6)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--schedule", type=int, nargs="+", default=[10, 10, 10])
    p.add_argument("--ckpt", default="checkpoints/denoiser.pt")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gpu-memory-mb", type=int, default=None,
                   help="GPU memory cap (MB).")
    add_adaptive_sigma_args(p, fixed_sigma_default=0.3)
    args = p.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    schedule = args.schedule

    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    k_ldpc = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)

    sched_kwargs, sched_info = decoder_sigma_kwargs(args)
    dec = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=schedule,
        alpha=args.alpha, beta=args.beta, k_payload=K_PAYLOAD,
        cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(schedule), llr_max=30.0,
        **sched_kwargs)

    if os.path.isfile(args.ckpt):
        dec.denoiser.load_weights_pt(args.ckpt)
        print(f"[INFO] Loaded {args.ckpt}")
    if not args.adaptive_sigma:
        dec.denoiser.sigma = args.sigma
    prior_model = dec.denoiser.prior_model

    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    awgn = AWGN()

    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=True)
    fmnist_labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    idx = args.img_idx % len(ds)
    pil_img, class_id = ds[idx]
    orig_img = np.array(pil_img, dtype=np.float32) / 255.0
    label = f"{fmnist_labels[class_id]} (#{idx})"
    orig_bytes = np.array(pil_img, dtype=np.uint8).reshape(-1)
    orig_bits = np.unpackbits(orig_bytes).astype(np.int32)
    assert orig_bits.size == K_PAYLOAD, (
        f"Source bit length {orig_bits.size} != expected {K_PAYLOAD}"
    )
    u = tf.constant(orig_bits[None, :], dtype=tf.int32)

    u_crc = crc_enc(tf.cast(u, ldpc_enc.rdtype))
    c = ldpc_enc(u_crc)
    x = mapper(c)
    no = ebnodb2no(args.ebno, NUM_BPS, ldpc_enc.coderate)
    y = awgn(x, no)
    llr_ch = demapper(y, no)

    # ---- step-by-step decoding ----
    llr = tf.reshape(llr_ch, [1, -1])
    if ldpc_enc.num_bits_per_symbol is not None:
        llr = tf.gather(llr, ldpc_enc.out_int_inv, axis=-1)

    llr_5g = tf.concat([tf.zeros([1, 2 * ldpc_enc.z], dec.rdtype), llr], axis=1)
    k_filler = ldpc_enc.k_ldpc - ldpc_enc.k
    nb_punc_bits = (ldpc_enc.n_ldpc - k_filler) - ldpc_enc.n - 2 * ldpc_enc.z
    llr_5g = tf.concat(
        [llr_5g, tf.zeros([1, nb_punc_bits - dec._nb_pruned_nodes], dec.rdtype)],
        axis=1)

    x1_sys = llr_5g[:, :ldpc_enc.k]
    nb_par = ldpc_enc.n_ldpc - k_filler - ldpc_enc.k - dec._nb_pruned_nodes
    x2_par = llr_5g[:, ldpc_enc.k:ldpc_enc.k + nb_par]
    z_short = -tf.cast(dec._llr_max, dec.rdtype) * tf.ones([1, k_filler], dec.rdtype)

    k_payload = min(K_PAYLOAD, int(ldpc_enc.k))
    payload0 = x1_sys[:, :k_payload]
    crc_rest = x1_sys[:, k_payload:]

    a = tf.cast(args.alpha, dec.rdtype)
    b = tf.cast(args.beta, dec.rdtype)

    # ============ Row 0: Baseline BP (no denoiser) ============
    baseline_row = []
    ch_img = llr_to_image_np(x1_sys, k_payload)
    ch_ber = compute_ber(x1_sys, orig_bits, k_payload)
    ch_mse = compute_mse(ch_img, orig_img)
    baseline_row.append(("Channel\nreception", ch_img, ch_ber, ch_mse))

    bl_intr = payload0
    bl_msg = None
    prev_rs = getattr(dec, "_return_state", False)
    prev_ho = getattr(dec, "_hard_out", False)
    try:
        dec._return_state = True
        dec._hard_out = False
        bl_cum = 0
        for iters in schedule:
            bl_x1 = tf.concat([bl_intr, crc_rest], axis=1)
            bl_llr = tf.concat([bl_x1, z_short, x2_par], axis=1)
            bl_hat, bl_msg = LDPCBPDecoder.call(
                dec, bl_llr, num_iter=int(iters), msg_v2c=bl_msg)
            bl_cum += iters
            bl_img = llr_to_image_np(bl_hat, k_payload)
            bl_ber = compute_ber(bl_hat, orig_bits, k_payload)
            bl_mse = compute_mse(bl_img, orig_img)
            baseline_row.append((f"BP {bl_cum} iter", bl_img, bl_ber, bl_mse))
    finally:
        dec._return_state = prev_rs
        dec._hard_out = prev_ho

    # ============ Row 1: BP + denoiser (BP posterior) ============
    dn_row = []
    dn_row.append(("Channel\nreception", ch_img, ch_ber, ch_mse))

    payload_intr = payload0
    msg_v2c = None
    corr_row = []
    corr_row.append(None)

    try:
        dec._return_state = True
        dec._hard_out = False

        cum_iter = 0
        chunk_logs = []
        for chunk_idx, iters in enumerate(schedule):
            x1_stage = tf.concat([payload_intr, crc_rest], axis=1)
            llr_bp = tf.concat([x1_stage, z_short, x2_par], axis=1)

            x_hat, msg_v2c = LDPCBPDecoder.call(
                dec, llr_bp, num_iter=int(iters), msg_v2c=msg_v2c)

            cum_iter += iters
            bp_img = llr_to_image_np(x_hat, k_payload)
            bp_ber = compute_ber(x_hat, orig_bits, k_payload)
            bp_mse = compute_mse(bp_img, orig_img)
            dn_row.append((f"BP {cum_iter} iter", bp_img, bp_ber, bp_mse))

            if chunk_idx < len(schedule) - 1:
                post_payload = x_hat[:, :k_payload]
                # Turbo-style: subtract current BP a priori input, not channel
                bp_ext = post_payload - payload_intr
                counts, ratios, sign_debug = compute_syndrome_weight_and_ratio(
                    x_hat,
                    dec._h_sparse,
                    decision_rule=args.syndrome_hard_decision,
                    compare_both_signs=args.syndrome_sign_debug,
                )
                if args.adaptive_sigma and dec.sigma_scheduler is not None:
                    batch_sig = dec.sigma_scheduler.select_sigma(chunk_idx, ratios)
                    src_ext = dec._denoiser(post_payload, sigma=batch_sig)
                else:
                    batch_sig = tf.fill(
                        [tf.shape(post_payload)[0]],
                        tf.cast(dec.denoiser.sigma, dec.rdtype))
                    src_ext = dec._denoiser(post_payload)
                chunk_logs.append({
                    "chunk_idx": chunk_idx,
                    "syndrome_weight": float(tf.reduce_mean(
                        tf.cast(counts, tf.float32)).numpy()),
                    "syndrome_ratio": float(tf.reduce_mean(
                        tf.cast(ratios, tf.float32)).numpy()),
                    "sigma": float(tf.reduce_mean(
                        tf.cast(batch_sig, tf.float32)).numpy()),
                    "sign_debug": sign_debug,
                })
                payload_intr = payload0 + b * bp_ext + a * src_ext

                corr_llr = tf.concat([payload_intr, crc_rest], axis=1)
                corr_img = llr_to_image_np(corr_llr, k_payload)
                corr_ber = compute_ber(corr_llr, orig_bits, k_payload)
                corr_mse = compute_mse(corr_img, orig_img)
                corr_row.append(("Corrected\nintrinsic",
                                 corr_img, corr_ber, corr_mse))
            else:
                corr_row.append(None)
    finally:
        dec._return_state = prev_rs
        dec._hard_out = prev_ho

    hat_info = x_hat[:, :int(ldpc_enc.k)]
    _, crc_ok = hard_crc_decode(crc_dec, hat_info)
    crc_pass = bool(crc_ok.numpy()[0])
    final_ber = compute_ber(x_hat, orig_bits, k_payload)

    bl_final_ber = compute_ber(bl_hat, orig_bits, k_payload)
    bl_info = bl_hat[:, :int(ldpc_enc.k)]
    _, bl_crc_ok = hard_crc_decode(crc_dec, bl_info)
    bl_crc_pass = bool(bl_crc_ok.numpy()[0])

    print(f"\n{'='*55}")
    print(f"  Image #{idx} ({label}) | Eb/N0={args.ebno} dB")
    sig_note = "adaptive-σ" if args.adaptive_sigma else f"σ={args.sigma}"
    print(f"  α={args.alpha}  β={args.beta}  {sig_note}")
    print(f"{'='*55}")
    print(f"  [Baseline BP]")
    for title, _, ber, mse in baseline_row:
        print(f"    {title.replace(chr(10),' '):20s}  BER={ber:.4f}  MSE={mse:.4f}")
    print(f"    {'Final':20s}  BER={bl_final_ber:.4f}  "
          f"CRC={'PASS' if bl_crc_pass else 'FAIL'}")
    print(f"  [BP + Denoiser]")
    for title, _, ber, mse in dn_row:
        print(f"    {title.replace(chr(10),' '):20s}  BER={ber:.4f}  MSE={mse:.4f}")
    print(f"    {'Final':20s}  BER={final_ber:.4f}  "
          f"CRC={'PASS' if crc_pass else 'FAIL'}")
    print("  [Chunk syndrome/sigma]")
    for d in chunk_logs:
        print("    "
              f"chunk{d['chunk_idx']}: w={d['syndrome_weight']:.1f}  "
              f"ratio={d['syndrome_ratio']:.4f}  sigma={d['sigma']:.3f}")
        if d["sign_debug"] is not None:
            pmean = float(tf.reduce_mean(
                tf.cast(d["sign_debug"]["primary_ratios"], tf.float32)).numpy())
            amean = float(tf.reduce_mean(
                tf.cast(d["sign_debug"]["alt_ratios"], tf.float32)).numpy())
            better = (d["sign_debug"]["primary_rule"]
                      if pmean <= amean else d["sign_debug"]["alt_rule"])
            print("      "
                  f"sign-check: {d['sign_debug']['primary_rule']}={pmean:.4f}, "
                  f"{d['sign_debug']['alt_rule']}={amean:.4f}, better={better}")
    if args.syndrome_sign_debug and chunk_logs:
        prim_vals, alt_vals = [], []
        for d in chunk_logs:
            sd = d.get("sign_debug")
            if sd is None:
                continue
            prim_vals.append(float(tf.reduce_mean(
                tf.cast(sd["primary_ratios"], tf.float32)).numpy()))
            alt_vals.append(float(tf.reduce_mean(
                tf.cast(sd["alt_ratios"], tf.float32)).numpy()))
        if prim_vals:
            mp, ma = float(np.mean(prim_vals)), float(np.mean(alt_vals))
            sd0 = next(d["sign_debug"] for d in chunk_logs if d.get("sign_debug"))
            better = sd0["primary_rule"] if mp <= ma else sd0["alt_rule"]
            print("  [Sign convention summary — mean syndrome ratio over denoiser chunks]")
            print(f"    {sd0['primary_rule']}: {mp:.6f}")
            print(f"    {sd0['alt_rule']}: {ma:.6f}")
            print(f"    smaller_is_better_convention: {better}")
    print(f"{'='*55}")

    # ---- plot (3 rows) ----
    n_cols = len(baseline_row) + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(2.8 * n_cols, 10.5),
                             gridspec_kw={"height_ratios": [1, 1, 1],
                                          "hspace": 0.50})

    row_colors = ["#999999", "#d62728", "#1f77b4"]
    row_labels = [
        "Baseline BP\n(no denoiser)",
        f"BP + Denoiser\n(α={args.alpha}, β={args.beta})",
        "Corrected intrinsic\n(ch + β·bp_ext + α·src_ext)",
    ]
    all_rows = [baseline_row, dn_row, corr_row]

    for r in range(3):
        if r < 2:
            axes[r, 0].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
            if r == 0:
                axes[r, 0].set_title(f"Original\n({label})", fontsize=9,
                                     fontweight="bold")
            else:
                axes[r, 0].set_title("Original", fontsize=9, color="gray")
        else:
            axes[r, 0].set_visible(False)
        axes[r, 0].axis("off")
        axes[r, 0].set_ylabel(row_labels[r], fontsize=8, rotation=90,
                              labelpad=12, color=row_colors[r])

    for r, row_data in enumerate(all_rows):
        clr = row_colors[r]
        for i, entry in enumerate(row_data):
            ax = axes[r, i + 1]
            if entry is not None:
                if len(entry) == 4:
                    title, img, ber, mse = entry
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                    ax.set_title(title, fontsize=9, color=clr)
                    ax.set_xlabel(f"BER={ber:.4f}\nMSE={mse:.4f}",
                                  fontsize=8, color=clr)
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    ax.axis("off")
            else:
                ax.axis("off")

    bl_crc_str = "PASS" if bl_crc_pass else "FAIL"
    dn_crc_str = "PASS" if crc_pass else "FAIL"
    fig.suptitle(
        f"Eb/N0={args.ebno} dB | schedule={schedule} | "
        f"{sig_note}\n"
        f"Baseline: BER={bl_final_ber:.4f} CRC {bl_crc_str}  |  "
        f"Denoiser (α={args.alpha}, β={args.beta}): "
        f"BER={final_ber:.4f} CRC {dn_crc_str}",
        fontsize=10, y=0.99)

    fig.subplots_adjust(left=0.07, right=0.96, top=0.92, bottom=0.03)
    os.makedirs("results", exist_ok=True)
    out = f"results/progression_idx{idx}_ebno{args.ebno}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
