"""
Measure whether the fixed EDM denoiser input sigma matches the actual pixel
RMSE of the legacy LDPC decoder's source input.

This is a diagnostic-only probe:

* legacy 5G-LDPC path, [5]x20, alpha=beta=0.1, source_input_mode="bp_post";
* explicit BPSK Es/N0 (default -2.5 dB), AWGN, perfect CSI;
* the production trajectory always uses fixed sigma=0.3;
* fixed-sigma and RMSE-matched denoiser outputs are side probes only and are
  never fed back into BP;
* no CRC/BLER or end-to-end performance metric is computed.

The legacy decoder has 20 BP chunks but only 19 source-feedback calls.  The
20th BP output is included as a terminal, diagnostic-only denoiser probe so the
requested trajectory has 20 points.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import torch
from sionna.phy.channel.awgn import AWGN
from sionna.phy.fec.crc import CRCEncoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.mapping import Demapper, Mapper

from decoder import LDPC5GDecoder_soft


IMG_H = 28
IMG_W = 28
BPP = 8
K_PAYLOAD = IMG_H * IMG_W * BPP
N_CODEWORD = 12600
SCHEDULE = [5] * 20
FIXED_SIGMA = 0.3
SIGMA_FLOOR = 1e-6


def describe(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
    }


def rmse_per_block(a, b):
    dims = tuple(range(1, a.ndim))
    return torch.sqrt(torch.mean((a - b) ** 2, dim=dims))


def bit_accuracy(logits, truth_bits):
    hard = logits.reshape(-1, IMG_H * IMG_W, BPP) > 0
    truth = truth_bits.reshape(-1, IMG_H * IMG_W, BPP).bool()
    equal = hard == truth
    overall = float(equal.float().mean().item())
    per_plane = equal.float().mean(dim=(0, 1)).cpu().numpy().astype(float)
    return overall, per_plane.tolist()


def load_fashion_mnist():
    import torchvision

    dataset = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=False)
    images = dataset.data.numpy().astype(np.uint8)
    bits = np.unpackbits(images.reshape(-1, IMG_H * IMG_W), axis=1)
    return images, tf.constant(bits, dtype=tf.int32)


def build_decoder(ldpc, checkpoint, device):
    common = dict(
        cn_update="boxplus-phi",
        vn_update="sum",
        cn_schedule="flooding",
        hard_out=False,
        return_infobits=True,
        llr_max=30.0,
    )
    decoder = LDPC5GDecoder_soft(
        ldpc,
        k_payload=K_PAYLOAD,
        num_iter=sum(SCHEDULE),
        bp_schedule=SCHEDULE,
        alpha=0.1,
        beta=0.1,
        ep_mode=False,
        source_input_mode="bp_post",
        adaptive_sigma=False,
        ep_track_payload_hist=True,
        denoiser_kwargs=dict(device=device),
        **common,
    )
    decoder.denoiser.load_weights_pt(checkpoint)
    decoder.denoiser.sigma = FIXED_SIGMA
    return decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=32, choices=(32, 64))
    parser.add_argument("--esn0-db", type=float, default=-2.5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--checkpoint",
        default="/home/LJH/onlyextrinsic_ada_sigma/checkpoints/denoiser.pt",
    )
    parser.add_argument(
        "--output",
        default="results/denoiser_sigma_alignment_32.json",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, bit_bank = load_fashion_mnist()

    crc = CRCEncoder("CRC24A")
    ldpc = LDPC5GEncoder(
        K_PAYLOAD + crc.crc_length,
        N_CODEWORD,
        num_bits_per_symbol=1,
    )
    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=1)
    demapper = Demapper(
        "app", constellation_type="pam", num_bits_per_symbol=1)
    awgn = AWGN()
    decoder = build_decoder(ldpc, str(checkpoint), device)

    tf.random.set_seed(args.seed)
    indices = tf.random.uniform(
        [args.blocks], 0, tf.shape(bit_bank)[0], dtype=tf.int32)
    payload = tf.gather(bit_bank, indices)

    # Explicit BPSK Es/N0. For unit-energy symbols, N0=10^(-EsN0/10).
    noise_variance = tf.cast(
        10.0 ** (-args.esn0_db / 10.0), ldpc.rdtype)
    codeword = ldpc(crc(tf.cast(payload, ldpc.rdtype)))
    received = awgn(mapper(codeword), noise_variance)
    channel_llr = demapper(received, noise_variance)

    # This single production decode creates the fixed-sigma=0.3 legacy
    # trajectory. Alternative denoiser calls below never feed back into it.
    _ = decoder(channel_llr)
    payload_history = decoder.last_payload_hist
    if len(payload_history) != len(SCHEDULE):
        raise RuntimeError(
            f"expected {len(SCHEDULE)} BP chunks, got {len(payload_history)}")
    if len(decoder.last_chunk_diagnostics) != len(SCHEDULE) - 1:
        raise RuntimeError(
            "legacy decoder should have 19 source-feedback calls for 20 chunks")

    idx_np = indices.numpy()
    truth_image = torch.from_numpy(
        images[idx_np].astype(np.float32) / 255.0
    ).reshape(args.blocks, 1, IMG_H, IMG_W).to(device)
    truth_bits = torch.from_numpy(payload.numpy()).to(device)
    prior = decoder.denoiser.prior_model

    chunk_results = []
    sigma_by_chunk_and_block = []
    fixed_accuracy = []
    matched_accuracy = []
    fixed_move = []
    matched_move = []

    with torch.no_grad():
        for chunk_idx, source_input_tf in enumerate(payload_history):
            source_input = torch.from_numpy(
                source_input_tf[:, :K_PAYLOAD].numpy()
            ).float().to(device)
            mu_cavity = prior.llr_to_soft_field(source_input)

            block_sigma = rmse_per_block(mu_cavity, truth_image)
            sigma_used = block_sigma.clamp(min=SIGMA_FLOOR)
            n_floored = int((block_sigma < SIGMA_FLOOR).sum().item())

            # One batched network call evaluates both sigma choices on exactly
            # the same cavity input.
            paired_mu = torch.cat([mu_cavity, mu_cavity], dim=0)
            paired_sigma = torch.cat([
                torch.full_like(block_sigma, FIXED_SIGMA),
                sigma_used,
            ])
            paired_output = prior.net(paired_mu, paired_sigma).clamp(0.0, 1.0)
            if not torch.isfinite(paired_output).all():
                raise RuntimeError(
                    f"non-finite denoiser output at chunk {chunk_idx + 1}")
            fixed_output, matched_output = paired_output.chunk(2, dim=0)

            fixed_logits = prior.soft_field_to_posterior_logits(fixed_output)
            matched_logits = prior.soft_field_to_posterior_logits(matched_output)
            input_acc, input_plane = bit_accuracy(source_input, truth_bits)
            fixed_acc, fixed_plane = bit_accuracy(fixed_logits, truth_bits)
            matched_acc, matched_plane = bit_accuracy(
                matched_logits, truth_bits)

            fixed_move_block = rmse_per_block(fixed_output, mu_cavity)
            matched_move_block = rmse_per_block(matched_output, mu_cavity)
            fixed_true_block = rmse_per_block(fixed_output, truth_image)
            matched_true_block = rmse_per_block(matched_output, truth_image)

            sigma_np = block_sigma.cpu().numpy()
            sigma_by_chunk_and_block.append(sigma_np.tolist())
            sigma_rms = float(torch.sqrt(
                torch.mean((mu_cavity - truth_image) ** 2)).item())

            row = {
                "chunk": chunk_idx + 1,
                "bp_iterations_cumulative": (chunk_idx + 1) * 5,
                "used_for_legacy_feedback": chunk_idx < len(SCHEDULE) - 1,
                "sigma_actual_rms": sigma_rms,
                "sigma_actual_over_fixed": sigma_rms / FIXED_SIGMA,
                "sigma_actual_block_distribution": describe(sigma_np),
                "sigma_floor": SIGMA_FLOOR,
                "sigma_floor_count": n_floored,
                "input_hard_bit_accuracy": input_acc,
                "input_hard_bit_plane_accuracy": input_plane,
                "fixed_sigma": {
                    "sigma": FIXED_SIGMA,
                    "posterior_bit_accuracy": fixed_acc,
                    "posterior_bit_plane_accuracy": fixed_plane,
                    "output_minus_input_rmse": describe(
                        fixed_move_block.cpu().numpy()),
                    "output_minus_truth_rmse": describe(
                        fixed_true_block.cpu().numpy()),
                },
                "matched_sigma": {
                    "posterior_bit_accuracy": matched_acc,
                    "posterior_bit_plane_accuracy": matched_plane,
                    "output_minus_input_rmse": describe(
                        matched_move_block.cpu().numpy()),
                    "output_minus_truth_rmse": describe(
                        matched_true_block.cpu().numpy()),
                },
                "matched_minus_fixed": {
                    "posterior_bit_accuracy": matched_acc - fixed_acc,
                    "posterior_bit_plane_accuracy": (
                        np.asarray(matched_plane) -
                        np.asarray(fixed_plane)
                    ).tolist(),
                    "output_minus_input_rmse_mean": (
                        float(matched_move_block.mean().item()) -
                        float(fixed_move_block.mean().item())
                    ),
                },
            }
            chunk_results.append(row)
            fixed_accuracy.append(fixed_acc)
            matched_accuracy.append(matched_acc)
            fixed_move.append(float(fixed_move_block.mean().item()))
            matched_move.append(float(matched_move_block.mean().item()))

            print(
                f"chunk {chunk_idx + 1:02d}: "
                f"sigma_actual={sigma_rms:.6f} "
                f"({sigma_rms / FIXED_SIGMA:.3f}x of 0.3), "
                f"bit_acc fixed/matched={fixed_acc:.4f}/{matched_acc:.4f}, "
                f"move fixed/matched={fixed_move[-1]:.6f}/"
                f"{matched_move[-1]:.6f}",
                flush=True,
            )

    sigma_matrix = np.asarray(sigma_by_chunk_and_block, dtype=np.float64)
    per_block_mean_sigma = sigma_matrix.mean(axis=0)
    trajectory = np.asarray(
        [row["sigma_actual_rms"] for row in chunk_results])
    summary = {
        "sigma_actual_trajectory": trajectory.tolist(),
        "sigma_actual_over_fixed_trajectory": (
            trajectory / FIXED_SIGMA).tolist(),
        "chunks_below_0_15": int(np.sum(trajectory < 0.15)),
        "chunks_below_0_3": int(np.sum(trajectory < FIXED_SIGMA)),
        "block_chunk_fraction_below_0_15": float(
            np.mean(sigma_matrix < 0.15)),
        "block_chunk_fraction_below_0_3": float(
            np.mean(sigma_matrix < FIXED_SIGMA)),
        "per_block_mean_sigma_distribution": describe(per_block_mean_sigma),
        "fixed_output_bit_accuracy_across_chunks": describe(fixed_accuracy),
        "matched_output_bit_accuracy_across_chunks": describe(
            matched_accuracy),
        "matched_minus_fixed_bit_accuracy_across_chunks": describe(
            np.asarray(matched_accuracy) - np.asarray(fixed_accuracy)),
        "fixed_output_move_rmse_across_chunks": describe(fixed_move),
        "matched_output_move_rmse_across_chunks": describe(matched_move),
        "matched_minus_fixed_move_rmse_across_chunks": describe(
            np.asarray(matched_move) - np.asarray(fixed_move)),
    }

    result = {
        "kind": "diagnostic_only_no_bler",
        "configuration": {
            "branch": "practical_sigma",
            "decoder": "legacy",
            "schedule": SCHEDULE,
            "alpha": 0.1,
            "beta": 0.1,
            "source_input_mode": "bp_post",
            "esn0_db": args.esn0_db,
            "channel": "BPSK/AWGN/perfect_CSI",
            "blocks": args.blocks,
            "seed": args.seed,
            "fixed_sigma": FIXED_SIGMA,
            "sigma_post": float(decoder.denoiser.sigma_post),
            "checkpoint": str(checkpoint),
            "device": device,
            "actual_source_feedback_calls": len(SCHEDULE) - 1,
            "terminal_diagnostic_probe": True,
        },
        "definitions": {
            "mu_cavity": (
                "sum_m 2^(7-m)*sigmoid(source_input_llr_m)/255"
            ),
            "sigma_actual_per_block": (
                "sqrt(mean_pixels((mu_cavity-x0_normalized)^2))"
            ),
            "sigma_actual_rms_per_chunk": (
                "sqrt(mean_blocks,pixels((mu_cavity-x0_normalized)^2))"
            ),
            "matched_sigma_call": (
                "per-block sigma_actual, floored only at 1e-6 for EDM log(sigma)"
            ),
        },
        "summary": summary,
        "chunks": chunk_results,
        "raw_sigma_actual_by_chunk_and_block": sigma_by_chunk_and_block,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
