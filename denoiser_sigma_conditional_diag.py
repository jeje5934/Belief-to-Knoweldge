"""
Conditional denoiser-sigma diagnostics for the practical_sigma LDPC path.

Two explicitly separated modes are provided:

1. ``profile``: measurement only. Run the current fixed-sigma=0.3 legacy
   decoder, split blocks by final CRC and stable-syndrome convergence time,
   measure per-chunk sigma_actual and receiver-visible proxies, and side-probe
   fixed/matched denoiser outputs without feeding them back.
2. ``schedule``: the one permitted performance check. Compare four predeclared
   sigma schedules on paired payload/noise and report CRC-BLER, Wilson CI, and
   final-block transitions relative to fixed sigma=0.3.

No decoder, denoiser, or source-prior implementation is modified.
"""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import torch
from sionna.phy.channel.awgn import AWGN
from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.mapping import Demapper, Mapper
from syndrome_sigma_schedule import AnnealingSigmaScheduler

from denoiser_sigma_alignment_diag import (
    BPP,
    FIXED_SIGMA,
    IMG_H,
    IMG_W,
    K_PAYLOAD,
    N_CODEWORD,
    SCHEDULE,
    SIGMA_FLOOR,
    build_decoder,
    describe,
    load_fashion_mnist,
    rmse_per_block,
)


SOURCE_CALLS = len(SCHEDULE) - 1
# Only one block at Es/N0=-2.9 dB reached stable syndrome zero by chunk 5.
# Use <=7 as a statistically useful early cohort while preserving the
# originally requested >=10 definition for late convergence.
EARLY_CHUNK_MAX = 7
LATE_CHUNK_MIN = 10


class StaticSigmaScheduler:
    """Open-loop schedule using the existing decoder sigma_scheduler API."""

    name = "static_profile_schedule"
    is_adaptive = True

    def __init__(self, values):
        values = [float(v) for v in values]
        if len(values) != SOURCE_CALLS:
            raise ValueError(
                f"expected {SOURCE_CALLS} sigma values, got {len(values)}")
        if min(values) <= 0:
            raise ValueError("all sigma values must be positive")
        self.values = values

    def select_sigma(self, chunk_idx, ratios):
        value = self.values[min(int(chunk_idx), len(self.values) - 1)]
        return tf.fill(
            [tf.shape(ratios)[0]], tf.cast(value, ratios.dtype))


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    half = z / denominator * math.sqrt(
        p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return [max(0.0, center - half), min(1.0, center + half)]


def rankdata(values):
    """Average ranks with tie handling, without a scipy dependency."""
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"n": int(len(x)), "pearson": None, "spearman": None}
    return {
        "n": int(len(x)),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
    }


def stable_zero_chunk(syndrome):
    """First 1-based chunk from which syndrome stays zero through chunk 19."""
    zero = np.asarray(syndrome) == 0
    for idx in range(len(zero)):
        if np.all(zero[idx:]):
            return idx + 1
    return None


def first_zero_chunk(syndrome):
    hits = np.flatnonzero(np.asarray(syndrome) == 0)
    return int(hits[0] + 1) if len(hits) else None


def classify_block(final_success, stable_chunk):
    if not final_success:
        return "failure"
    if stable_chunk is not None and stable_chunk <= EARLY_CHUNK_MAX:
        return "early_success"
    if stable_chunk is None or stable_chunk >= LATE_CHUNK_MIN:
        return "late_success"
    return "middle_success"


def make_system():
    crc_encoder = CRCEncoder("CRC24A")
    crc_decoder = CRCDecoder(crc_encoder)
    ldpc = LDPC5GEncoder(
        K_PAYLOAD + crc_encoder.crc_length,
        N_CODEWORD,
        num_bits_per_symbol=1,
    )
    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=1)
    demapper = Demapper(
        "app", constellation_type="pam", num_bits_per_symbol=1)
    return (
        crc_encoder,
        crc_decoder,
        ldpc,
        mapper,
        demapper,
        AWGN(),
    )


def make_channel_batch(
    round_idx,
    batch,
    seed,
    esn0_db,
    images,
    bit_bank,
    crc_encoder,
    ldpc,
    mapper,
    demapper,
    awgn,
):
    indices = tf.random.stateless_uniform(
        [batch],
        seed=tf.constant([seed, round_idx], dtype=tf.int32),
        minval=0,
        maxval=tf.shape(bit_bank)[0],
        dtype=tf.int32,
    )
    payload = tf.gather(bit_bank, indices)
    noise_variance = tf.cast(10.0 ** (-esn0_db / 10.0), ldpc.rdtype)
    codeword = ldpc(crc_encoder(tf.cast(payload, ldpc.rdtype)))
    transmitted = mapper(codeword)
    # Sionna AWGN uses a unit-power circular complex Gaussian multiplied by
    # sqrt(N0). Generate the same distribution statelessly so profile and
    # schedule runs can reproduce identical payload/noise across processes.
    if transmitted.dtype.is_complex:
        real_dtype = transmitted.dtype.real_dtype
        noise_real = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=tf.constant([seed + 1, round_idx], dtype=tf.int32),
            dtype=real_dtype,
        )
        noise_imag = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=tf.constant([seed + 2, round_idx], dtype=tf.int32),
            dtype=real_dtype,
        )
        unit_noise = tf.complex(noise_real, noise_imag) / tf.cast(
            math.sqrt(2.0), transmitted.dtype)
    else:
        unit_noise = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=tf.constant([seed + 1, round_idx], dtype=tf.int32),
            dtype=transmitted.dtype,
        )
    received = transmitted + unit_noise * tf.cast(
        tf.sqrt(noise_variance), transmitted.dtype)
    channel_llr = demapper(received, noise_variance)
    truth_image = images[indices.numpy()].astype(np.float32) / 255.0
    return indices.numpy(), payload, truth_image, channel_llr


def plane_accuracy_per_block(logits, truth_bits):
    hard = logits.reshape(-1, IMG_H * IMG_W, BPP) > 0
    truth = truth_bits.reshape(-1, IMG_H * IMG_W, BPP).bool()
    return (hard == truth).float().mean(dim=1).cpu().numpy()


def trajectory_summary(matrix, indices):
    values = np.asarray(matrix, dtype=np.float64)[indices]
    if len(values) == 0:
        return []
    result = []
    for chunk_idx in range(values.shape[1]):
        x = values[:, chunk_idx]
        dist = describe(x)
        result.append({
            "chunk": chunk_idx + 1,
            "rms": float(np.sqrt(np.mean(x ** 2))),
            **dist,
        })
    return result


def proxy_trajectory_summary(matrix, indices):
    values = np.asarray(matrix, dtype=np.float64)[indices]
    if len(values) == 0:
        return []
    result = []
    for chunk_idx in range(values.shape[1]):
        x = values[:, chunk_idx]
        finite = x[np.isfinite(x)]
        result.append({
            "chunk": chunk_idx + 1,
            **(describe(finite) if len(finite) else {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "q10": None,
                "q25": None,
                "median": None,
                "q75": None,
                "q90": None,
                "max": None,
            }),
        })
    return result


def group_correlations(sigma, syndrome, mean_abs_llr, fixed_move, indices):
    sigma = np.asarray(sigma, dtype=np.float64)[indices, :SOURCE_CALLS]
    syndrome = np.asarray(syndrome, dtype=np.float64)[indices]
    mean_abs_llr = np.asarray(mean_abs_llr, dtype=np.float64)[
        indices, :SOURCE_CALLS]
    fixed_move = np.asarray(fixed_move, dtype=np.float64)[
        indices, :SOURCE_CALLS]
    if sigma.size == 0:
        return {}

    chunks = np.broadcast_to(
        np.arange(1, SOURCE_CALLS + 1, dtype=np.float64), sigma.shape)
    proxies = {
        "chunk_index": chunks,
        "syndrome_weight": syndrome,
        "mean_abs_llr": mean_abs_llr,
        "fixed_output_move": fixed_move,
    }
    pooled = {
        name: correlation(values.reshape(-1), sigma.reshape(-1))
        for name, values in proxies.items()
    }

    residual = {}
    sigma_res = sigma - np.mean(sigma, axis=0, keepdims=True)
    for name, values in proxies.items():
        if name == "chunk_index":
            continue
        values_res = values - np.mean(values, axis=0, keepdims=True)
        residual[name] = correlation(
            values_res.reshape(-1), sigma_res.reshape(-1))

    chunk_mean = np.mean(sigma, axis=0, keepdims=True)
    total_ss = float(np.sum((sigma - np.mean(sigma)) ** 2))
    residual_ss = float(np.sum((sigma - chunk_mean) ** 2))
    chunk_r2 = (
        None if total_ss == 0.0 else 1.0 - residual_ss / total_ss)
    return {
        "pooled": pooled,
        "within_chunk_residual": residual,
        "chunk_index_r_squared": chunk_r2,
    }


def run_profile(args):
    (
        crc_encoder,
        crc_decoder,
        ldpc,
        mapper,
        demapper,
        awgn,
    ) = make_system()
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = build_decoder(ldpc, args.checkpoint, device)

    n_rounds = args.blocks // args.batch
    if n_rounds * args.batch != args.blocks:
        raise ValueError("blocks must be divisible by batch")

    raw = {
        "index": [],
        "final_crc_success": [],
        "first_zero_chunk": [],
        "stable_zero_chunk": [],
        "group": [],
        "sigma_actual": [],
        "syndrome_weight": [],
        "mean_abs_llr": [],
        "fixed_output_move": [],
        "input_plane_accuracy": [],
        "fixed_plane_accuracy": [],
        "matched_plane_accuracy": [],
    }

    prior = decoder.denoiser.prior_model
    for round_idx in range(n_rounds):
        (
            indices,
            payload,
            truth_image_np,
            channel_llr,
        ) = make_channel_batch(
            round_idx,
            args.batch,
            args.seed,
            args.esn0_db,
            images,
            bit_bank,
            crc_encoder,
            ldpc,
            mapper,
            demapper,
            awgn,
        )
        final_logits = decoder(channel_llr)
        _, crc_valid = hard_crc_decode(crc_decoder, final_logits)
        final_success = np.asarray(crc_valid.numpy()).reshape(-1).astype(bool)
        payload_history = decoder.last_payload_hist
        diagnostics = decoder.last_chunk_diagnostics
        if len(payload_history) != len(SCHEDULE):
            raise RuntimeError("expected 20 payload-history tensors")
        if len(diagnostics) != SOURCE_CALLS:
            raise RuntimeError("expected 19 source-call diagnostics")

        syndrome = np.stack([
            np.asarray(row["syndrome_weight"].numpy()).reshape(-1)
            for row in diagnostics
        ], axis=1)
        truth_image = torch.from_numpy(truth_image_np).reshape(
            args.batch, 1, IMG_H, IMG_W).to(device)
        truth_bits = torch.from_numpy(payload.numpy()).to(device)

        batch_sigma = []
        batch_abs_llr = []
        batch_fixed_move = []
        batch_input_plane = []
        batch_fixed_plane = []
        batch_matched_plane = []
        with torch.no_grad():
            for source_input_tf in payload_history:
                source_input = torch.from_numpy(
                    source_input_tf[:, :K_PAYLOAD].numpy()
                ).float().to(device)
                mu_cavity = prior.llr_to_soft_field(source_input)
                sigma_actual = rmse_per_block(mu_cavity, truth_image)
                matched_sigma = sigma_actual.clamp(min=SIGMA_FLOOR)

                paired_mu = torch.cat([mu_cavity, mu_cavity], dim=0)
                paired_sigma = torch.cat([
                    torch.full_like(sigma_actual, FIXED_SIGMA),
                    matched_sigma,
                ])
                paired_output = prior.net(
                    paired_mu, paired_sigma).clamp(0.0, 1.0)
                if not torch.isfinite(paired_output).all():
                    raise RuntimeError("non-finite denoiser output")
                fixed_output, matched_output = paired_output.chunk(2, dim=0)
                fixed_logits = prior.soft_field_to_posterior_logits(
                    fixed_output)
                matched_logits = prior.soft_field_to_posterior_logits(
                    matched_output)

                batch_sigma.append(sigma_actual.cpu().numpy())
                batch_abs_llr.append(
                    source_input.abs().mean(dim=1).cpu().numpy())
                batch_fixed_move.append(
                    rmse_per_block(fixed_output, mu_cavity).cpu().numpy())
                batch_input_plane.append(
                    plane_accuracy_per_block(source_input, truth_bits))
                batch_fixed_plane.append(
                    plane_accuracy_per_block(fixed_logits, truth_bits))
                batch_matched_plane.append(
                    plane_accuracy_per_block(matched_logits, truth_bits))

        batch_sigma = np.stack(batch_sigma, axis=1)
        batch_abs_llr = np.stack(batch_abs_llr, axis=1)
        batch_fixed_move = np.stack(batch_fixed_move, axis=1)
        batch_input_plane = np.stack(batch_input_plane, axis=1)
        batch_fixed_plane = np.stack(batch_fixed_plane, axis=1)
        batch_matched_plane = np.stack(batch_matched_plane, axis=1)

        for block_idx in range(args.batch):
            first_zero = first_zero_chunk(syndrome[block_idx])
            stable_zero = stable_zero_chunk(syndrome[block_idx])
            group = classify_block(final_success[block_idx], stable_zero)
            raw["index"].append(int(indices[block_idx]))
            raw["final_crc_success"].append(
                bool(final_success[block_idx]))
            raw["first_zero_chunk"].append(first_zero)
            raw["stable_zero_chunk"].append(stable_zero)
            raw["group"].append(group)
            raw["sigma_actual"].append(
                batch_sigma[block_idx].astype(float).tolist())
            raw["syndrome_weight"].append(
                syndrome[block_idx].astype(float).tolist())
            raw["mean_abs_llr"].append(
                batch_abs_llr[block_idx].astype(float).tolist())
            raw["fixed_output_move"].append(
                batch_fixed_move[block_idx].astype(float).tolist())
            raw["input_plane_accuracy"].append(
                batch_input_plane[block_idx].astype(float).tolist())
            raw["fixed_plane_accuracy"].append(
                batch_fixed_plane[block_idx].astype(float).tolist())
            raw["matched_plane_accuracy"].append(
                batch_matched_plane[block_idx].astype(float).tolist())

        print(
            f"profile round {round_idx + 1}/{n_rounds}: "
            f"CRC failures={int((~final_success).sum())}/{args.batch}",
            flush=True,
        )

    groups = {}
    raw_groups = np.asarray(raw["group"])
    for group_name in [
        "all",
        "success_all",
        "failure",
        "early_success",
        "middle_success",
        "late_success",
    ]:
        if group_name == "all":
            mask = np.ones(args.blocks, dtype=bool)
        elif group_name == "success_all":
            mask = np.asarray(raw["final_crc_success"], dtype=bool)
        else:
            mask = raw_groups == group_name
        indices = np.flatnonzero(mask)
        plane_summary = {}
        for field in [
            "input_plane_accuracy",
            "fixed_plane_accuracy",
            "matched_plane_accuracy",
        ]:
            array = np.asarray(raw[field], dtype=np.float64)
            plane_summary[field] = (
                np.mean(array[indices], axis=0).tolist()
                if len(indices) else []
            )
        sigma_array = np.asarray(raw["sigma_actual"], dtype=np.float64)
        per_block_mean = (
            sigma_array[indices].mean(axis=1) if len(indices)
            else np.array([])
        )
        groups[group_name] = {
            "n_blocks": int(len(indices)),
            "sigma_actual": trajectory_summary(
                raw["sigma_actual"], indices),
            "syndrome_weight": proxy_trajectory_summary(
                raw["syndrome_weight"], indices),
            "mean_abs_llr": proxy_trajectory_summary(
                raw["mean_abs_llr"], indices),
            "fixed_output_move": proxy_trajectory_summary(
                raw["fixed_output_move"], indices),
            "per_block_mean_sigma_distribution": (
                describe(per_block_mean) if len(per_block_mean) else None
            ),
            "correlations": group_correlations(
                raw["sigma_actual"],
                raw["syndrome_weight"],
                raw["mean_abs_llr"],
                raw["fixed_output_move"],
                indices,
            ),
            "bit_plane_trajectory": plane_summary,
        }

    previous = None
    if args.previous_result and Path(args.previous_result).is_file():
        previous_data = json.loads(
            Path(args.previous_result).read_text(encoding="utf-8"))
        previous = previous_data["summary"]["sigma_actual_trajectory"]

    result = {
        "kind": "conditional_sigma_profile_no_bler_experiment",
        "configuration": {
            "branch": "practical_sigma",
            "decoder": "legacy",
            "schedule": SCHEDULE,
            "alpha": 0.1,
            "beta": 0.1,
            "fixed_sigma": FIXED_SIGMA,
            "sigma_post": float(decoder.denoiser.sigma_post),
            "source_input_mode": "bp_post",
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "channel": "BPSK/AWGN/perfect_CSI",
            "early_definition": (
                "final CRC pass and stable syndrome zero by chunk 7 "
                "(only one block reached zero by chunk 5)"
            ),
            "late_definition": (
                "final CRC pass and stable syndrome zero at/after chunk 10, "
                "or no stable zero by chunk 19"
            ),
            "actual_source_feedback_calls": SOURCE_CALLS,
            "terminal_diagnostic_probe": True,
        },
        "previous_success_dominated_32_trajectory": previous,
        "groups": groups,
        "raw": raw,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "group counts:",
        {name: data["n_blocks"] for name, data in groups.items()},
        flush=True,
    )
    print(f"saved {output}", flush=True)


def make_candidate_schedules(profile, scale):
    groups = profile["groups"]
    if groups["failure"]["n_blocks"] >= 5:
        difficult_group = "failure"
    elif groups["late_success"]["n_blocks"] >= 5:
        difficult_group = "late_success"
    else:
        difficult_group = "all"
    difficult = groups[difficult_group]["sigma_actual"]
    measured = [
        min(FIXED_SIGMA, max(0.01, scale * row["rms"]))
        for row in difficult[:SOURCE_CALLS]
    ]
    return difficult_group, {
        "fixed_0.3": None,
        "geom_0.3_to_0.05": AnnealingSigmaScheduler(
            0.3, 0.05, SOURCE_CALLS, mode="geom"),
        "geom_0.3_to_0.01": AnnealingSigmaScheduler(
            0.3, 0.01, SOURCE_CALLS, mode="geom"),
        f"profile_{scale:g}x_{difficult_group}": StaticSigmaScheduler(
            measured),
    }, measured


def run_schedule(args):
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    if float(profile["configuration"]["esn0_db"]) != args.esn0_db:
        raise ValueError("profile and schedule Es/N0 differ")
    (
        crc_encoder,
        crc_decoder,
        ldpc,
        mapper,
        demapper,
        awgn,
    ) = make_system()
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = build_decoder(ldpc, args.checkpoint, device)
    difficult_group, candidates, measured_schedule = make_candidate_schedules(
        profile, args.profile_scale)

    n_rounds = args.blocks // args.batch
    if n_rounds * args.batch != args.blocks:
        raise ValueError("blocks must be divisible by batch")
    success_masks = {name: [] for name in candidates}

    for round_idx in range(n_rounds):
        _, _, _, channel_llr = make_channel_batch(
            round_idx,
            args.batch,
            args.seed,
            args.esn0_db,
            images,
            bit_bank,
            crc_encoder,
            ldpc,
            mapper,
            demapper,
            awgn,
        )
        for name, scheduler in candidates.items():
            decoder.sigma_scheduler = scheduler
            decoder.denoiser.sigma = FIXED_SIGMA
            final_logits = decoder(channel_llr)
            _, crc_valid = hard_crc_decode(crc_decoder, final_logits)
            success_masks[name].extend(
                np.asarray(crc_valid.numpy()).reshape(-1).astype(bool).tolist())
        print(
            f"schedule round {round_idx + 1}/{n_rounds}",
            flush=True,
        )

    baseline = np.asarray(success_masks["fixed_0.3"], dtype=bool)
    results = {}
    for name, success_values in success_masks.items():
        success = np.asarray(success_values, dtype=bool)
        failures = int((~success).sum())
        broken = int(np.sum(baseline & ~success))
        rescued = int(np.sum(~baseline & success))
        results[name] = {
            "failures": failures,
            "blocks": args.blocks,
            "crc_bler": failures / args.blocks,
            "wilson_95": wilson(failures, args.blocks),
            "vs_fixed": {
                "fixed_success_to_candidate_failure": broken,
                "fixed_failure_to_candidate_success": rescued,
                "net_failure_change": broken - rescued,
                "both_failure": int(np.sum(~baseline & ~success)),
                "both_success": int(np.sum(baseline & success)),
            },
            "success_mask": success.tolist(),
        }

    schedule_values = {
        "fixed_0.3": [0.3] * SOURCE_CALLS,
        "geom_0.3_to_0.05": list(
            candidates["geom_0.3_to_0.05"].path),
        "geom_0.3_to_0.01": list(
            candidates["geom_0.3_to_0.01"].path),
        f"profile_{args.profile_scale:g}x_{difficult_group}": measured_schedule,
    }
    result = {
        "kind": "limited_paired_schedule_performance",
        "configuration": {
            "branch": "practical_sigma",
            "decoder": "legacy",
            "schedule": SCHEDULE,
            "alpha": 0.1,
            "beta": 0.1,
            "sigma_post": float(decoder.denoiser.sigma_post),
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "channel": "BPSK/AWGN/perfect_CSI",
            "paired_payload_and_noise": True,
            "profile_source_group": difficult_group,
            "profile_scale": args.profile_scale,
        },
        "schedule_values": schedule_values,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for name, row in results.items():
        print(
            f"{name}: {row['failures']}/{args.blocks} "
            f"BLER={row['crc_bler']:.5f} "
            f"CI={row['wilson_95']} "
            f"broken/rescued={row['vs_fixed']['fixed_success_to_candidate_failure']}/"
            f"{row['vs_fixed']['fixed_failure_to_candidate_success']}",
            flush=True,
        )
    print(f"saved {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--esn0-db", type=float, default=-2.9)
    common.add_argument("--batch", type=int, default=64)
    common.add_argument("--seed", type=int, default=20260724)
    common.add_argument(
        "--checkpoint",
        default="/home/LJH/onlyextrinsic_ada_sigma/checkpoints/denoiser.pt",
    )

    profile = subparsers.add_parser("profile", parents=[common])
    profile.add_argument("--blocks", type=int, default=256)
    profile.add_argument(
        "--previous-result",
        default="results/denoiser_sigma_alignment_32.json",
    )
    profile.add_argument(
        "--output",
        default="results/denoiser_sigma_conditional_256.json",
    )

    schedule = subparsers.add_parser("schedule", parents=[common])
    schedule.add_argument("--blocks", type=int, default=1024)
    schedule.add_argument(
        "--profile",
        default="results/denoiser_sigma_conditional_256.json",
    )
    schedule.add_argument("--profile-scale", type=float, default=2.0)
    schedule.add_argument(
        "--output",
        default="results/denoiser_sigma_schedule_1024.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "profile":
        run_profile(arguments)
    else:
        run_schedule(arguments)
