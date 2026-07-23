#!/usr/bin/env python3
"""Instrument existing RSC and LDPC paths without changing their decisions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.no_ldpc_metrics import (  # noqa: E402
    _uniform_window_ssim,
    payload_bits_to_uint8_images,
)
from experiments.waterfall_common import (  # noqa: E402
    PAYLOAD_BITS,
    TRANSMITTED_BITS,
    awgn_llr,
    load_fashion_mnist_bits,
    paired_batches,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["rsc", "ldpc"], required=True)
    parser.add_argument("--esn0-db", type=float, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--dataset-root", default="/tmp/fmnist")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/LJH/onlyextrinsic_ada_sigma/checkpoints/denoiser.pt"),
    )
    parser.add_argument(
        "--practical-worktree",
        type=Path,
        default=Path("/home/LJH/onlyextrinsic_ada_sigma_practical_sigma"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--legacy-minus-echo",
        action="store_true",
        help="instrument the existing [2]x15 bp_post/minus purity paths",
    )
    parser.add_argument(
        "--paired-score-accounting-only",
        action="store_true",
        help="compare final paired RSC+SPC score-off/on block outcomes",
    )
    parser.add_argument(
        "--unbounded-score-provider",
        action="store_true",
        help="reproduce the pre-audit canonical C used by the 24/256 result",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    return parser.parse_args()


def _finite_or_string(value: float):
    return float(value) if np.isfinite(value) else "inf"


def _ratio(numerator: int, denominator: int):
    if denominator == 0:
        return "inf" if numerator else None
    return float(numerator / denominator)


def _llr_distribution(values) -> dict:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    labels = ("minimum", "p10", "p25", "median", "p75", "p90", "p95", "p99",
              "maximum")
    if not absolute.size:
        return {
            "count": 0,
            "mean": None,
            "quantiles": {label: None for label in labels},
        }
    quantiles = np.quantile(absolute, probabilities)
    return {
        "count": int(absolute.size),
        "mean": float(absolute.mean()),
        "quantiles": {
            label: float(value) for label, value in zip(labels, quantiles)
        },
    }


def _plane_view(values, bits_per_symbol=8):
    array = np.asarray(values)
    if array.shape[-1] % bits_per_symbol:
        raise ValueError("last dimension must be divisible by bits_per_symbol")
    return array.reshape(-1, bits_per_symbol)


def mask_payload_bit_planes(values, planes, bits_per_symbol=8):
    masked = np.array(values, copy=True)
    shaped = masked.reshape(masked.shape[:-1] + (-1, bits_per_symbol))
    shaped[..., list(planes)] = 0.0
    return masked


def _safe_cosine(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def summarize_input_output_relation(
    input_llr,
    posterior_llr,
    *,
    bits_per_symbol=8,
) -> dict:
    cavity = _plane_view(input_llr, bits_per_symbol).astype(np.float64)
    posterior = _plane_view(posterior_llr, bits_per_symbol).astype(np.float64)
    if cavity.shape != posterior.shape:
        raise ValueError("input and posterior arrays must have equal shape")
    extrinsic = posterior - cavity
    rows = []
    for plane in range(bits_per_symbol):
        source = cavity[:, plane]
        output = posterior[:, plane]
        site = extrinsic[:, plane]
        source_energy = float(np.dot(source, source))
        posterior_gain = (
            None if source_energy == 0.0
            else float(np.dot(source, output) / source_energy)
        )
        site_gain = (
            None if source_energy == 0.0
            else float(np.dot(source, site) / source_energy)
        )
        source_rms = float(np.sqrt(np.mean(source ** 2)))
        rows.append({
            "plane": plane,
            "posterior_input_gain": posterior_gain,
            "posterior_input_cosine": _safe_cosine(source, output),
            "posterior_hard_agreement_with_input": float(
                np.mean((output > 0.0) == (source > 0.0))
            ),
            "src_ext_input_gain": site_gain,
            "src_ext_input_cosine": _safe_cosine(source, site),
            "input_rms": source_rms,
            "posterior_rms": float(np.sqrt(np.mean(output ** 2))),
            "src_ext_rms": float(np.sqrt(np.mean(site ** 2))),
            "src_ext_rms_over_input": (
                None if source_rms == 0.0
                else float(np.sqrt(np.mean(site ** 2)) / source_rms)
            ),
        })
    return {"planes_msb_to_lsb": rows}


def summarize_echo_probe(
    truth,
    cavity_llr,
    posterior_llr,
    single_plane_masked_posteriors,
    lower_two_masked_posterior,
    *,
    bits_per_symbol=8,
) -> dict:
    truth_planes = _plane_view(truth, bits_per_symbol).astype(np.uint8)
    cavity = _plane_view(cavity_llr, bits_per_symbol).astype(np.float64)
    posterior = _plane_view(posterior_llr, bits_per_symbol).astype(np.float64)
    if truth_planes.shape != cavity.shape or cavity.shape != posterior.shape:
        raise ValueError("truth, cavity, and posterior arrays must have equal shape")
    if len(single_plane_masked_posteriors) != bits_per_symbol:
        raise ValueError("one masked posterior is required for each bit-plane")

    input_accuracy = np.mean(
        (cavity > 0.0) == truth_planes, axis=0)
    output_accuracy = np.mean(
        (posterior > 0.0) == truth_planes, axis=0)
    lower_masked = _plane_view(
        lower_two_masked_posterior, bits_per_symbol).astype(np.float64)
    lower_masked_accuracy = np.mean(
        (lower_masked > 0.0) == truth_planes, axis=0)

    causal_rows = []
    for plane, masked_values in enumerate(single_plane_masked_posteriors):
        masked = _plane_view(
            masked_values, bits_per_symbol).astype(np.float64)
        source = cavity[:, plane]
        response = posterior[:, plane] - masked[:, plane]
        residual_site_response = response - source
        source_energy = float(np.dot(source, source))
        source_rms = float(np.sqrt(np.mean(source ** 2)))
        posterior_rms = float(np.sqrt(np.mean(posterior[:, plane] ** 2)))
        response_rms = float(np.sqrt(np.mean(response ** 2)))
        residual_rms = float(np.sqrt(np.mean(residual_site_response ** 2)))
        masked_accuracy = float(np.mean(
            (masked[:, plane] > 0.0) == truth_planes[:, plane]))
        causal_rows.append({
            "plane": plane,
            "masked_output_accuracy": masked_accuracy,
            "accuracy_drop_when_plane_removed": float(
                output_accuracy[plane] - masked_accuracy),
            "posterior_response_gain_from_removed_input": (
                None if source_energy == 0.0
                else float(np.dot(source, response) / source_energy)
            ),
            "posterior_response_input_cosine": _safe_cosine(source, response),
            "posterior_response_rms_over_input": (
                None if source_rms == 0.0 else float(response_rms / source_rms)
            ),
            "posterior_response_rms_over_output": (
                None if posterior_rms == 0.0
                else float(response_rms / posterior_rms)
            ),
            "src_ext_response_gain_after_input_subtraction": (
                None if source_energy == 0.0
                else float(np.dot(source, residual_site_response) / source_energy)
            ),
            "src_ext_response_rms_over_input": (
                None if source_rms == 0.0 else float(residual_rms / source_rms)
            ),
        })

    return {
        "input_cavity_accuracy_msb_to_lsb": [
            float(value) for value in input_accuracy
        ],
        "output_posterior_accuracy_msb_to_lsb": [
            float(value) for value in output_accuracy
        ],
        "output_minus_input_accuracy_msb_to_lsb": [
            float(value) for value in output_accuracy - input_accuracy
        ],
        "lower_two_joint_mask": {
            "output_accuracy_msb_to_lsb": [
                float(value) for value in lower_masked_accuracy
            ],
            "target_planes_original_accuracy": [
                float(output_accuracy[-2]), float(output_accuracy[-1])
            ],
            "target_planes_masked_accuracy": [
                float(lower_masked_accuracy[-2]),
                float(lower_masked_accuracy[-1]),
            ],
            "target_planes_accuracy_drop": [
                float(output_accuracy[-2] - lower_masked_accuracy[-2]),
                float(output_accuracy[-1] - lower_masked_accuracy[-1]),
            ],
        },
        "causal_single_plane_ablation_msb_to_lsb": causal_rows,
        "unmasked_input_output_relation": summarize_input_output_relation(
            cavity_llr, posterior_llr, bits_per_symbol=bits_per_symbol),
    }


def summarize_stage_transitions(truth, stages) -> dict:
    reference = np.asarray(truth, dtype=np.uint8)
    names = list(stages)
    decisions = {
        name: (np.asarray(stages[name]) > 0.0).astype(np.uint8)
        for name in names
    }
    for name, decision in decisions.items():
        if decision.shape != reference.shape:
            raise ValueError(f"stage {name} shape does not match truth")
    result = {
        "stage_wrong_bits": {
            name: int(np.sum(decision != reference))
            for name, decision in decisions.items()
        },
        "stage_wrong_blocks": {
            name: int(np.sum(np.any(decision != reference, axis=-1)))
            for name, decision in decisions.items()
        },
        "adjacent_transitions": {},
    }
    for previous, current in zip(names, names[1:]):
        previous_wrong = decisions[previous] != reference
        current_wrong = decisions[current] != reference
        result["adjacent_transitions"][f"{previous}_to_{current}"] = {
            "corrections": int(np.sum(previous_wrong & ~current_wrong)),
            "new_errors": int(np.sum(~previous_wrong & current_wrong)),
            "net_wrong_bit_change": int(
                np.sum(current_wrong) - np.sum(previous_wrong)),
        }
    return result


def summarize_paired_block_accounting(
    truth,
    score_off_bits,
    score_off_crc_valid,
    score_on_bits,
    score_on_crc_valid,
) -> dict:
    reference = np.asarray(truth, dtype=np.uint8)
    off = np.asarray(score_off_bits, dtype=np.uint8)
    on = np.asarray(score_on_bits, dtype=np.uint8)
    off_crc = np.asarray(score_off_crc_valid, dtype=bool).reshape(-1)
    on_crc = np.asarray(score_on_crc_valid, dtype=bool).reshape(-1)
    if reference.shape != off.shape or reference.shape != on.shape:
        raise ValueError("truth and both decoded payload arrays must match")
    if reference.shape[0] != off_crc.size or off_crc.size != on_crc.size:
        raise ValueError("CRC-valid arrays must match the block count")

    off_error_counts = np.sum(off != reference, axis=-1)
    on_error_counts = np.sum(on != reference, axis=-1)
    off_failed = off_error_counts > 0
    on_failed = on_error_counts > 0
    broken = ~off_failed & on_failed
    rescued = off_failed & ~on_failed
    both_failed = off_failed & on_failed
    both_succeeded = ~off_failed & ~on_failed
    crc_broken = off_crc & ~on_crc
    crc_rescued = ~off_crc & on_crc
    return {
        "blocks": int(reference.shape[0]),
        "true_payload": {
            "score_off_failed": int(off_failed.sum()),
            "score_on_failed": int(on_failed.sum()),
            "successful_blocks_broken_by_score": int(broken.sum()),
            "failed_blocks_rescued_by_score": int(rescued.sum()),
            "both_failed": int(both_failed.sum()),
            "both_succeeded": int(both_succeeded.sum()),
            "net_bler_error_count_change": int(broken.sum() - rescued.sum()),
            "broken_indices": [int(value) for value in np.flatnonzero(broken)],
            "rescued_indices": [int(value) for value in np.flatnonzero(rescued)],
            "score_off_error_bits": int(off_error_counts.sum()),
            "score_on_error_bits": int(on_error_counts.sum()),
        },
        "crc": {
            "score_off_failed": int((~off_crc).sum()),
            "score_on_failed": int((~on_crc).sum()),
            "successful_blocks_broken_by_score": int(crc_broken.sum()),
            "failed_blocks_rescued_by_score": int(crc_rescued.sum()),
            "net_bler_error_count_change": int(
                crc_broken.sum() - crc_rescued.sum()),
            "broken_indices": [
                int(value) for value in np.flatnonzero(crc_broken)
            ],
            "rescued_indices": [
                int(value) for value in np.flatnonzero(crc_rescued)
            ],
        },
        "true_crc_failure_masks_identical_score_off": bool(
            np.array_equal(off_failed, ~off_crc)),
        "true_crc_failure_masks_identical_score_on": bool(
            np.array_equal(on_failed, ~on_crc)),
    }


def summarize_diagnostics(
    payload_true,
    cavity_llr,
    source_posterior_llr,
    crc_valid,
    *,
    source_shape=(28, 28),
    bits_per_symbol=8,
    injection_alpha=0.1,
    injection_clip=30.0,
) -> dict:
    truth = np.asarray(payload_true, dtype=np.uint8)
    cavity = np.asarray(cavity_llr, dtype=np.float64)
    source = np.asarray(source_posterior_llr, dtype=np.float64)
    valid = np.asarray(crc_valid, dtype=bool).reshape(-1)
    if truth.shape != cavity.shape or truth.shape != source.shape:
        raise ValueError("truth, cavity, and source arrays must have equal shape")
    if truth.shape[0] != valid.size:
        raise ValueError("CRC-valid vector length does not match the batch")

    cavity_hat = (cavity > 0.0).astype(np.uint8)
    source_hat = (source > 0.0).astype(np.uint8)
    cavity_wrong = cavity_hat != truth
    cavity_correct = ~cavity_wrong
    source_correct = source_hat == truth
    disagreement = source_hat != cavity_hat

    wrong_count = int(cavity_wrong.sum())
    source_correct_on_wrong = int((source_correct & cavity_wrong).sum())
    useful = int((disagreement & cavity_wrong).sum())
    harmful = int((disagreement & cavity_correct).sum())
    disagreement_count = int(disagreement.sum())

    useful_mask = disagreement & cavity_wrong
    harmful_mask = disagreement & cavity_correct
    source_extrinsic = source - cavity
    injected = cavity + injection_alpha * np.clip(
        source_extrinsic, -injection_clip, injection_clip)
    injected_hat = (injected > 0.0).astype(np.uint8)
    sign_flipped = injected_hat != cavity_hat
    correction_success = sign_flipped & cavity_wrong & (injected_hat == truth)
    contamination_success = (
        sign_flipped & cavity_correct & (injected_hat != truth))
    correction_success_count = int(correction_success.sum())
    contamination_success_count = int(contamination_success.sum())

    source_plane_accuracy = source_correct.reshape(
        -1, bits_per_symbol).mean(axis=0)
    cavity_plane_accuracy = cavity_correct.reshape(
        -1, bits_per_symbol).mean(axis=0)
    cavity_wrong_by_plane = cavity_wrong.reshape(-1, bits_per_symbol)
    source_correct_by_plane = source_correct.reshape(-1, bits_per_symbol)
    conditional_plane_correct = (
        source_correct_by_plane & cavity_wrong_by_plane).sum(axis=0)
    conditional_plane_total = cavity_wrong_by_plane.sum(axis=0)
    conditional_plane_accuracy = [
        (
            None if int(total) == 0
            else float(correct / total)
        )
        for correct, total in zip(
            conditional_plane_correct, conditional_plane_total)
    ]

    reference_images = payload_bits_to_uint8_images(
        truth, source_shape, bits_per_symbol)
    cavity_images = payload_bits_to_uint8_images(
        cavity_hat, source_shape, bits_per_symbol)
    failure = ~valid
    failure_indices = np.flatnonzero(failure)
    failure_error_counts = cavity_wrong.sum(axis=-1)[failure].astype(np.int64)
    if failure_indices.size:
        squared_error = (
            reference_images[failure].astype(np.float64)
            - cavity_images[failure].astype(np.float64)
        ) ** 2
        failure_mse = squared_error.reshape(failure_indices.size, -1).mean(axis=-1)
        failure_psnr = np.where(
            failure_mse == 0.0,
            np.inf,
            10.0 * np.log10((255.0 ** 2) / failure_mse),
        )
        failure_ssim = np.asarray([
            _uniform_window_ssim(reference, estimate)
            for reference, estimate in zip(
                reference_images[failure], cavity_images[failure])
        ])
        failure_plane_errors = cavity_wrong[failure].reshape(
            -1, bits_per_symbol).sum(axis=0).astype(np.int64)
    else:
        failure_psnr = np.asarray([], dtype=np.float64)
        failure_ssim = np.asarray([], dtype=np.float64)
        failure_plane_errors = np.zeros(bits_per_symbol, dtype=np.int64)

    failure_plane_total = int(failure_plane_errors.sum())
    return {
        "blocks": int(truth.shape[0]),
        "payload_bits_per_block": int(truth.shape[-1]),
        "cavity_bit_accuracy": float(cavity_correct.mean()),
        "source_posterior_bit_accuracy": float(source_correct.mean()),
        "cavity_wrong_bits": wrong_count,
        "conditional": {
            "source_accuracy_where_cavity_wrong": (
                None if wrong_count == 0
                else float(source_correct_on_wrong / wrong_count)
            ),
            "source_correct_where_cavity_wrong": source_correct_on_wrong,
            "source_cavity_disagreements": disagreement_count,
            "useful_corrections": useful,
            "harmful_contaminations": harmful,
            "useful_fraction_of_disagreements": (
                None if disagreement_count == 0
                else float(useful / disagreement_count)
            ),
            "harmful_fraction_of_disagreements": (
                None if disagreement_count == 0
                else float(harmful / disagreement_count)
            ),
            "useful_to_harmful_ratio": _ratio(useful, harmful),
            "cavity_abs_llr_at_useful_correction_opportunities": (
                _llr_distribution(cavity[useful_mask])
            ),
            "cavity_abs_llr_at_harmful_contamination_attempts": (
                _llr_distribution(cavity[harmful_mask])
            ),
            "local_alpha_clip_injection": {
                "alpha": float(injection_alpha),
                "extrinsic_clip": float(injection_clip),
                "sign_flips": int(sign_flipped.sum()),
                "successful_corrections": correction_success_count,
                "successful_contaminations": contamination_success_count,
                "correction_success_fraction_of_opportunities": (
                    None if useful == 0
                    else float(correction_success_count / useful)
                ),
                "contamination_success_fraction_of_attempts": (
                    None if harmful == 0
                    else float(contamination_success_count / harmful)
                ),
                "effective_correction_to_contamination_ratio": _ratio(
                    correction_success_count, contamination_success_count),
            },
        },
        "bit_planes_msb_to_lsb": {
            "cavity_accuracy": [float(value) for value in cavity_plane_accuracy],
            "source_posterior_accuracy": [
                float(value) for value in source_plane_accuracy
            ],
            "source_posterior_accuracy_where_cavity_wrong": (
                conditional_plane_accuracy
            ),
            "cavity_wrong_counts": [
                int(value) for value in conditional_plane_total
            ],
        },
        "crc_failure_blocks": int(failure_indices.size),
        "crc_failure_indices": [int(value) for value in failure_indices],
        "failure_block_bit_errors": {
            "values": [int(value) for value in failure_error_counts],
            "mean": (
                None if not failure_error_counts.size
                else float(failure_error_counts.mean())
            ),
            "median": (
                None if not failure_error_counts.size
                else float(np.median(failure_error_counts))
            ),
            "maximum": (
                None if not failure_error_counts.size
                else int(failure_error_counts.max())
            ),
        },
        "failure_image_quality": {
            "psnr_db_values": [_finite_or_string(value) for value in failure_psnr],
            "psnr_db_mean": (
                None if not failure_psnr.size
                else _finite_or_string(float(failure_psnr.mean()))
            ),
            "psnr_db_median": (
                None if not failure_psnr.size
                else _finite_or_string(float(np.median(failure_psnr)))
            ),
            "ssim_values": [float(value) for value in failure_ssim],
            "ssim_mean": (
                None if not failure_ssim.size else float(failure_ssim.mean())
            ),
            "ssim_median": (
                None if not failure_ssim.size else float(np.median(failure_ssim))
            ),
        },
        "failure_bit_plane_errors_msb_to_lsb": {
            "counts": [int(value) for value in failure_plane_errors],
            "fractions": (
                [0.0] * bits_per_symbol
                if failure_plane_total == 0
                else [
                    float(value / failure_plane_total)
                    for value in failure_plane_errors
                ]
            ),
        },
        "_failure_indices_array": failure_indices,
        "_reference_images_array": reference_images,
        "_cavity_images_array": cavity_images,
    }


def save_failure_montage(summary: dict, path: Path, title: str) -> int:
    indices = summary["_failure_indices_array"][:3]
    if not indices.size:
        return 0
    reference = summary["_reference_images_array"]
    cavity = summary["_cavity_images_array"]
    rows = int(indices.size)
    figure, axes = plt.subplots(
        rows, 3, figsize=(8.0, 2.6 * rows), squeeze=False, constrained_layout=True)
    error_values = summary["failure_block_bit_errors"]["values"]
    psnr_values = summary["failure_image_quality"]["psnr_db_values"]
    ssim_values = summary["failure_image_quality"]["ssim_values"]
    for row, index in enumerate(indices):
        difference = np.abs(
            reference[index].astype(np.int16) - cavity[index].astype(np.int16))
        panels = (
            (reference[index], "original", "gray", 0, 255),
            (cavity[index], "pre-source cavity", "gray", 0, 255),
            (difference, "|pixel error|", "magma", 0, 255),
        )
        for column, (image, label, cmap, lower, upper) in enumerate(panels):
            axes[row, column].imshow(
                image, cmap=cmap, vmin=lower, vmax=upper, interpolation="nearest")
            axes[row, column].set_title(label)
            axes[row, column].axis("off")
        psnr = psnr_values[row]
        psnr_label = psnr if isinstance(psnr, str) else f"{psnr:.2f}"
        axes[row, 1].set_xlabel(
            f"errors={error_values[row]}, PSNR={psnr_label} dB, "
            f"SSIM={ssim_values[row]:.5f}"
        )
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return rows


def _json_ready(summary: dict) -> dict:
    return {
        key: value for key, value in summary.items()
        if not key.startswith("_")
    }


def _mask_rsc_source_planes(source_message, planes, bits_per_symbol=8):
    block_length = bits_per_symbol + 1
    masked = np.array(source_message, copy=True)
    shaped = masked.reshape(masked.shape[:-1] + (-1, block_length))
    shaped[..., list(planes)] = 0.0
    return masked


def run_rsc(args, bank) -> list[dict]:
    from coding.bcjr import bcjr_decode
    from decoders import NoLDPCConfig, RSCSourceIterativeDecoder, SourceSPCSISO
    from experiments.no_ldpc_smoke import load_score_provider

    config = NoLDPCConfig(
        outer_iterations=2,
        alpha_schedule=(0.1, 0.1),
        llr_clip=30.0,
        sigma=0.3,
        sigma_post=3.0,
        interleaver_seed=20260722,
        bcjr_mode="logmap",
    )
    decoder = RSCSourceIterativeDecoder(config)
    source_siso = SourceSPCSISO(
        8, load_score_provider(config, args.checkpoint, args.device))
    rows = []
    for esn0_db in args.esn0_db:
        payloads = []
        cavities = []
        source_posteriors = []
        lower_two_masked_posteriors = []
        single_plane_masked_posteriors = [[] for _ in range(8)]
        crc_valids = []
        stage_payloads = {
            "first_bcjr_pre_source": [],
            "first_source_terminal": [],
            "second_bcjr_pre_source": [],
            "second_source_terminal": [],
        }
        for _, payload, noise in paired_batches(
            bank,
            blocks=args.blocks,
            batch_size=args.batch_size,
            esn0_db=esn0_db,
            seed=args.seed,
        ):
            frame = decoder.encode(payload)
            channel_llr = awgn_llr(frame.transmitted_bits, noise, esn0_db)
            systematic_llr, parity_llr = decoder.rate_matcher.depuncture_llr(
                channel_llr)
            first_bcjr = bcjr_decode(
                systematic_llr,
                parity_llr,
                decoder.trellis,
                a_priori_llr=np.zeros_like(systematic_llr),
                mode="logmap",
                start_state=0,
                end_state=0,
            )
            source_order = decoder.interleaver.deinterleave(
                first_bcjr.extrinsic_llr[..., : config.rsc_information_length])
            source_message = source_order[..., : config.source_coded_length]
            crc_llr = source_order[..., config.source_coded_length :]
            source_result = source_siso(source_message)
            payload_cavity = decoder._payload_llr_from_source(source_message)
            payload_source = decoder._payload_llr_from_source(
                source_result.posterior_llr)
            for plane in range(8):
                masked_source_message = _mask_rsc_source_planes(
                    source_message, (plane,))
                masked_source_result = source_siso(masked_source_message)
                single_plane_masked_posteriors[plane].append(
                    decoder._payload_llr_from_source(
                        masked_source_result.posterior_llr))
            lower_masked_source_message = _mask_rsc_source_planes(
                source_message, (6, 7))
            lower_masked_source_result = source_siso(lower_masked_source_message)
            lower_two_masked_posteriors.append(
                decoder._payload_llr_from_source(
                    lower_masked_source_result.posterior_llr))

            first_scaled = config.alpha_at(0) * np.clip(
                source_result.extrinsic_llr,
                -config.llr_clip,
                config.llr_clip,
            )
            first_terminal = decoder._payload_llr_from_source(
                source_message + first_scaled)
            first_feedback = np.concatenate(
                [first_scaled, np.zeros_like(crc_llr)], axis=-1)
            second_prior = decoder.interleaver.interleave(first_feedback)
            second_bcjr = bcjr_decode(
                systematic_llr,
                parity_llr,
                decoder.trellis,
                a_priori_llr=np.concatenate([
                    second_prior,
                    np.zeros(
                        second_prior.shape[:-1] + (config.memory,),
                        dtype=np.float64,
                    ),
                ], axis=-1),
                mode="logmap",
                start_state=0,
                end_state=0,
            )
            second_source_order = decoder.interleaver.deinterleave(
                second_bcjr.extrinsic_llr[
                    ..., : config.rsc_information_length])
            second_source_message = second_source_order[
                ..., : config.source_coded_length]
            second_source_result = source_siso(second_source_message)
            second_scaled = config.alpha_at(1) * np.clip(
                second_source_result.extrinsic_llr,
                -config.llr_clip,
                config.llr_clip,
            )
            second_cavity = decoder._payload_llr_from_source(
                second_source_message)
            second_terminal = decoder._payload_llr_from_source(
                second_source_message + second_scaled)
            payload_hat = (payload_cavity > 0.0).astype(np.uint8)
            crc_hat = (crc_llr > 0.0).astype(np.uint8)
            crc_valid = np.asarray(
                config.crc.check_parts(payload_hat, crc_hat), dtype=bool)
            payloads.append(payload)
            cavities.append(payload_cavity)
            source_posteriors.append(payload_source)
            stage_payloads["first_bcjr_pre_source"].append(payload_cavity)
            stage_payloads["first_source_terminal"].append(first_terminal)
            stage_payloads["second_bcjr_pre_source"].append(second_cavity)
            stage_payloads["second_source_terminal"].append(second_terminal)
            crc_valids.append(crc_valid)
        summary = summarize_diagnostics(
            np.concatenate(payloads),
            np.concatenate(cavities),
            np.concatenate(source_posteriors),
            np.concatenate(crc_valids),
        )
        png = args.image_dir / f"rsc_failure_cavity_esn0_{esn0_db:g}dB.png"
        saved = save_failure_montage(
            summary, png, f"RSC pre-source failures, Es/N0={esn0_db:g} dB")
        row = _json_ready(summary)
        row["echo_probe"] = summarize_echo_probe(
            np.concatenate(payloads),
            np.concatenate(cavities),
            np.concatenate(source_posteriors),
            [
                np.concatenate(values)
                for values in single_plane_masked_posteriors
            ],
            np.concatenate(lower_two_masked_posteriors),
        )
        row["two_outer_stage_transitions"] = summarize_stage_transitions(
            np.concatenate(payloads),
            {
                name: np.concatenate(values)
                for name, values in stage_payloads.items()
            },
        )
        row.update({
            "arm": "rsc_first_bcjr_pre_source",
            "esn0_db": float(esn0_db),
            "representative_png": str(png) if saved else None,
            "representative_failure_blocks": saved,
        })
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def run_rsc_paired_score_accounting(args, bank) -> dict:
    from decoders import (
        IndependentBitCategoricalProvider,
        NoLDPCConfig,
        RSCSourceIterativeDecoder,
        SourceSPCSISO,
    )
    from experiments.no_ldpc_smoke import load_score_provider

    if args.esn0_db != [2.6]:
        raise ValueError("paired score accounting is fixed at Es/N0=2.6 dB")
    if args.blocks != 256:
        raise ValueError("paired score accounting requires exactly 256 blocks")
    config = NoLDPCConfig(
        outer_iterations=2,
        alpha_schedule=(0.1, 0.1),
        llr_clip=30.0,
        sigma=0.3,
        sigma_post=3.0,
        interleaver_seed=20260722,
        bcjr_mode="logmap",
    )
    decoder = RSCSourceIterativeDecoder(config)
    score_off_siso = SourceSPCSISO(
        config.source_bits_per_symbol,
        IndependentBitCategoricalProvider(config.source_bits_per_symbol),
    )
    score_provider = load_score_provider(
        config, args.checkpoint, args.device)
    if args.unbounded_score_provider:
        del score_provider.posterior_llr_bounds
    score_on_siso = SourceSPCSISO(
        config.source_bits_per_symbol, score_provider)
    payloads = []
    score_off_payloads = []
    score_on_payloads = []
    score_off_crc_valids = []
    score_on_crc_valids = []
    for _, payload, noise in paired_batches(
        bank,
        blocks=args.blocks,
        batch_size=args.batch_size,
        esn0_db=2.6,
        seed=args.seed,
    ):
        frame = decoder.encode(payload)
        channel_llr = awgn_llr(frame.transmitted_bits, noise, 2.6)
        score_off = decoder.decode(channel_llr, score_off_siso)
        score_on = decoder.decode(channel_llr, score_on_siso)
        payloads.append(payload)
        score_off_payloads.append(score_off.payload_bits)
        score_on_payloads.append(score_on.payload_bits)
        score_off_crc_valids.append(
            np.asarray(score_off.crc_valid, dtype=bool).reshape(-1))
        score_on_crc_valids.append(
            np.asarray(score_on.crc_valid, dtype=bool).reshape(-1))

    accounting = summarize_paired_block_accounting(
        np.concatenate(payloads),
        np.concatenate(score_off_payloads),
        np.concatenate(score_off_crc_valids),
        np.concatenate(score_on_payloads),
        np.concatenate(score_on_crc_valids),
    )
    accounting["conditions"] = {
        "esn0_db": 2.6,
        "blocks": 256,
        "seed": int(args.seed),
        "same_payloads_and_noise": True,
        "outer_iterations": 2,
        "alpha_schedule": [0.1, 0.1],
        "llr_clip": 30.0,
        "bcjr_mode": "logmap",
        "use_spc": True,
        "score_off_source": "IndependentBitCategoricalProvider + SourceSPCSISO",
        "score_on_source": "ScorePriorCategoricalProvider + SourceSPCSISO",
        "score_posterior_bound": (
            "pre-audit unbounded"
            if args.unbounded_score_provider
            else "LDPC-compatible marginal bound"
        ),
        "block_error_rule": "any payload bit error",
        "crc_error_rule": "CRC-16 check failure",
    }
    return accounting


def run_ldpc(args, bank) -> list[dict]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    sys.dont_write_bytecode = True

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
    from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
    from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder

    if not args.practical_worktree.is_dir():
        raise FileNotFoundError(args.practical_worktree)
    sys.path.insert(0, str(args.practical_worktree.resolve()))
    import torch
    from decoder import LDPC5GDecoder_soft

    crc = CRCEncoder("CRC24A")
    crc_decoder = CRCDecoder(crc)
    encoder = LDPC5GEncoder(
        PAYLOAD_BITS + crc.crc_length,
        TRANSMITTED_BITS,
        num_bits_per_symbol=1,
    )
    common = dict(
        cn_update="boxplus-phi",
        vn_update="sum",
        cn_schedule="flooding",
        hard_out=False,
        return_infobits=True,
        llr_max=30.0,
    )
    bp100 = LDPC5GDecoder(encoder, num_iter=100, **common)
    torch_device = args.device if torch.cuda.is_available() else "cpu"
    legacy = LDPC5GDecoder_soft(
        encoder,
        bp_schedule=[5] * 20,
        alpha=0.1,
        beta=0.1,
        ep_mode=False,
        source_input_mode="bp_post",
        adaptive_sigma=False,
        k_payload=PAYLOAD_BITS,
        denoiser_kwargs=dict(device=torch_device),
        num_iter=100,
        **common,
    )
    legacy.denoiser.load_weights_pt(str(args.checkpoint))
    legacy.denoiser.sigma = 0.3

    rows = []
    for esn0_db in args.esn0_db:
        payloads = []
        cavities = []
        source_posteriors = []
        lower_two_masked_posteriors = []
        single_plane_masked_posteriors = [[] for _ in range(8)]
        crc_valids = []
        for _, payload, noise in paired_batches(
            bank,
            blocks=args.blocks,
            batch_size=args.batch_size,
            esn0_db=esn0_db,
            seed=args.seed,
        ):
            payload_tf = tf.constant(payload, dtype=encoder.rdtype)
            information = crc(payload_tf)
            codeword = encoder(information).numpy().astype(np.uint8)
            channel_llr = tf.constant(
                awgn_llr(codeword, noise, esn0_db), dtype=encoder.rdtype)
            bp_logits = bp100(channel_llr)
            _, crc_valid_tf = crc_decoder(bp_logits)
            payload_cavity = bp_logits[:, :PAYLOAD_BITS]
            source_extrinsic = legacy.denoiser(
                payload_cavity,
                sigma=tf.fill([payload.shape[0]], tf.constant(0.3, tf.float32)),
            )
            payloads.append(payload)
            cavity_np = payload_cavity.numpy()
            cavities.append(cavity_np)
            source_posteriors.append(cavity_np + source_extrinsic.numpy())
            sigma_batch = tf.fill(
                [payload.shape[0]], tf.constant(0.3, tf.float32))
            for plane in range(8):
                masked_cavity_np = mask_payload_bit_planes(
                    cavity_np, (plane,))
                masked_cavity = tf.constant(
                    masked_cavity_np, dtype=encoder.rdtype)
                masked_extrinsic = legacy.denoiser(
                    masked_cavity, sigma=sigma_batch)
                single_plane_masked_posteriors[plane].append(
                    masked_cavity_np + masked_extrinsic.numpy())
            lower_masked_cavity_np = mask_payload_bit_planes(
                cavity_np, (6, 7))
            lower_masked_cavity = tf.constant(
                lower_masked_cavity_np, dtype=encoder.rdtype)
            lower_masked_extrinsic = legacy.denoiser(
                lower_masked_cavity, sigma=sigma_batch)
            lower_two_masked_posteriors.append(
                lower_masked_cavity_np + lower_masked_extrinsic.numpy())
            crc_valids.append(
                np.asarray(crc_valid_tf.numpy(), dtype=bool).reshape(-1))
        summary = summarize_diagnostics(
            np.concatenate(payloads),
            np.concatenate(cavities),
            np.concatenate(source_posteriors),
            np.concatenate(crc_valids),
        )
        png = args.image_dir / f"ldpc_failure_cavity_esn0_{esn0_db:g}dB.png"
        saved = save_failure_montage(
            summary, png, f"LDPC BP-100 pre-source failures, Es/N0={esn0_db:g} dB")
        row = _json_ready(summary)
        row["echo_probe"] = summarize_echo_probe(
            np.concatenate(payloads),
            np.concatenate(cavities),
            np.concatenate(source_posteriors),
            [
                np.concatenate(values)
                for values in single_plane_masked_posteriors
            ],
            np.concatenate(lower_two_masked_posteriors),
        )
        row.update({
            "arm": "practical_sigma_bp100_pre_source",
            "esn0_db": float(esn0_db),
            "representative_png": str(png) if saved else None,
            "representative_failure_blocks": saved,
        })
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def run_legacy_minus_echo_probe(args, bank) -> dict:
    """Record existing purity-path denoiser inputs/sites without changing them."""
    import tensorflow as tf
    import torch
    from sionna.phy.fec.crc import CRCEncoder
    from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder

    if len(args.esn0_db) != 1:
        raise ValueError(
            "--legacy-minus-echo requires exactly one Es/N0 point")
    if str(args.practical_worktree.resolve()) not in sys.path:
        sys.path.insert(0, str(args.practical_worktree.resolve()))
    from decoder import LDPC5GDecoder_soft

    crc = CRCEncoder("CRC24A")
    encoder = LDPC5GEncoder(
        PAYLOAD_BITS + crc.crc_length,
        TRANSMITTED_BITS,
        num_bits_per_symbol=1,
    )
    batches = []
    for _, payload, noise in paired_batches(
        bank,
        blocks=args.blocks,
        batch_size=args.batch_size,
        esn0_db=args.esn0_db[0],
        seed=args.seed,
    ):
        information = crc(tf.constant(payload, dtype=encoder.rdtype))
        codeword = encoder(information).numpy().astype(np.uint8)
        channel_llr = awgn_llr(
            codeword, noise, args.esn0_db[0])
        batches.append(channel_llr)

    class RecordingDenoiser(tf.keras.layers.Layer):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.inputs = []
            self.extrinsics = []

        def call(self, llr_tf, sigma=None):
            output = self.inner(llr_tf, sigma=sigma)
            self.inputs.append(llr_tf.numpy().copy())
            self.extrinsics.append(output.numpy().copy())
            return output

        @property
        def sigma(self):
            return self.inner.sigma

        @sigma.setter
        def sigma(self, value):
            self.inner.sigma = value

        def clear(self):
            self.inputs.clear()
            self.extrinsics.clear()

    common = dict(
        cn_update="boxplus-phi",
        vn_update="sum",
        cn_schedule="flooding",
        hard_out=False,
        return_infobits=True,
        llr_max=30.0,
    )

    def probe_mode(mode):
        decoder = LDPC5GDecoder_soft(
            encoder,
            bp_schedule=[2] * 15,
            alpha=0.1,
            beta=0.0,
            ep_mode=False,
            source_input_mode=mode,
            adaptive_sigma=False,
            k_payload=PAYLOAD_BITS,
            denoiser_kwargs=dict(
                device=args.device if torch.cuda.is_available() else "cpu"),
            num_iter=30,
            **common,
        )
        decoder.denoiser.load_weights_pt(str(args.checkpoint))
        decoder.denoiser.sigma = 0.3
        recorder = RecordingDenoiser(decoder.denoiser)
        decoder._denoiser = recorder
        chunk_inputs = None
        chunk_posteriors = None
        for channel_llr in batches:
            _ = decoder(tf.constant(channel_llr, dtype=encoder.rdtype))
            if chunk_inputs is None:
                chunk_inputs = [[] for _ in recorder.inputs]
                chunk_posteriors = [[] for _ in recorder.inputs]
            if len(recorder.inputs) != len(chunk_inputs):
                raise AssertionError(
                    f"{mode} source-call count changed between batches")
            for chunk, (source_input, source_extrinsic) in enumerate(
                zip(recorder.inputs, recorder.extrinsics)
            ):
                chunk_inputs[chunk].append(source_input)
                chunk_posteriors[chunk].append(
                    source_input + source_extrinsic)
            recorder.clear()
        if chunk_inputs is None or chunk_posteriors is None:
            raise AssertionError(f"{mode} did not run any source calls")
        chunk_inputs = [np.concatenate(values) for values in chunk_inputs]
        chunk_posteriors = [
            np.concatenate(values) for values in chunk_posteriors]
        relations = [
            summarize_input_output_relation(source_input, posterior)
            for source_input, posterior in zip(
                chunk_inputs, chunk_posteriors)
        ]

        def compact(relation):
            planes = relation["planes_msb_to_lsb"]
            return {
                "posterior_input_gain_msb_to_lsb": [
                    row["posterior_input_gain"] for row in planes
                ],
                "posterior_hard_agreement_with_input_msb_to_lsb": [
                    row["posterior_hard_agreement_with_input"]
                    for row in planes
                ],
                "src_ext_input_gain_msb_to_lsb": [
                    row["src_ext_input_gain"] for row in planes
                ],
                "src_ext_rms_over_input_msb_to_lsb": [
                    row["src_ext_rms_over_input"] for row in planes
                ],
            }

        overall = summarize_input_output_relation(
            np.concatenate(chunk_inputs),
            np.concatenate(chunk_posteriors),
        )
        result = {
            "mode": mode,
            "source_calls": len(relations),
            "overall": overall,
            "first_chunk": relations[0],
            "last_chunk": relations[-1],
            "per_chunk": [
                {"chunk": chunk, **compact(relation)}
                for chunk, relation in enumerate(relations)
            ],
        }
        del decoder, recorder, chunk_inputs, chunk_posteriors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    return {
        "configuration": {
            "esn0_db": float(args.esn0_db[0]),
            "blocks": int(args.blocks),
            "seed": int(args.seed),
            "bp_schedule": [2] * 15,
            "alpha": 0.1,
            "beta": 0.0,
            "sigma": 0.3,
            "practical_worktree_read_only": True,
        },
        "bp_post": probe_mode("bp_post"),
        "minus_source_feedback": probe_mode("minus_source_feedback"),
    }


def main() -> None:
    args = parse_args()
    maximum_blocks = 256 if args.paired_score_accounting_only else 64
    if not 1 <= args.blocks <= maximum_blocks:
        raise ValueError(
            f"diagnostic blocks must lie in [1, {maximum_blocks}]")
    if args.paired_score_accounting_only and args.arm != "rsc":
        raise ValueError("paired score accounting is available only for RSC")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    bank = load_fashion_mnist_bits(args.dataset_root)
    if args.paired_score_accounting_only:
        output = {
            "experiment": "rsc_paired_score_block_accounting",
            "instrumentation_only": True,
            "arm": "rsc_score_spc_vs_rsc_spc",
            "accounting": run_rsc_paired_score_accounting(args, bank),
        }
        args.output.write_text(
            json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output["accounting"], sort_keys=True), flush=True)
        print(f"wrote {args.output}", flush=True)
        return
    rows = run_rsc(args, bank) if args.arm == "rsc" else run_ldpc(args, bank)
    output = {
        "experiment": "source_residual_diagnostic",
        "instrumentation_only": True,
        "arm": args.arm,
        "blocks": args.blocks,
        "seed": args.seed,
        "rows": rows,
    }
    if args.arm == "ldpc":
        output.update({
            "branch": "practical_sigma",
            "branch_worktree": str(args.practical_worktree),
            "branch_read_only": True,
            "bp_iterations": 100,
            "source_module": "legacy SoftDenoiser",
        })
        if args.legacy_minus_echo:
            output["legacy_minus_echo_probe"] = run_legacy_minus_echo_probe(
                args, bank)
    else:
        output.update({
            "branch": "no-LDPC",
            "bcjr_pass": 1,
            "source_injected": False,
            "source_module": "ScorePriorCategoricalProvider + SourceSPCSISO",
        })
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
