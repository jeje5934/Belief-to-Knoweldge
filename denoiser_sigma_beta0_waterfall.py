"""
Finalize the practical_sigma denoiser-sigma schedule with beta=0.

This is an experiment harness only.  The production decoder is imported
unchanged and all sigma variation is supplied through its existing
``sigma_scheduler`` interface.

Modes
-----
coarse
    Lightweight fixed-sigma beta=0 knee scan.
waterfall
    Paired fixed/geometric-schedule waterfall with Wilson intervals.
state
    Fit receiver-visible state regressions to the earlier conditional profile
    and compare syndrome/mean-|LLR| schedules against the best time schedule.
alpha
    Lightweight alpha check with the selected sigma strategy.
plot
    Render the final waterfall and the state/alpha knee comparisons.
merge
    Merge independently seeded low-BLER shards into a waterfall JSON.
"""

from __future__ import annotations

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
from sionna.phy.fec.crc import CRCDecoder
from syndrome_sigma_schedule import AnnealingSigmaScheduler

from denoiser_sigma_alignment_diag import (
    FIXED_SIGMA,
    SCHEDULE,
    build_decoder,
    load_fashion_mnist,
)
from denoiser_sigma_conditional_diag import (
    SOURCE_CALLS,
    make_channel_batch,
    make_system,
    wilson,
)


DEFAULT_CHECKPOINT = (
    "/home/LJH/onlyextrinsic_ada_sigma/checkpoints/denoiser.pt"
)
SIGMA_MIN = 0.01
SIGMA_MAX = 0.30


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def dump_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_beta0_decoder(ldpc, checkpoint, device, alpha=0.1):
    decoder = build_decoder(ldpc, checkpoint, device)
    decoder.alpha = float(alpha)
    decoder.beta = 0.0
    decoder.denoiser.sigma = FIXED_SIGMA
    return decoder


def summarize_mask(success, reference=None):
    success = np.asarray(success, dtype=bool)
    failures = int(np.sum(~success))
    row = {
        "failures": failures,
        "blocks": int(len(success)),
        "crc_bler": float(failures / len(success)),
        "wilson_95": wilson(failures, len(success)),
        "success_mask": success.tolist(),
    }
    if reference is not None:
        reference = np.asarray(reference, dtype=bool)
        broken = int(np.sum(reference & ~success))
        rescued = int(np.sum(~reference & success))
        row["vs_reference"] = {
            "reference_success_to_candidate_failure": broken,
            "reference_failure_to_candidate_success": rescued,
            "net_failure_change": broken - rescued,
            "both_failure": int(np.sum(~reference & ~success)),
            "both_success": int(np.sum(reference & success)),
        }
    return row


def geometric_scheduler(endpoint):
    return AnnealingSigmaScheduler(
        FIXED_SIGMA, float(endpoint), SOURCE_CALLS, mode="geom"
    )


def time_candidates():
    return {
        "fixed_0.3": None,
        "geom_0.3_to_0.02": geometric_scheduler(0.02),
        "geom_0.3_to_0.05": geometric_scheduler(0.05),
        "geom_0.3_to_0.08": geometric_scheduler(0.08),
    }


def run_candidates_at_snr(
    *,
    esn0_db,
    blocks,
    batch,
    seed,
    alpha,
    checkpoint,
    candidate_factory,
    reference_name,
):
    if blocks % batch:
        raise ValueError("blocks must be divisible by batch")

    crc_encoder, crc_decoder, ldpc, mapper, demapper, awgn = make_system()
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = build_beta0_decoder(ldpc, checkpoint, device, alpha=alpha)
    candidates = candidate_factory(decoder)
    masks = {name: [] for name in candidates}
    rounds = blocks // batch

    for round_idx in range(rounds):
        _, _, _, channel_llr = make_channel_batch(
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
        )
        for name, scheduler in candidates.items():
            decoder.alpha = float(alpha)
            decoder.beta = 0.0
            decoder.sigma_scheduler = scheduler
            decoder.denoiser.sigma = FIXED_SIGMA
            final_logits = decoder(channel_llr)
            _, crc_valid = crc_decoder(final_logits)
            masks[name].extend(
                np.asarray(crc_valid.numpy())
                .reshape(-1)
                .astype(bool)
                .tolist()
            )
        print(
            f"Es/N0={esn0_db:+.2f} round {round_idx + 1}/{rounds}",
            flush=True,
        )

    reference = np.asarray(masks[reference_name], dtype=bool)
    return {
        name: summarize_mask(
            values,
            None if name == reference_name else reference,
        )
        for name, values in masks.items()
    }


def run_coarse(args):
    snrs = parse_float_list(args.snrs)
    result = {
        "kind": "beta0_fixed_sigma_coarse_scan",
        "configuration": {
            "branch": "practical_sigma",
            "decoder": "legacy",
            "bp_schedule": SCHEDULE,
            "alpha": 0.1,
            "beta": 0.0,
            "sigma": 0.3,
            "sigma_post": 3.0,
            "blocks_per_snr": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "channel": "BPSK/AWGN/perfect_CSI",
        },
        "snr_results": {},
    }
    for snr in snrs:
        rows = run_candidates_at_snr(
            esn0_db=snr,
            blocks=args.blocks,
            batch=args.batch,
            seed=args.seed,
            alpha=0.1,
            checkpoint=args.checkpoint,
            candidate_factory=lambda decoder: {"fixed_0.3": None},
            reference_name="fixed_0.3",
        )
        result["snr_results"][f"{snr:.3f}"] = rows
        dump_json(args.output, result)
        row = rows["fixed_0.3"]
        print(
            f"fixed {snr:+.2f}: {row['failures']}/{row['blocks']} "
            f"BLER={row['crc_bler']:.6f} CI={row['wilson_95']}",
            flush=True,
        )


def run_waterfall(args):
    snrs = parse_float_list(args.snrs)
    allocations = parse_int_list(args.blocks_per_snr)
    if len(snrs) != len(allocations):
        raise ValueError("snrs and blocks-per-snr must have equal lengths")

    schedule_values = {
        "fixed_0.3": [0.3] * SOURCE_CALLS,
        "geom_0.3_to_0.02": list(geometric_scheduler(0.02).path),
        "geom_0.3_to_0.05": list(geometric_scheduler(0.05).path),
        "geom_0.3_to_0.08": list(geometric_scheduler(0.08).path),
    }
    result = {
        "kind": "beta0_paired_sigma_schedule_waterfall",
        "configuration": {
            "branch": "practical_sigma",
            "decoder": "legacy",
            "bp_schedule": SCHEDULE,
            "alpha": 0.1,
            "beta": 0.0,
            "sigma_post": 3.0,
            "batch": args.batch,
            "seed": args.seed,
            "channel": "BPSK/AWGN/perfect_CSI",
            "paired_payload_noise_within_and_across_snr": True,
        },
        "schedule_values": schedule_values,
        "snr_results": {},
    }
    if args.resume and Path(args.output).is_file():
        previous = json.loads(Path(args.output).read_text(encoding="utf-8"))
        if previous.get("kind") != result["kind"]:
            raise ValueError("resume output has a different experiment kind")
        result["snr_results"].update(previous.get("snr_results", {}))
    for snr, blocks in zip(snrs, allocations):
        snr_key = f"{snr:.3f}"
        if snr_key in result["snr_results"]:
            previous_blocks = {
                row["blocks"] for row in result["snr_results"][snr_key].values()
            }
            if previous_blocks == {blocks}:
                print(f"resume: skipping completed SNR {snr:+.2f}", flush=True)
                continue
            raise ValueError(
                f"resume SNR {snr:+.2f} has blocks {previous_blocks}, "
                f"requested {blocks}"
            )
        rows = run_candidates_at_snr(
            esn0_db=snr,
            blocks=blocks,
            batch=args.batch,
            seed=args.seed,
            alpha=0.1,
            checkpoint=args.checkpoint,
            candidate_factory=lambda decoder: time_candidates(),
            reference_name="fixed_0.3",
        )
        result["snr_results"][snr_key] = rows
        dump_json(args.output, result)
        print(f"saved completed SNR {snr:+.2f} to {args.output}", flush=True)


def regression_features(proxy, chunks):
    p = np.log1p(np.maximum(np.asarray(proxy, dtype=np.float64), 0.0))
    t = np.asarray(chunks, dtype=np.float64) / max(SOURCE_CALLS - 1, 1)
    return np.column_stack(
        [
            np.ones_like(p),
            t,
            t * t,
            p,
            p * p,
            t * p,
        ]
    )


def fit_proxy_regression(profile, proxy_name):
    raw = profile["raw"]
    sigma = np.asarray(raw["sigma_actual"], dtype=np.float64)[
        :, :SOURCE_CALLS
    ]
    proxy = np.asarray(raw[proxy_name], dtype=np.float64)[:, :SOURCE_CALLS]
    chunks = np.broadcast_to(np.arange(SOURCE_CALLS), sigma.shape)
    x = regression_features(proxy.reshape(-1), chunks.reshape(-1))
    target = np.log(np.maximum(sigma.reshape(-1), 1e-4))
    coefficients, _, _, _ = np.linalg.lstsq(x, target, rcond=None)
    predicted = np.exp(x @ coefficients)
    truth = sigma.reshape(-1)
    ss_res = float(np.sum((truth - predicted) ** 2))
    ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
    rmse = float(np.sqrt(np.mean((truth - predicted) ** 2)))

    # Block-wise 5-fold cross-validation, preserving all chunks of a block.
    fold_prediction = np.empty_like(truth)
    block_ids = np.repeat(np.arange(sigma.shape[0]), SOURCE_CALLS)
    for fold in range(5):
        test = (block_ids % 5) == fold
        train = ~test
        fold_coef, _, _, _ = np.linalg.lstsq(
            x[train], target[train], rcond=None
        )
        fold_prediction[test] = np.exp(x[test] @ fold_coef)
    cv_rmse = float(np.sqrt(np.mean((truth - fold_prediction) ** 2)))
    cv_corr = float(np.corrcoef(truth, fold_prediction)[0, 1])
    return coefficients, {
        "feature_order": [
            "1",
            "normalized_chunk",
            "normalized_chunk_squared",
            "log1p_proxy",
            "log1p_proxy_squared",
            "normalized_chunk_x_log1p_proxy",
        ],
        "coefficients": coefficients.astype(float).tolist(),
        "train_r_squared_linear_sigma": (
            None if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        ),
        "train_rmse": rmse,
        "five_fold_block_cv_rmse": cv_rmse,
        "five_fold_block_cv_pearson": cv_corr,
        "target_floor_for_log_fit": 1e-4,
    }


class ProxyRegressionScheduler:
    name = "conditional_profile_regression"
    is_adaptive = True

    def __init__(
        self,
        coefficients,
        scale,
        proxy_name,
        decoder=None,
        num_checks=None,
    ):
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.scale = float(scale)
        self.proxy_name = proxy_name
        self.decoder = decoder
        self.num_checks = num_checks
        self.selected = []

    def _proxy(self, ratios):
        if self.proxy_name == "syndrome_weight":
            return ratios.numpy().astype(np.float64) * float(self.num_checks)
        if self.proxy_name == "mean_abs_llr":
            if self.decoder is None or not self.decoder._last_payload_hist:
                raise RuntimeError("payload history unavailable to LLR scheduler")
            payload = self.decoder._last_payload_hist[-1]
            return (
                tf.reduce_mean(tf.abs(payload), axis=1)
                .numpy()
                .astype(np.float64)
            )
        raise ValueError(self.proxy_name)

    def select_sigma(self, chunk_idx, ratios):
        proxy = self._proxy(ratios)
        chunks = np.full_like(proxy, int(chunk_idx), dtype=np.float64)
        x = regression_features(proxy, chunks)
        estimate = np.exp(x @ self.coefficients)
        sigma = np.clip(self.scale * estimate, SIGMA_MIN, SIGMA_MAX)
        self.selected.append(
            {
                "chunk": int(chunk_idx) + 1,
                "mean": float(np.mean(sigma)),
                "min": float(np.min(sigma)),
                "max": float(np.max(sigma)),
            }
        )
        return tf.constant(sigma, dtype=ratios.dtype)


def load_regressions(profile_path):
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    syndrome_coef, syndrome_diag = fit_proxy_regression(
        profile, "syndrome_weight"
    )
    llr_coef, llr_diag = fit_proxy_regression(profile, "mean_abs_llr")
    return profile, syndrome_coef, syndrome_diag, llr_coef, llr_diag


def state_candidate_factory(
    endpoint,
    syndrome_coef,
    llr_coef,
    llr_scale,
):
    def factory(decoder):
        num_checks = int(decoder._h_sparse.dense_shape[0])
        return {
            f"time_geom_0.3_to_{endpoint:g}": geometric_scheduler(endpoint),
            "syndrome_regression_c1.5": ProxyRegressionScheduler(
                syndrome_coef, 1.5, "syndrome_weight", num_checks=num_checks
            ),
            "syndrome_regression_c2": ProxyRegressionScheduler(
                syndrome_coef, 2.0, "syndrome_weight", num_checks=num_checks
            ),
            "syndrome_regression_c3": ProxyRegressionScheduler(
                syndrome_coef, 3.0, "syndrome_weight", num_checks=num_checks
            ),
            f"mean_abs_llr_regression_c{llr_scale:g}": (
                ProxyRegressionScheduler(
                    llr_coef,
                    llr_scale,
                    "mean_abs_llr",
                    decoder=decoder,
                    num_checks=num_checks,
                )
            ),
        }

    return factory


def run_state(args):
    (
        profile,
        syndrome_coef,
        syndrome_diag,
        llr_coef,
        llr_diag,
    ) = load_regressions(args.profile)
    endpoint = float(args.time_endpoint)
    reference = f"time_geom_0.3_to_{endpoint:g}"
    candidates_factory = state_candidate_factory(
        endpoint,
        syndrome_coef,
        llr_coef,
        args.llr_scale,
    )
    rows = run_candidates_at_snr(
        esn0_db=args.esn0_db,
        blocks=args.blocks,
        batch=args.batch,
        seed=args.seed,
        alpha=0.1,
        checkpoint=args.checkpoint,
        candidate_factory=candidates_factory,
        reference_name=reference,
    )
    result = {
        "kind": "beta0_paired_state_sigma_comparison",
        "configuration": {
            "branch": "practical_sigma",
            "decoder": "legacy",
            "bp_schedule": SCHEDULE,
            "alpha": 0.1,
            "beta": 0.0,
            "sigma_post": 3.0,
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "channel": "BPSK/AWGN/perfect_CSI",
            "paired_payload_and_noise": True,
            "time_reference_endpoint": endpoint,
            "sigma_clip": [SIGMA_MIN, SIGMA_MAX],
            "training_profile": args.profile,
            "training_profile_beta": profile["configuration"]["beta"],
            "mean_abs_llr_predeclared_scale": args.llr_scale,
        },
        "regression": {
            "syndrome_weight_plus_chunk": syndrome_diag,
            "mean_abs_llr_plus_chunk": llr_diag,
        },
        "results": rows,
    }
    dump_json(args.output, result)
    for name, row in rows.items():
        transition = row.get("vs_reference")
        print(
            f"{name}: {row['failures']}/{row['blocks']} "
            f"BLER={row['crc_bler']:.6f} CI={row['wilson_95']} "
            f"vs_time={transition}",
            flush=True,
        )


def selected_strategy_factory(
    *,
    strategy,
    endpoint,
    scale,
    proxy_name,
    profile,
):
    if strategy == "time":
        return lambda decoder: geometric_scheduler(endpoint)

    _, syndrome_coef, _, llr_coef, _ = load_regressions(profile)
    coefficients = (
        syndrome_coef if proxy_name == "syndrome_weight" else llr_coef
    )

    def factory(decoder):
        return ProxyRegressionScheduler(
            coefficients,
            scale,
            proxy_name,
            decoder=decoder if proxy_name == "mean_abs_llr" else None,
            num_checks=int(decoder._h_sparse.dense_shape[0]),
        )

    return factory


def run_alpha(args):
    scheduler_factory = selected_strategy_factory(
        strategy=args.strategy,
        endpoint=args.time_endpoint,
        scale=args.state_scale,
        proxy_name=args.state_proxy,
        profile=args.profile,
    )

    def factory(decoder):
        # A fresh scheduler instance is needed for every alpha arm only if it
        # keeps diagnostic state; candidate labels carry alpha, and the caller
        # resets decoder.alpha before every decode.
        return {
            "alpha_0.05": scheduler_factory(decoder),
            "alpha_0.1": scheduler_factory(decoder),
            "alpha_0.2": scheduler_factory(decoder),
        }

    # Alpha differs per candidate, so use a dedicated paired loop.
    if args.blocks % args.batch:
        raise ValueError("blocks must be divisible by batch")
    crc_encoder, crc_decoder, ldpc, mapper, demapper, awgn = make_system()
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = build_beta0_decoder(ldpc, args.checkpoint, device)
    candidates = factory(decoder)
    alphas = {"alpha_0.05": 0.05, "alpha_0.1": 0.1, "alpha_0.2": 0.2}
    masks = {name: [] for name in candidates}
    rounds = args.blocks // args.batch
    for round_idx in range(rounds):
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
            decoder.alpha = alphas[name]
            decoder.beta = 0.0
            decoder.sigma_scheduler = scheduler
            decoder.denoiser.sigma = FIXED_SIGMA
            final_logits = decoder(channel_llr)
            _, crc_valid = crc_decoder(final_logits)
            masks[name].extend(
                np.asarray(crc_valid.numpy())
                .reshape(-1)
                .astype(bool)
                .tolist()
            )
        print(f"alpha round {round_idx + 1}/{rounds}", flush=True)

    reference = np.asarray(masks["alpha_0.1"], dtype=bool)
    rows = {
        name: summarize_mask(
            values, None if name == "alpha_0.1" else reference
        )
        for name, values in masks.items()
    }
    result = {
        "kind": "beta0_paired_alpha_confirmation",
        "configuration": {
            "branch": "practical_sigma",
            "decoder": "legacy",
            "bp_schedule": SCHEDULE,
            "beta": 0.0,
            "sigma_post": 3.0,
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "channel": "BPSK/AWGN/perfect_CSI",
            "paired_payload_and_noise": True,
            "sigma_strategy": args.strategy,
            "time_endpoint": args.time_endpoint,
            "state_proxy": args.state_proxy,
            "state_scale": args.state_scale,
        },
        "results": rows,
    }
    dump_json(args.output, result)
    for name, row in rows.items():
        print(
            f"{name}: {row['failures']}/{row['blocks']} "
            f"BLER={row['crc_bler']:.6f} CI={row['wilson_95']} "
            f"vs_alpha0.1={row.get('vs_reference')}",
            flush=True,
        )


def merge_waterfall_shards(args):
    result = json.loads(Path(args.base).read_text(encoding="utf-8"))
    snr_key = f"{args.esn0_db:.3f}"
    shards = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.shards.split(",")
        if path.strip()
    ]
    if not shards:
        raise ValueError("at least one shard is required")
    names = list(shards[0]["snr_results"][snr_key])
    combined_masks = {name: [] for name in names}
    shard_metadata = []
    for shard_path, shard in zip(
        [path for path in args.shards.split(",") if path.strip()], shards
    ):
        rows = shard["snr_results"][snr_key]
        if list(rows) != names:
            raise ValueError("candidate mismatch across shards")
        shard_metadata.append(
            {
                "path": shard_path,
                "seed": shard["configuration"]["seed"],
                "blocks": next(iter(rows.values()))["blocks"],
            }
        )
        for name in names:
            combined_masks[name].extend(rows[name]["success_mask"])
    reference = np.asarray(combined_masks["fixed_0.3"], dtype=bool)
    combined_rows = {
        name: summarize_mask(
            values, None if name == "fixed_0.3" else reference
        )
        for name, values in combined_masks.items()
    }
    result["snr_results"][snr_key] = combined_rows
    result.setdefault("sharded_low_bler_points", {})[snr_key] = shard_metadata
    dump_json(args.output, result)
    print(
        f"merged {len(reference)} blocks at Es/N0={args.esn0_db:+.2f} "
        f"into {args.output}",
        flush=True,
    )


def plot_results(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    waterfall = json.loads(
        Path(args.waterfall).read_text(encoding="utf-8")
    )
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    alpha = json.loads(Path(args.alpha).read_text(encoding="utf-8"))

    labels = {
        "fixed_0.3": "fixed σ=0.3",
        "geom_0.3_to_0.02": "geom 0.3→0.02",
        "geom_0.3_to_0.05": "geom 0.3→0.05",
        "geom_0.3_to_0.08": "geom 0.3→0.08",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    snrs = sorted(float(key) for key in waterfall["snr_results"])
    for name, label in labels.items():
        rows = [
            waterfall["snr_results"][f"{snr:.3f}"][name] for snr in snrs
        ]
        y = np.asarray([max(row["crc_bler"], 0.5 / row["blocks"]) for row in rows])
        lo = np.asarray([row["wilson_95"][0] for row in rows])
        hi = np.asarray([row["wilson_95"][1] for row in rows])
        ax.errorbar(
            snrs,
            y,
            yerr=np.vstack([np.maximum(y - lo, 0), np.maximum(hi - y, 0)]),
            marker="o",
            capsize=3,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Es/N0 (dB)")
    ax.set_ylabel("CRC-BLER")
    ax.set_title("β=0 sigma-schedule waterfall")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    def bar_panel(axis, result, title, label_map):
        names = list(result["results"])
        rows = [result["results"][name] for name in names]
        y = np.asarray([row["crc_bler"] for row in rows])
        lo = np.asarray([row["wilson_95"][0] for row in rows])
        hi = np.asarray([row["wilson_95"][1] for row in rows])
        axis.bar(np.arange(len(names)), y, color="#4C78A8")
        axis.errorbar(
            np.arange(len(names)),
            y,
            yerr=np.vstack([y - lo, hi - y]),
            fmt="none",
            color="black",
            capsize=3,
        )
        axis.set_xticks(np.arange(len(names)))
        axis.set_xticklabels(
            [label_map.get(name, name) for name in names],
            rotation=20,
            ha="right",
            fontsize=8,
        )
        for index, value in enumerate(y):
            axis.text(
                index,
                hi[index] + 0.015 * max(hi),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_ylim(0.0, max(hi) * 1.18)
        axis.set_ylabel("CRC-BLER")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.3)

    bar_panel(
        axes[1],
        state,
        "State schedules at knee (−2.7 dB)",
        {
            "time_geom_0.3_to_0.02": "time 0.3→0.02",
            "syndrome_regression_c1.5": "syndrome ×1.5",
            "syndrome_regression_c2": "syndrome ×2",
            "syndrome_regression_c3": "syndrome ×3",
            "mean_abs_llr_regression_c2": "mean|LLR| ×2",
        },
    )
    bar_panel(
        axes[2],
        alpha,
        "Alpha check at knee (−2.7 dB)",
        {
            "alpha_0.05": "α=0.05",
            "alpha_0.1": "α=0.1",
            "alpha_0.2": "α=0.2",
        },
    )
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"saved {output}", flush=True)


def add_common(parser):
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    coarse = sub.add_parser("coarse")
    add_common(coarse)
    coarse.add_argument("--snrs", default="-3.2,-3.0,-2.8,-2.6,-2.4")
    coarse.add_argument("--blocks", type=int, default=256)
    coarse.add_argument(
        "--output", default="results/denoiser_sigma_beta0_coarse.json"
    )

    waterfall = sub.add_parser("waterfall")
    add_common(waterfall)
    waterfall.add_argument("--snrs", required=True)
    waterfall.add_argument("--blocks-per-snr", required=True)
    waterfall.add_argument("--resume", action="store_true")
    waterfall.add_argument(
        "--output", default="results/denoiser_sigma_beta0_waterfall.json"
    )

    state = sub.add_parser("state")
    add_common(state)
    state.add_argument("--esn0-db", type=float, required=True)
    state.add_argument("--blocks", type=int, default=1024)
    state.add_argument("--time-endpoint", type=float, required=True)
    state.add_argument("--llr-scale", type=float, default=2.0)
    state.add_argument(
        "--profile",
        default="results/denoiser_sigma_conditional_256.json",
    )
    state.add_argument(
        "--output", default="results/denoiser_sigma_beta0_state.json"
    )

    alpha = sub.add_parser("alpha")
    add_common(alpha)
    alpha.add_argument("--esn0-db", type=float, required=True)
    alpha.add_argument("--blocks", type=int, default=1024)
    alpha.add_argument("--strategy", choices=("time", "state"), required=True)
    alpha.add_argument("--time-endpoint", type=float, default=0.05)
    alpha.add_argument(
        "--state-proxy",
        choices=("syndrome_weight", "mean_abs_llr"),
        default="syndrome_weight",
    )
    alpha.add_argument("--state-scale", type=float, default=2.0)
    alpha.add_argument(
        "--profile",
        default="results/denoiser_sigma_conditional_256.json",
    )
    alpha.add_argument(
        "--output", default="results/denoiser_sigma_beta0_alpha.json"
    )

    merge = sub.add_parser("merge")
    merge.add_argument("--base", required=True)
    merge.add_argument("--shards", required=True)
    merge.add_argument("--esn0-db", type=float, required=True)
    merge.add_argument("--output", required=True)

    plot = sub.add_parser("plot")
    plot.add_argument(
        "--waterfall",
        default="results/denoiser_sigma_beta0_waterfall.json",
    )
    plot.add_argument(
        "--state", default="results/denoiser_sigma_beta0_state.json"
    )
    plot.add_argument(
        "--alpha", default="results/denoiser_sigma_beta0_alpha.json"
    )
    plot.add_argument(
        "--output", default="results/denoiser_sigma_beta0_waterfall.png"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "coarse":
        run_coarse(arguments)
    elif arguments.mode == "waterfall":
        run_waterfall(arguments)
    elif arguments.mode == "state":
        run_state(arguments)
    elif arguments.mode == "alpha":
        run_alpha(arguments)
    elif arguments.mode == "merge":
        merge_waterfall_shards(arguments)
    else:
        plot_results(arguments)
