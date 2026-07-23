#!/usr/bin/env python3
"""Paired alpha-schedule search for the RSC/BCJR source decoder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from decoders import (  # noqa: E402
    IndependentBitCategoricalProvider,
    NoLDPCConfig,
    RSCSourceIterativeDecoder,
    SourceSPCSISO,
)
from experiments.no_ldpc_metrics import image_metrics  # noqa: E402
from experiments.no_ldpc_smoke import (  # noqa: E402
    awgn_llr,
    build_payload,
    load_score_provider,
)


DEFAULT_SCHEDULES = [
    "0",
    "0.025",
    "0.05",
    "0.1",
    "0.2",
    "0.4",
    "0.025,0.025",
    "0.05,0.05",
    "0.1,0.1",
    "0.2,0.2",
    "0.025,0.025,0.025,0.025",
    "0.05,0.05,0.05,0.05",
    "0.1,0.1,0.1,0.1",
    "0.2,0.2,0.2,0.2",
]


def wilson_interval(errors: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Two-sided Wilson score interval for a binomial error probability."""

    if total <= 0 or not 0 <= errors <= total:
        raise ValueError("Wilson interval requires 0 <= errors <= total and total > 0")
    proportion = errors / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = z * np.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return [float(max(0.0, center - half_width)),
            float(min(1.0, center + half_width))]


class RecordingSourceSISO:
    """Transparent source-SISO wrapper used only for LLR diagnostics."""

    def __init__(self, source_siso):
        self.source_siso = source_siso
        self.cavity_llr: list[np.ndarray] = []
        self.extrinsic_llr: list[np.ndarray] = []

    def reset(self) -> None:
        self.cavity_llr.clear()
        self.extrinsic_llr.clear()

    def __call__(self, cavity_llr):
        result = self.source_siso(cavity_llr)
        self.cavity_llr.append(np.asarray(cavity_llr, dtype=np.float64).copy())
        self.extrinsic_llr.append(
            np.asarray(result.extrinsic_llr, dtype=np.float64).copy())
        return result


def _abs_quantiles(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    quantiles = np.percentile(absolute, [50.0, 90.0, 95.0, 99.0, 99.9])
    return {
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "p99_9": float(quantiles[4]),
        "max": float(np.max(absolute)),
    }


def llr_diagnostics(
    recorder: RecordingSourceSISO,
    schedule: tuple[float, ...],
    llr_clip: float,
) -> dict:
    """Describe final BCJR APPs and actual source-extrinsic clipping activity."""

    if not recorder.extrinsic_llr:
        return {}
    final_cavity = recorder.cavity_llr[-1]
    if len(recorder.extrinsic_llr) == 1:
        prior_used_by_final_bcjr = np.zeros_like(final_cavity)
    else:
        previous_alpha = schedule[len(recorder.extrinsic_llr) - 2]
        prior_used_by_final_bcjr = previous_alpha * np.clip(
            recorder.extrinsic_llr[-2], -llr_clip, llr_clip)
    final_bcjr_app = final_cavity + prior_used_by_final_bcjr
    all_source_extrinsic = np.concatenate(
        [values.reshape(-1) for values in recorder.extrinsic_llr])
    absolute_source_extrinsic = np.abs(all_source_extrinsic)
    return {
        "final_bcjr_app_abs": _abs_quantiles(final_bcjr_app),
        "final_source_extrinsic_abs": _abs_quantiles(recorder.extrinsic_llr[-1]),
        "all_pass_source_extrinsic_abs": _abs_quantiles(all_source_extrinsic),
        "clip_threshold": float(llr_clip),
        "clip_exceed_count": int(np.count_nonzero(
            absolute_source_extrinsic > llr_clip)),
        "clip_value_count": int(absolute_source_extrinsic.size),
        "clip_exceed_fraction": float(np.mean(
            absolute_source_extrinsic > llr_clip)),
        "final_bcjr_app_exceed_fraction": float(np.mean(
            np.abs(final_bcjr_app) > llr_clip)),
    }


def parse_schedule(value: str) -> tuple[float, ...]:
    try:
        schedule = tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid alpha schedule {value!r}") from error
    if not schedule or any(not np.isfinite(alpha) for alpha in schedule):
        raise argparse.ArgumentTypeError("alpha schedule must be finite and non-empty")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in schedule):
        raise argparse.ArgumentTypeError("alpha values must lie in [0, 1]")
    return schedule


def schedule_label(schedule: tuple[float, ...]) -> str:
    return ",".join(f"{alpha:g}" for alpha in schedule)


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["schedule"], []).append(row)
    summary = []
    for label, group in grouped.items():
        item = {
            "schedule": label,
            "alpha_schedule": group[0]["alpha_schedule"],
            "bcjr_passes": group[0]["bcjr_passes"],
            "snr_points": len(group),
            "evaluated_blocks": int(sum(row["blocks"] for row in group)),
            "true_payload_block_errors": int(sum(
                row["true_payload_block_errors"] for row in group)),
            "bit_errors": int(sum(row["bit_errors"] for row in group)),
            "mean_true_payload_bler": float(np.mean([
                row["true_payload_bler"] for row in group])),
            "mean_crc_detected_bler": float(np.mean([
                row["crc_detected_bler"] for row in group])),
            "total_elapsed_seconds": float(sum(
                row["elapsed_seconds"] for row in group)),
            "total_source_siso_calls": int(sum(
                row["source_siso_calls"] for row in group)),
            "total_score_model_calls": int(sum(
                row["score_model_calls"] for row in group)),
        }
        item["true_payload_bler_wilson95"] = wilson_interval(
            item["true_payload_block_errors"], item["evaluated_blocks"])
        summary.append(item)
    summary.sort(key=lambda item: (
        item["true_payload_block_errors"],
        item["bit_errors"],
        item["bcjr_passes"],
        item["total_elapsed_seconds"],
    ))
    for rank, item in enumerate(summary, start=1):
        item["rank"] = rank
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--esn0-db", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--schedules", nargs="+", default=DEFAULT_SCHEDULES)
    parser.add_argument("--source", choices=["spc_only", "full_score"], default="full_score")
    parser.add_argument("--bcjr-mode", choices=["logmap", "maxlog"], default="logmap")
    parser.add_argument("--llr-clip", type=float, default=30.0)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--sigma-post", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--payload-source", choices=["fashion_mnist", "random"],
        default="fashion_mnist",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/tmp/fmnist"))
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/denoiser.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("results/no_ldpc_schedule_search.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.blocks <= 256:
        raise ValueError("schedule-search blocks must be in [1, 256]")
    schedules = [parse_schedule(value) for value in args.schedules]
    labels = [schedule_label(schedule) for schedule in schedules]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate alpha schedules after normalization")

    base_config = NoLDPCConfig(
        outer_iterations=1,
        alpha_schedule=(0.0,),
        llr_clip=args.llr_clip,
        sigma=args.sigma,
        sigma_post=args.sigma_post,
        interleaver_seed=args.seed,
        bcjr_mode=args.bcjr_mode,
    )
    encoder_decoder = RSCSourceIterativeDecoder(base_config)
    rng = np.random.default_rng(args.seed)
    payload = build_payload(args, base_config, rng)
    frame = encoder_decoder.encode(payload)
    standard_noise = rng.standard_normal(frame.transmitted_bits.shape)

    if args.source == "spc_only":
        base_source_siso = SourceSPCSISO(
            base_config.source_bits_per_symbol,
            IndependentBitCategoricalProvider(base_config.source_bits_per_symbol),
        )
    else:
        base_source_siso = SourceSPCSISO(
            base_config.source_bits_per_symbol,
            load_score_provider(base_config, args.checkpoint, args.device),
        )
    source_siso = RecordingSourceSISO(base_source_siso)

    rows = []
    for esn0_db in args.esn0_db:
        llr = awgn_llr(frame.transmitted_bits, standard_noise, esn0_db)
        for schedule, label in zip(schedules, labels):
            config = NoLDPCConfig(
                outer_iterations=len(schedule),
                alpha_schedule=schedule,
                llr_clip=args.llr_clip,
                sigma=args.sigma,
                sigma_post=args.sigma_post,
                interleaver_seed=args.seed,
                bcjr_mode=args.bcjr_mode,
            )
            decoder = RSCSourceIterativeDecoder(config)
            source_siso.reset()
            started = time.perf_counter()
            decoded = decoder.decode(llr, source_siso)
            elapsed = time.perf_counter() - started
            if not np.all(np.isfinite(decoded.payload_llr)):
                raise AssertionError("non-finite payload LLR")
            metrics = decoder.metrics(payload, decoded)
            metrics.update(image_metrics(
                payload,
                decoded.payload_bits,
                config.source_shape,
                config.source_bits_per_symbol,
                decoded.crc_valid,
            ))
            metrics["true_payload_bler_wilson95"] = wilson_interval(
                metrics["true_payload_block_errors"], metrics["blocks"])
            row = dict(metrics)
            row.update({
                "scheme": args.source,
                "schedule": label,
                "alpha_schedule": list(schedule),
                "esn0_db": float(esn0_db),
                "bcjr_mode": args.bcjr_mode,
                "elapsed_seconds": elapsed,
                "source_siso_calls": decoded.source_calls,
                "score_model_calls": (
                    decoded.source_calls if args.source == "full_score" else 0),
                "bcjr_passes": decoded.bcjr_passes,
                "llr_diagnostics": llr_diagnostics(
                    source_siso, schedule, args.llr_clip),
            })
            rows.append(row)
            print(
                f"schedule={label:>22s} Es/N0={esn0_db:5.2f} "
                f"BLER={row['true_payload_bler']:.4f} "
                f"BER={row['ber']:.3e} time={elapsed:.2f}s",
                flush=True,
            )

    summary = summarize_rows(rows)
    print("\nranked schedules:", flush=True)
    for item in summary:
        print(
            f"{item['rank']:2d}. {item['schedule']:>22s} "
            f"block_errors={item['true_payload_block_errors']:3d}/"
            f"{item['evaluated_blocks']} bit_errors={item['bit_errors']:5d} "
            f"passes={item['bcjr_passes']}",
            flush=True,
        )

    serializable_config = asdict(base_config)
    serializable_config["crc"] = base_config.crc.name
    output = {
        "experiment": "no_ldpc_alpha_schedule_search",
        "selection_warning": (
            "small paired search only; validate finalists on a new seed before selection"
        ),
        "paired_payload": True,
        "paired_standard_noise": True,
        "energy_axis": "Es/N0",
        "source": args.source,
        "bcjr_mode": args.bcjr_mode,
        "seed": args.seed,
        "esn0_db": [float(value) for value in args.esn0_db],
        "config": serializable_config,
        "frame_metadata": frame.metadata,
        "rows": rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
