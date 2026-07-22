#!/usr/bin/env python3
"""Paired small-sample ablation runner for the no-LDPC path.

This entry point is intentionally limited to the common SPC-expanded encoder.
The `bcjr_only` arm ignores the SPC factor but transmits the same frame, making
it an alpha=0 implementation invariant rather than the unresolved no-SPC
baseline from the full comparison matrix.
"""

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
    SourceCategoricalSISO,
    SourceSPCSISO,
)
from experiments.no_ldpc_smoke import (  # noqa: E402
    awgn_llr,
    build_payload,
    load_score_provider,
)
from experiments.no_ldpc_metrics import image_metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--esn0-db", type=float, nargs="+", default=[4.0, 8.0])
    parser.add_argument("--outer-list", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument(
        "--alpha-list", type=float, nargs="+",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0],
    )
    parser.add_argument("--source", choices=["spc_only", "full_score"], default="spc_only")
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/no_ldpc_compare.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.blocks <= 32:
        raise ValueError("comparison blocks must be in [1, 32] before Stage C")
    if any(value <= 0 for value in args.outer_list):
        raise ValueError("outer iterations must be positive")
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

    no_spc_config = NoLDPCConfig(
        outer_iterations=1,
        alpha_schedule=(0.0,),
        llr_clip=args.llr_clip,
        sigma=args.sigma,
        sigma_post=args.sigma_post,
        interleaver_seed=args.seed,
        bcjr_mode=args.bcjr_mode,
        use_spc=False,
        allow_parity_repetition=True,
    )
    no_spc_decoder = RSCSourceIterativeDecoder(no_spc_config)
    no_spc_frame = no_spc_decoder.encode(payload)

    if args.source == "spc_only":
        source_siso = SourceSPCSISO(
            base_config.source_bits_per_symbol,
            IndependentBitCategoricalProvider(base_config.source_bits_per_symbol),
        )
    else:
        score_provider = load_score_provider(base_config, args.checkpoint, args.device)
        source_siso = SourceSPCSISO(base_config.source_bits_per_symbol, score_provider)
        score_only_siso = SourceCategoricalSISO(
            base_config.source_bits_per_symbol, score_provider)

    rows = []
    for esn0_db in args.esn0_db:
        llr = awgn_llr(frame.transmitted_bits, standard_noise, esn0_db)
        no_spc_llr = awgn_llr(
            no_spc_frame.transmitted_bits, standard_noise, esn0_db)

        started = time.perf_counter()
        no_spc_decoded = no_spc_decoder.decode(no_spc_llr, None)
        no_spc_metrics = no_spc_decoder.metrics(payload, no_spc_decoded)
        no_spc_metrics.update(image_metrics(
            payload,
            no_spc_decoded.payload_bits,
            no_spc_config.source_shape,
            no_spc_config.source_bits_per_symbol,
            no_spc_decoded.crc_valid,
        ))
        no_spc_metrics.update({
            "scheme": "rsc_bcjr_no_spc",
            "esn0_db": float(esn0_db),
            "outer_iterations": 1,
            "alpha": 0.0,
            "bcjr_mode": args.bcjr_mode,
            "elapsed_seconds": time.perf_counter() - started,
            "score_model_calls": 0,
            "bcjr_passes": 1,
            "rate_matching_note": "18 uniformly spaced repeated parity observations",
        })
        rows.append(no_spc_metrics)
        print(json.dumps(no_spc_metrics, sort_keys=True), flush=True)

        baseline_decoder = RSCSourceIterativeDecoder(base_config)
        started = time.perf_counter()
        baseline = baseline_decoder.decode(llr, None)
        baseline_metrics = baseline_decoder.metrics(payload, baseline)
        baseline_metrics.update({
            "scheme": "bcjr_only_same_spc_frame",
            "esn0_db": float(esn0_db),
            "outer_iterations": 1,
            "alpha": 0.0,
            "bcjr_mode": args.bcjr_mode,
            "elapsed_seconds": time.perf_counter() - started,
            "score_model_calls": 0,
            "bcjr_passes": 1,
        })
        rows.append(baseline_metrics)
        print(json.dumps(baseline_metrics, sort_keys=True), flush=True)

        for outer_iterations in args.outer_list:
            for alpha in args.alpha_list:
                config = NoLDPCConfig(
                    outer_iterations=outer_iterations,
                    alpha_schedule=(alpha,),
                    llr_clip=args.llr_clip,
                    sigma=args.sigma,
                    sigma_post=args.sigma_post,
                    interleaver_seed=args.seed,
                    bcjr_mode=args.bcjr_mode,
                )
                decoder = RSCSourceIterativeDecoder(config)
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
                metrics.update({
                    "scheme": args.source,
                    "esn0_db": float(esn0_db),
                    "outer_iterations": outer_iterations,
                    "alpha": float(alpha),
                    "bcjr_mode": args.bcjr_mode,
                    "elapsed_seconds": elapsed,
                    "source_siso_calls": decoded.source_calls,
                    "score_model_calls": (
                        decoded.source_calls if args.source == "full_score" else 0),
                    "bcjr_passes": decoded.bcjr_passes,
                })
                rows.append(metrics)
                print(json.dumps(metrics, sort_keys=True), flush=True)

                if args.source == "full_score":
                    score_only_config = NoLDPCConfig(
                        outer_iterations=outer_iterations,
                        alpha_schedule=(alpha,),
                        llr_clip=args.llr_clip,
                        sigma=args.sigma,
                        sigma_post=args.sigma_post,
                        interleaver_seed=args.seed,
                        bcjr_mode=args.bcjr_mode,
                        use_spc=False,
                        allow_parity_repetition=True,
                    )
                    score_only_decoder = RSCSourceIterativeDecoder(score_only_config)
                    started = time.perf_counter()
                    score_only = score_only_decoder.decode(no_spc_llr, score_only_siso)
                    score_only_metrics = score_only_decoder.metrics(payload, score_only)
                    score_only_metrics.update(image_metrics(
                        payload,
                        score_only.payload_bits,
                        score_only_config.source_shape,
                        score_only_config.source_bits_per_symbol,
                        score_only.crc_valid,
                    ))
                    score_only_metrics.update({
                        "scheme": "rsc_bcjr_score_no_spc",
                        "esn0_db": float(esn0_db),
                        "outer_iterations": outer_iterations,
                        "alpha": float(alpha),
                        "bcjr_mode": args.bcjr_mode,
                        "elapsed_seconds": time.perf_counter() - started,
                        "source_siso_calls": score_only.source_calls,
                        "score_model_calls": score_only.source_calls,
                        "bcjr_passes": score_only.bcjr_passes,
                        "rate_matching_note": (
                            "18 uniformly spaced repeated parity observations"),
                    })
                    rows.append(score_only_metrics)
                    print(json.dumps(score_only_metrics, sort_keys=True), flush=True)

    serializable_config = asdict(base_config)
    serializable_config["crc"] = base_config.crc.name
    output = {
        "experiment": "no_ldpc_compare_small",
        "stage_c_guard": "blocks <= 32",
        "paired_payload": True,
        "paired_standard_noise": True,
        "energy_axis": "Es/N0",
        "baseline_warning": (
            "bcjr_only_same_spc_frame ignores the SPC factor but still transmits "
            "the SPC-expanded frame and exists only for the alpha=0 invariant; "
            "rsc_bcjr_no_spc is the actual no-SPC arm with named parity repetition"),
        "config": serializable_config,
        "frame_metadata": frame.metadata,
        "no_spc_frame_metadata": no_spc_frame.metadata,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
