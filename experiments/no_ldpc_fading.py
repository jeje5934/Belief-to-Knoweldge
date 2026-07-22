#!/usr/bin/env python3
"""Paired QPSK fast-fading smoke test with perfect or imperfect CSI.

The existing repository ChannelModel generates one shared channel LLR tensor per
Es/N0 point. Every decoder arm consumes that exact tensor, so fading, noise, and
CSI-error realizations are paired by construction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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
from experiments.no_ldpc_smoke import build_payload, load_score_siso  # noqa: E402
from experiments.no_ldpc_metrics import image_metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--esn0-db", type=float, nargs="+", default=[8.0, 12.0])
    parser.add_argument("--outer-iterations", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--sigma-e2", type=float, default=0.0)
    parser.add_argument("--perfect-csi", action="store_true")
    parser.add_argument("--llr-method", choices=["A", "B"], default="B")
    parser.add_argument("--ar-coeff", type=float, default=0.0)
    parser.add_argument("--llr-clip", type=float, default=30.0)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--sigma-post", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--schemes", nargs="+",
        choices=["bcjr_only", "spc_only", "full_score"],
        default=["bcjr_only", "spc_only"],
    )
    parser.add_argument(
        "--payload-source", choices=["fashion_mnist", "random"],
        default="fashion_mnist",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/tmp/fmnist"))
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/denoiser.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/no_ldpc_fading.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.blocks <= 32:
        raise ValueError("fading smoke blocks must be in [1, 32]")
    if args.sigma_e2 < 0:
        raise ValueError("sigma_e2 must be non-negative")

    import tensorflow as tf
    from channel_models import ChannelModel, QPSK_FADING

    config = NoLDPCConfig(
        outer_iterations=args.outer_iterations,
        alpha_schedule=(args.alpha,),
        llr_clip=args.llr_clip,
        sigma=args.sigma,
        sigma_post=args.sigma_post,
        interleaver_seed=args.seed,
        channel_model=QPSK_FADING,
        imperfect_csi_variance=args.sigma_e2,
    )
    decoder = RSCSourceIterativeDecoder(config)
    rng = np.random.default_rng(args.seed)
    payload = build_payload(args, config, rng)
    frame = decoder.encode(payload)
    transmitted_tf = tf.constant(frame.transmitted_bits, dtype=tf.float32)

    source_siso = {
        "bcjr_only": None,
        "spc_only": SourceSPCSISO(
            config.source_bits_per_symbol,
            IndependentBitCategoricalProvider(config.source_bits_per_symbol),
        ),
    }
    if "full_score" in args.schemes:
        source_siso["full_score"] = load_score_siso(
            config, args.checkpoint, args.device)

    rows = []
    channel_rows = []
    for point_index, esn0_db in enumerate(args.esn0_db):
        channel_seed = args.seed + point_index
        tf.random.set_seed(channel_seed)
        channel = ChannelModel(
            kind=QPSK_FADING,
            sigma_e2=args.sigma_e2,
            perfect_csi=args.perfect_csi,
            llr_method=args.llr_method,
            ar_coeff=args.ar_coeff,
        )
        n0 = tf.constant(10.0 ** (-float(esn0_db) / 10.0), dtype=tf.float32)
        shared_llr = channel.transmit(transmitted_tf, n0).numpy().astype(np.float64)
        h = channel.last["h"].numpy()
        h_hat = channel.last["h_hat"].numpy()
        channel_rows.append({
            "esn0_db": float(esn0_db),
            "tensorflow_seed": channel_seed,
            "mean_channel_power": float(np.mean(np.abs(h) ** 2)),
            "mean_csi_squared_error": float(np.mean(np.abs(h_hat - h) ** 2)),
            "llr_finite": bool(np.all(np.isfinite(shared_llr))),
        })
        if not np.all(np.isfinite(shared_llr)):
            raise AssertionError("channel produced NaN/Inf LLR")

        for scheme in args.schemes:
            active_decoder = (
                decoder.with_alpha(0.0, args.outer_iterations)
                if scheme == "bcjr_only" else decoder)
            started = time.perf_counter()
            decoded = active_decoder.decode(shared_llr, source_siso[scheme])
            elapsed = time.perf_counter() - started
            metrics = active_decoder.metrics(payload, decoded)
            metrics.update(image_metrics(
                payload,
                decoded.payload_bits,
                config.source_shape,
                config.source_bits_per_symbol,
                decoded.crc_valid,
            ))
            metrics.update({
                "scheme": scheme,
                "esn0_db": float(esn0_db),
                "elapsed_seconds": elapsed,
                "source_siso_calls": decoded.source_calls,
                "score_model_calls": (
                    decoded.source_calls if scheme == "full_score" else 0),
                "bcjr_passes": decoded.bcjr_passes,
            })
            rows.append(metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)

    serializable_config = asdict(config)
    serializable_config["crc"] = config.crc.name
    output = {
        "experiment": "no_ldpc_qpsk_fast_fading_smoke",
        "paired_channel_llr": True,
        "energy_axis": "Es/N0",
        "perfect_csi": bool(args.perfect_csi),
        "sigma_e2": args.sigma_e2,
        "llr_method": args.llr_method,
        "ar_coeff": args.ar_coeff,
        "config": serializable_config,
        "frame_metadata": frame.metadata,
        "channel_points": channel_rows,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
