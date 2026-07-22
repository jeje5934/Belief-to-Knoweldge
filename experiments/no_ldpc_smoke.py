#!/usr/bin/env python3
"""Small paired AWGN smoke test for the no-LDPC implementation.

This script intentionally uses Es/N0 rather than silently translating through
different code rates. Every scheme receives the same payload, encoded frame,
and normalized Gaussian noise realization at each Es/N0 point.
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
    ScorePriorCategoricalProvider,
    SourceSPCSISO,
)
from experiments.no_ldpc_metrics import image_metrics  # noqa: E402


def awgn_llr(bits: np.ndarray, standard_noise: np.ndarray, esn0_db: float) -> np.ndarray:
    """BPSK LLR for x=2b-1 and real noise variance N0/2."""
    n0 = 10.0 ** (-float(esn0_db) / 10.0)
    symbols = 2.0 * bits.astype(np.float64) - 1.0
    received = symbols + np.sqrt(n0 / 2.0) * standard_noise
    return 4.0 * received / n0


def load_score_provider(config: NoLDPCConfig, checkpoint: Path, device: str):
    import torch
    from source_prior import SourcePriorDenoiser

    if not checkpoint.is_file():
        raise FileNotFoundError(f"score checkpoint not found: {checkpoint}")
    prior = SourcePriorDenoiser(
        img_h=config.source_shape[0],
        img_w=config.source_shape[1],
        bits_per_pixel=config.source_bits_per_symbol,
        sigma_post=config.sigma_post,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    prior.load_state_dict(state)
    prior.eval()
    return ScorePriorCategoricalProvider(
        prior, sigma=config.sigma, sigma_post=config.sigma_post)


def load_score_siso(config: NoLDPCConfig, checkpoint: Path, device: str):
    return SourceSPCSISO(
        config.source_bits_per_symbol,
        load_score_provider(config, checkpoint, device),
    )


def build_payload(args, config: NoLDPCConfig, rng: np.random.Generator) -> np.ndarray:
    if args.payload_source == "random":
        return rng.integers(
            0, 2, size=(args.blocks, config.payload_length), dtype=np.uint8)

    import torchvision

    dataset = torchvision.datasets.FashionMNIST(
        root=str(args.dataset_root), train=False, download=args.download_dataset)
    indices = rng.choice(len(dataset), size=args.blocks, replace=False)
    images = np.stack(
        [np.asarray(dataset[int(index)][0], dtype=np.uint8) for index in indices],
        axis=0,
    )
    return np.unpackbits(images.reshape(args.blocks, -1), axis=-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--esn0-db", type=float, nargs="+", default=[4.0, 8.0])
    parser.add_argument("--outer-iterations", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
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
    parser.add_argument(
        "--schemes", nargs="+",
        choices=["bcjr_only", "spc_only", "full_score"],
        default=["bcjr_only", "spc_only"],
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/denoiser.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/no_ldpc_smoke.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.blocks <= 32:
        raise ValueError("smoke test blocks must be in [1, 32]")
    config = NoLDPCConfig(
        outer_iterations=args.outer_iterations,
        alpha_schedule=(args.alpha,),
        llr_clip=args.llr_clip,
        sigma=args.sigma,
        sigma_post=args.sigma_post,
        interleaver_seed=args.seed,
    )
    decoder = RSCSourceIterativeDecoder(config)
    rng = np.random.default_rng(args.seed)
    payload = build_payload(args, config, rng)
    frame = decoder.encode(payload)
    if frame.transmitted_bits.shape[-1] != config.target_length:
        raise AssertionError("transmitted resource length invariant failed")
    standard_noise = rng.standard_normal(frame.transmitted_bits.shape)

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

    results = []
    for esn0_db in args.esn0_db:
        llr = awgn_llr(frame.transmitted_bits, standard_noise, esn0_db)
        for scheme in args.schemes:
            active_decoder = (
                decoder.with_alpha(0.0, args.outer_iterations)
                if scheme == "bcjr_only" else decoder)
            started = time.perf_counter()
            decoded = active_decoder.decode(llr, source_siso[scheme])
            elapsed = time.perf_counter() - started
            if not np.all(np.isfinite(decoded.payload_llr)):
                raise AssertionError(f"{scheme} emitted NaN/Inf payload LLR")
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
            results.append(metrics)
            print(
                f"{scheme:>10} Es/N0={esn0_db:>5.2f} dB "
                f"BLER={metrics['true_payload_bler']:.4f} "
                f"CRC-BLER={metrics['crc_detected_bler']:.4f} "
                f"BER={metrics['ber']:.3e} "
                f"undetected={metrics['undetected_errors']} "
                f"time={elapsed:.2f}s",
                flush=True,
            )

    serializable_config = asdict(config)
    serializable_config["crc"] = {
        "name": config.crc.name,
        "width": config.crc.width,
        "polynomial": hex(config.crc.polynomial),
        "init": hex(config.crc.init),
        "xor_out": hex(config.crc.xor_out),
    }
    output = {
        "experiment": "no_ldpc_smoke",
        "paired_payload": True,
        "paired_standard_noise": True,
        "payload_source": args.payload_source,
        "energy_axis": "Es/N0",
        "config": serializable_config,
        "frame_metadata": frame.metadata,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
