#!/usr/bin/env python3
"""Paired seven-arm baseline matrix including lossless neural compression.

All implemented arms share the original payload, CRC-16 length, N=12600,
Es/N0, and normalized Gaussian noise realization. Different encoders naturally
transmit different BPSK symbols, but noise sample index i is paired across arms.
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
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

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
    parser.add_argument("--outer-iterations", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ldpc-bp-schedule", type=int, nargs="+", default=[10, 10, 10])
    parser.add_argument("--ldpc-source-alpha", type=float, default=0.1)
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
    parser.add_argument(
        "--compression-checkpoint",
        type=Path,
        default=Path("compression_baseline/results/pixelcnn_fmnist.pt"),
    )
    parser.add_argument("--compression-container-bits", type=int, default=4873)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--schemes", nargs="+",
        choices=[
            "ldpc_only",
            "ldpc_score_warm",
            "rsc_bcjr_no_spc",
            "rsc_bcjr_spc",
            "rsc_bcjr_score_no_spc",
            "rsc_bcjr_score_spc",
            "neural_compression_ldpc",
        ],
        default=[
            "ldpc_only",
            "ldpc_score_warm",
            "rsc_bcjr_no_spc",
            "rsc_bcjr_spc",
            "rsc_bcjr_score_no_spc",
            "rsc_bcjr_score_spc",
            "neural_compression_ldpc",
        ],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/no_ldpc_baseline_matrix.json"))
    return parser.parse_args()


def payload_metrics(payload, payload_llr, crc_llr, crc, config) -> dict:
    payload_hat = (np.asarray(payload_llr) > 0.0).astype(np.uint8)
    crc_hat = (np.asarray(crc_llr) > 0.0).astype(np.uint8)
    bit_errors = np.sum(payload_hat != payload, axis=-1)
    block_error = bit_errors > 0
    crc_valid = np.asarray(crc.check_parts(payload_hat, crc_hat), dtype=bool)
    metrics = {
        "blocks": int(block_error.size),
        "bit_errors": int(bit_errors.sum()),
        "ber": float(bit_errors.sum() / payload.size),
        "true_payload_block_errors": int(block_error.sum()),
        "true_payload_bler": float(block_error.mean()),
        "crc_detected_block_errors": int((~crc_valid).sum()),
        "crc_detected_bler": float((~crc_valid).mean()),
        "undetected_errors": int((crc_valid & block_error).sum()),
    }
    metrics.update(image_metrics(
        payload,
        payload_hat,
        config.source_shape,
        config.source_bits_per_symbol,
        crc_valid,
    ))
    return metrics


def neural_compression_metrics(
    payload, reconstructed_payload, crc_valid, container_error, config
) -> dict:
    bit_errors = np.sum(reconstructed_payload != payload, axis=-1)
    block_error = bit_errors > 0
    crc_valid = np.asarray(crc_valid, dtype=bool)
    container_error = np.asarray(container_error, dtype=bool)
    metrics = {
        "blocks": int(block_error.size),
        "bit_errors": int(bit_errors.sum()),
        "ber": float(bit_errors.sum() / payload.size),
        "true_payload_block_errors": int(block_error.sum()),
        "true_payload_bler": float(block_error.mean()),
        "crc_detected_block_errors": int((~crc_valid).sum()),
        "crc_detected_bler": float((~crc_valid).mean()),
        "undetected_errors": int((crc_valid & block_error).sum()),
        "compression_container_block_errors": int(container_error.sum()),
        "compression_undetected_container_errors": int(
            (crc_valid & container_error).sum()
        ),
        "compression_crc_failures_outages": int((~crc_valid).sum()),
        "attempted_entropy_decodes": int(crc_valid.sum()),
        "successful_exact_decompressions": int((crc_valid & ~block_error).sum()),
        "corrupted_entropy_decode_outputs": int((crc_valid & block_error).sum()),
    }
    metrics.update(image_metrics(
        payload,
        reconstructed_payload,
        config.source_shape,
        config.source_bits_per_symbol,
        crc_valid,
    ))
    return metrics


def timed_row(scheme, esn0_db, function):
    started = time.perf_counter()
    metrics, extra = function()
    row = dict(metrics)
    row.update({
        "scheme": scheme,
        "esn0_db": float(esn0_db),
        "elapsed_seconds": time.perf_counter() - started,
    })
    row.update(extra)
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def run_rsc_arm(decoder, payload, llr, source_siso, *, score_enabled, rate_matching=None):
    decoded = decoder.decode(llr, source_siso)
    extra = {
        "bcjr_passes": decoded.bcjr_passes,
        "source_siso_calls": decoded.source_calls,
        "score_model_calls": decoded.source_calls if score_enabled else 0,
    }
    if rate_matching is not None:
        extra["rate_matching"] = rate_matching
    metrics = decoder.metrics(payload, decoded)
    metrics.update(image_metrics(
        payload,
        decoded.payload_bits,
        decoder.config.source_shape,
        decoder.config.source_bits_per_symbol,
        decoded.crc_valid,
    ))
    return metrics, extra


def main() -> None:
    args = parse_args()
    if not 1 <= args.blocks <= 32:
        raise ValueError("baseline-matrix blocks must be in [1, 32] before Stage C")

    needs_raw_ldpc = any(
        scheme in {"ldpc_only", "ldpc_score_warm"} for scheme in args.schemes)
    needs_compression = "neural_compression_ldpc" in args.schemes
    needs_ldpc = needs_raw_ldpc or needs_compression
    needs_score = any("score" in scheme for scheme in args.schemes)
    if args.compression_container_bits <= 0:
        raise ValueError("compression container must contain at least one bit")
    config = NoLDPCConfig(
        outer_iterations=args.outer_iterations,
        alpha_schedule=(args.alpha,),
        llr_clip=args.llr_clip,
        sigma=args.sigma,
        sigma_post=args.sigma_post,
        interleaver_seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    payload = build_payload(args, config, rng)
    standard_noise = rng.standard_normal((args.blocks, config.target_length))

    spc_decoder = RSCSourceIterativeDecoder(config)
    spc_frame = spc_decoder.encode(payload)
    no_spc_config = NoLDPCConfig(
        outer_iterations=args.outer_iterations,
        alpha_schedule=(args.alpha,),
        llr_clip=args.llr_clip,
        sigma=args.sigma,
        sigma_post=args.sigma_post,
        interleaver_seed=args.seed,
        use_spc=False,
        allow_parity_repetition=True,
    )
    no_spc_decoder = RSCSourceIterativeDecoder(no_spc_config)
    no_spc_frame = no_spc_decoder.encode(payload)
    spc_only_siso = SourceSPCSISO(
        config.source_bits_per_symbol,
        IndependentBitCategoricalProvider(config.source_bits_per_symbol),
    )

    score_spc_siso = score_only_siso = None
    if needs_score:
        score_provider = load_score_provider(config, args.checkpoint, args.device)
        score_spc_siso = SourceSPCSISO(config.source_bits_per_symbol, score_provider)
        score_only_siso = SourceCategoricalSISO(
            config.source_bits_per_symbol, score_provider)

    ldpc_encoder = ldpc_decoder = ldpc_score_decoder = ldpc_bits = None
    compression_model = compression_ldpc_decoder = compression_bits = None
    compression_container = compression_lengths = None
    compression_encode_seconds = None
    if needs_ldpc:
        import tensorflow as tf
        from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
        from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
        total_bp_iterations = sum(args.ldpc_bp_schedule)
        if needs_raw_ldpc:
            information = config.crc.append(payload).astype(np.float32)
            ldpc_encoder = LDPC5GEncoder(
                config.payload_length + config.crc.width,
                config.target_length,
                num_bits_per_symbol=1,
            )
            ldpc_bits = ldpc_encoder(tf.constant(information)).numpy().astype(np.uint8)
            ldpc_decoder = LDPC5GDecoder(
                ldpc_encoder,
                cn_update="boxplus-phi",
                vn_update="sum",
                cn_schedule="flooding",
                hard_out=False,
                return_infobits=True,
                num_iter=total_bp_iterations,
                llr_max=args.llr_clip,
            )
            if "ldpc_score_warm" in args.schemes:
                from decoder import LDPC5GDecoder_soft

                ldpc_score_decoder = LDPC5GDecoder_soft(
                    ldpc_encoder,
                    bp_schedule=args.ldpc_bp_schedule,
                    alpha=args.ldpc_source_alpha,
                    beta=0.0,
                    k_payload=config.payload_length,
                    denoiser_kwargs={
                        "device": args.device,
                        "sigma_post": args.sigma_post,
                    },
                    cn_update="boxplus-phi",
                    vn_update="sum",
                    cn_schedule="flooding",
                    hard_out=False,
                    return_infobits=True,
                    num_iter=total_bp_iterations,
                    llr_max=args.llr_clip,
                )
                ldpc_score_decoder.denoiser.load_weights_pt(str(args.checkpoint))
                ldpc_score_decoder.denoiser.sigma = args.sigma

        if needs_compression:
            import torch
            from compression_baseline import encode_batch, load_model

            compression_model, _ = load_model(
                args.compression_checkpoint, args.device)
            images = np.packbits(payload, axis=-1).reshape(
                args.blocks, *config.source_shape)
            images_tensor = torch.from_numpy(images).unsqueeze(1)
            started = time.perf_counter()
            streams, compression_lengths = encode_batch(
                compression_model, images_tensor, args.device)
            compression_encode_seconds = time.perf_counter() - started
            longest = max(compression_lengths)
            if longest > args.compression_container_bits:
                overflow = [
                    (index, length)
                    for index, length in enumerate(compression_lengths)
                    if length > args.compression_container_bits
                ]
                raise ValueError(
                    "neural-compression container overflow; refusing to truncate: "
                    f"capacity={args.compression_container_bits}, overflow={overflow}"
                )
            compression_container = np.zeros(
                (args.blocks, args.compression_container_bits), dtype=np.uint8)
            for index, stream in enumerate(streams):
                compression_container[index, : stream.size] = stream
            compression_information = config.crc.append(
                compression_container).astype(np.float32)
            compression_ldpc_encoder = LDPC5GEncoder(
                args.compression_container_bits + config.crc.width,
                config.target_length,
                num_bits_per_symbol=1,
            )
            compression_bits = compression_ldpc_encoder(
                tf.constant(compression_information)).numpy().astype(np.uint8)
            compression_ldpc_decoder = LDPC5GDecoder(
                compression_ldpc_encoder,
                cn_update="boxplus-phi",
                vn_update="sum",
                cn_schedule="flooding",
                hard_out=False,
                return_infobits=True,
                num_iter=total_bp_iterations,
                llr_max=args.llr_clip,
            )

    rows = []
    for esn0_db in args.esn0_db:
        spc_llr = awgn_llr(spc_frame.transmitted_bits, standard_noise, esn0_db)
        no_spc_llr = awgn_llr(
            no_spc_frame.transmitted_bits, standard_noise, esn0_db)

        if "rsc_bcjr_no_spc" in args.schemes:
            rows.append(timed_row(
                "rsc_bcjr_no_spc", esn0_db,
                lambda: run_rsc_arm(
                    no_spc_decoder, payload, no_spc_llr, None,
                    score_enabled=False,
                    rate_matching="uniform_18_parity_repetitions",
                ),
            ))
        if "rsc_bcjr_spc" in args.schemes:
            rows.append(timed_row(
                "rsc_bcjr_spc", esn0_db,
                lambda: run_rsc_arm(
                    spc_decoder, payload, spc_llr, spc_only_siso,
                    score_enabled=False,
                ),
            ))
        if "rsc_bcjr_score_no_spc" in args.schemes:
            rows.append(timed_row(
                "rsc_bcjr_score_no_spc", esn0_db,
                lambda: run_rsc_arm(
                    no_spc_decoder, payload, no_spc_llr, score_only_siso,
                    score_enabled=True,
                    rate_matching="uniform_18_parity_repetitions",
                ),
            ))
        if "rsc_bcjr_score_spc" in args.schemes:
            rows.append(timed_row(
                "rsc_bcjr_score_spc", esn0_db,
                lambda: run_rsc_arm(
                    spc_decoder, payload, spc_llr, score_spc_siso,
                    score_enabled=True,
                ),
            ))

        if needs_ldpc:
            import tensorflow as tf
            ldpc_llr = None
            if needs_raw_ldpc:
                ldpc_llr = awgn_llr(
                    ldpc_bits, standard_noise, esn0_db).astype(np.float32)
            if "ldpc_only" in args.schemes:
                def run_ldpc_only():
                    logits = ldpc_decoder(tf.constant(ldpc_llr)).numpy()
                    metrics = payload_metrics(
                        payload,
                        logits[..., : config.payload_length],
                        logits[..., config.payload_length :],
                        config.crc,
                        config,
                    )
                    return metrics, {
                        "bp_iterations": sum(args.ldpc_bp_schedule),
                        "score_model_calls": 0,
                    }
                rows.append(timed_row("ldpc_only", esn0_db, run_ldpc_only))
            if "ldpc_score_warm" in args.schemes:
                def run_ldpc_score():
                    logits = ldpc_score_decoder(tf.constant(ldpc_llr)).numpy()
                    metrics = payload_metrics(
                        payload,
                        logits[..., : config.payload_length],
                        logits[..., config.payload_length :],
                        config.crc,
                        config,
                    )
                    return metrics, {
                        "bp_schedule": list(args.ldpc_bp_schedule),
                        "source_alpha": args.ldpc_source_alpha,
                        "score_model_calls": max(0, len(args.ldpc_bp_schedule) - 1),
                    }
                rows.append(timed_row(
                    "ldpc_score_warm", esn0_db, run_ldpc_score))
            if needs_compression:
                compression_llr = awgn_llr(
                    compression_bits, standard_noise, esn0_db).astype(np.float32)

                def run_neural_compression():
                    from compression_baseline import decode_batch

                    logits = compression_ldpc_decoder(
                        tf.constant(compression_llr)).numpy()
                    container_logits = logits[..., : args.compression_container_bits]
                    crc_logits = logits[..., args.compression_container_bits :]
                    container_hat = (container_logits > 0.0).astype(np.uint8)
                    crc_hat = (crc_logits > 0.0).astype(np.uint8)
                    crc_valid = np.asarray(
                        config.crc.check_parts(container_hat, crc_hat), dtype=bool)
                    container_error = np.any(
                        container_hat != compression_container, axis=-1)

                    reconstructed = np.zeros_like(payload)
                    valid_indices = np.flatnonzero(crc_valid)
                    if valid_indices.size:
                        decoded = decode_batch(
                            compression_model,
                            [container_hat[index] for index in valid_indices],
                            args.device,
                        ).numpy()
                        decoded_bits = np.unpackbits(
                            decoded.reshape(valid_indices.size, -1), axis=-1)
                        reconstructed[valid_indices] = decoded_bits
                    metrics = neural_compression_metrics(
                        payload,
                        reconstructed,
                        crc_valid,
                        container_error,
                        config,
                    )
                    return metrics, {
                        "bp_iterations": sum(args.ldpc_bp_schedule),
                        "score_model_calls": 0,
                        "compression_model_forward_calls": (
                            784 * (args.blocks + int(valid_indices.size))
                        ),
                        "compression_encode_seconds": compression_encode_seconds,
                        "compressed_bits_min": int(min(compression_lengths)),
                        "compressed_bits_mean": float(np.mean(compression_lengths)),
                        "compressed_bits_max": int(max(compression_lengths)),
                        "compression_container_bits": args.compression_container_bits,
                        "ldpc_information_bits": (
                            args.compression_container_bits + config.crc.width
                        ),
                        "transmitted_source_rate_mean": float(
                            np.mean(compression_lengths) / config.target_length
                        ),
                        "effective_ldpc_rate": float(
                            (args.compression_container_bits + config.crc.width)
                            / config.target_length
                        ),
                        "entropy_decode_policy": "CRC-pass blocks only",
                        "compression_probability_batching": (
                            "one image per model forward for partition invariance"
                        ),
                        "crc_failure_reconstruction": "all-zero outage placeholder",
                    }

                rows.append(timed_row(
                    "neural_compression_ldpc", esn0_db, run_neural_compression))

    serializable_config = asdict(config)
    serializable_config["crc"] = config.crc.name
    output = {
        "experiment": "no_ldpc_baseline_matrix_small",
        "paired_payload": True,
        "paired_standard_noise": True,
        "energy_axis": "Es/N0",
        "common_crc": config.crc.name,
        "common_crc_length": config.crc.width,
        "common_transmitted_length": config.target_length,
        "compression_resource_matching": (
            "fixed zero-padded arithmetic stream container; CRC-16 over the full "
            "container; no truncation; conventional 5G LDPC to N=12600"
        ),
        "compression_probability_batching": (
            "canonical per-image model evaluation, independent of batch partition"
        ),
        "config": serializable_config,
        "spc_frame_metadata": spc_frame.metadata,
        "no_spc_frame_metadata": no_spc_frame.metadata,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
