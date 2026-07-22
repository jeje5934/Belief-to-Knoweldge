#!/usr/bin/env python3
"""Paired six-arm baseline matrix; neural compression remains unavailable.

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
        ],
        default=[
            "ldpc_only",
            "ldpc_score_warm",
            "rsc_bcjr_no_spc",
            "rsc_bcjr_spc",
            "rsc_bcjr_score_no_spc",
            "rsc_bcjr_score_spc",
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

    needs_ldpc = any(scheme.startswith("ldpc_") for scheme in args.schemes)
    needs_score = any("score" in scheme for scheme in args.schemes)
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
    if needs_ldpc:
        import tensorflow as tf
        from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
        from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
        from decoder import LDPC5GDecoder_soft

        information = config.crc.append(payload).astype(np.float32)
        ldpc_encoder = LDPC5GEncoder(
            config.payload_length + config.crc.width,
            config.target_length,
            num_bits_per_symbol=1,
        )
        ldpc_bits = ldpc_encoder(tf.constant(information)).numpy().astype(np.uint8)
        total_bp_iterations = sum(args.ldpc_bp_schedule)
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
            ldpc_llr = awgn_llr(ldpc_bits, standard_noise, esn0_db).astype(np.float32)
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
        "missing_scheme": (
            "lossless neural compression + conventional LDPC + CRC: "
            "no implementation/resource was found in the repository"),
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
