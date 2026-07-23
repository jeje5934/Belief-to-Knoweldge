#!/usr/bin/env python3
"""Chunked RSC/BCJR waterfall and alpha-sweep runner."""

from __future__ import annotations

import argparse
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
from experiments.no_ldpc_smoke import load_score_provider  # noqa: E402
from experiments.waterfall_common import (  # noqa: E402
    PAYLOAD_BITS,
    PAYLOAD_RATE,
    TRANSMITTED_BITS,
    awgn_llr,
    completed_row,
    load_fashion_mnist_bits,
    paired_batches,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=["rsc_score_spc", "rsc_no_spc", "rsc_spc"],
        default="rsc_score_spc",
    )
    parser.add_argument("--esn0-db", type=float, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1])
    parser.add_argument("--llr-clip", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--dataset-root", default="/tmp/fmnist")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/denoiser.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(alpha < 0.0 or alpha > 1.0 for alpha in args.alphas):
        raise ValueError("alpha values must lie in [0, 1]")
    if args.llr_clip <= 0.0:
        raise ValueError("llr_clip must be positive")
    if args.arm != "rsc_score_spc" and args.alphas != [0.1]:
        raise ValueError("alpha sweep is defined only for rsc_score_spc")

    canonical_base = NoLDPCConfig(
        outer_iterations=2,
        alpha_schedule=(0.1, 0.1),
        llr_clip=args.llr_clip,
        sigma=0.3,
        sigma_post=3.0,
        interleaver_seed=20260722,
        bcjr_mode="logmap",
    )
    score_provider = None
    if args.arm == "rsc_score_spc":
        score_provider = load_score_provider(
            canonical_base, args.checkpoint, args.device)
    bank = load_fashion_mnist_bits(args.dataset_root)
    rows = []

    for alpha in args.alphas:
        if args.arm == "rsc_no_spc":
            config = NoLDPCConfig(
                outer_iterations=1,
                alpha_schedule=(0.0,),
                llr_clip=args.llr_clip,
                sigma=0.3,
                sigma_post=3.0,
                interleaver_seed=20260722,
                bcjr_mode="logmap",
                use_spc=False,
                allow_parity_repetition=True,
            )
            source_siso = None
        else:
            config = NoLDPCConfig(
                outer_iterations=2,
                alpha_schedule=(float(alpha), float(alpha)),
                llr_clip=args.llr_clip,
                sigma=0.3,
                sigma_post=3.0,
                interleaver_seed=20260722,
                bcjr_mode="logmap",
            )
            provider = (
                score_provider
                if args.arm == "rsc_score_spc"
                else IndependentBitCategoricalProvider(config.source_bits_per_symbol)
            )
            source_siso = SourceSPCSISO(config.source_bits_per_symbol, provider)
        decoder = RSCSourceIterativeDecoder(config)

        for esn0_db in args.esn0_db:
            bit_errors = true_errors = crc_errors = undetected = false_crc = 0
            processed = 0
            started = time.perf_counter()
            for _, payload, standard_noise in paired_batches(
                bank,
                blocks=args.blocks,
                batch_size=args.batch_size,
                esn0_db=esn0_db,
                seed=args.seed,
            ):
                frame = decoder.encode(payload)
                if frame.transmitted_bits.shape[-1] != TRANSMITTED_BITS:
                    raise AssertionError("RSC encoder violated N=12600")
                logits = awgn_llr(frame.transmitted_bits, standard_noise, esn0_db)
                decoded = decoder.decode(logits, source_siso)
                payload_hat = decoded.payload_bits
                crc_valid = np.asarray(decoded.crc_valid, dtype=bool).reshape(-1)
                per_block_bits = np.sum(payload_hat != payload, axis=-1)
                block_error = per_block_bits > 0
                bit_errors += int(per_block_bits.sum())
                true_errors += int(block_error.sum())
                crc_errors += int((~crc_valid).sum())
                undetected += int((crc_valid & block_error).sum())
                false_crc += int((~crc_valid & ~block_error).sum())
                processed += payload.shape[0]
                if processed % max(args.batch_size * 10, args.batch_size) == 0:
                    print(
                        f"{args.arm} alpha={alpha:g} Es/N0={esn0_db:g} "
                        f"processed={processed}/{args.blocks}",
                        flush=True,
                    )
            label = args.arm if args.alphas == [0.1] else f"{args.arm}_alpha_{alpha:g}"
            row = completed_row(
                arm=label,
                esn0_db=esn0_db,
                blocks=args.blocks,
                bit_errors=bit_errors,
                true_block_errors=true_errors,
                crc_block_errors=crc_errors,
                undetected_errors=undetected,
                crc_fail_payload_correct=false_crc,
                elapsed_seconds=time.perf_counter() - started,
            )
            row.update({
                "alpha": float(alpha),
                "outer_iterations": config.outer_iterations,
                "bcjr_mode": config.bcjr_mode,
                "llr_clip": config.llr_clip,
            })
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    output = {
        "experiment": "three_arm_rsc_waterfall",
        "arm": args.arm,
        "branch": "no-LDPC",
        "payload_bits": PAYLOAD_BITS,
        "transmitted_bits": TRANSMITTED_BITS,
        "payload_rate": PAYLOAD_RATE,
        "energy_axis": "Es/N0",
        "bpsk_mapping": "0 -> -1, 1 -> +1",
        "llr_convention": "log P(1)/P(0); positive -> bit 1",
        "crc": "CRC-16-CCITT-FALSE",
        "crc_success_rule": "CRC pass",
        "true_payload_bler_separately_measured": True,
        "paired_plan_seed": args.seed,
        "canonical_config": {
            "outer_iterations": 2,
            "alpha": 0.1,
            "sigma": 0.3,
            "sigma_post": 3.0,
            "llr_clip": args.llr_clip,
            "bcjr_mode": "logmap",
            "interleaver_seed": 20260722,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
