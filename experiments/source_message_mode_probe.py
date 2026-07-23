#!/usr/bin/env python3
"""Small paired probe of exact-cavity versus legacy inclusive source input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from coding.bcjr import bcjr_decode
from decoders import (
    IndependentBitCategoricalProvider,
    NoLDPCConfig,
    RSCSourceIterativeDecoder,
    SourceSPCSISO,
)
from experiments.no_ldpc_smoke import load_score_provider
from experiments.waterfall_common import (
    awgn_llr,
    completed_row,
    load_fashion_mnist_bits,
    paired_batches,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--esn0-db", type=float, default=2.6)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--dataset-root", default="/tmp/fmnist")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/denoiser.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/source_wiring_diagnostic_fixed/message_mode_probe64.json"),
    )
    return parser.parse_args()


def decode_batch(
    decoder: RSCSourceIterativeDecoder,
    transmitted_llr: np.ndarray,
    source_siso,
    *,
    inclusive_source_input: bool,
):
    config = decoder.config
    systematic_llr, parity_llr = decoder.rate_matcher.depuncture_llr(transmitted_llr)
    prior_information = np.zeros(
        systematic_llr.shape[:-1] + (config.rsc_information_length,),
        dtype=np.float64,
    )
    zero_tail = np.zeros(systematic_llr.shape[:-1] + (config.memory,), dtype=np.float64)
    for iteration in range(config.outer_iterations):
        full_prior = np.concatenate([prior_information, zero_tail], axis=-1)
        channel_result = bcjr_decode(
            systematic_llr,
            parity_llr,
            decoder.trellis,
            a_priori_llr=full_prior,
            mode=config.bcjr_mode,
            start_state=0,
            end_state=0,
        )
        channel_message = (
            channel_result.app_llr
            if inclusive_source_input
            else channel_result.extrinsic_llr
        )
        source_order = decoder.interleaver.deinterleave(
            channel_message[..., : config.rsc_information_length]
        )
        source_cavity = source_order[..., : config.source_coded_length]
        crc_llr = source_order[..., config.source_coded_length :]
        source_result = source_siso(source_cavity)
        scaled = config.alpha_at(iteration) * np.clip(
            source_result.extrinsic_llr, -config.llr_clip, config.llr_clip
        )
        payload_llr = decoder._payload_llr_from_source(source_cavity + scaled)
        feedback = np.concatenate([scaled, np.zeros_like(crc_llr)], axis=-1)
        prior_information = decoder.interleaver.interleave(feedback)
    payload_hat = (payload_llr > 0.0).astype(np.uint8)
    crc_hat = (crc_llr > 0.0).astype(np.uint8)
    crc_valid = np.asarray(
        config.crc.check_parts(payload_hat, crc_hat), dtype=bool
    )
    return payload_hat, crc_valid


def main() -> None:
    args = parse_args()
    if not 1 <= args.blocks <= 128:
        raise ValueError("probe blocks must lie in [1, 128]")
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
    score_siso = SourceSPCSISO(
        8, load_score_provider(config, args.checkpoint, args.device)
    )
    spc_siso = SourceSPCSISO(8, IndependentBitCategoricalProvider(8))
    bank = load_fashion_mnist_bits(args.dataset_root)
    totals = {
        "spc_only_exact_cavity": [0, 0, 0, 0, 0],
        "score_spc_exact_cavity": [0, 0, 0, 0, 0],
        "score_spc_inclusive_app": [0, 0, 0, 0, 0],
    }
    for _, payload, noise in paired_batches(
        bank,
        blocks=args.blocks,
        batch_size=args.batch_size,
        esn0_db=args.esn0_db,
        seed=args.seed,
    ):
        frame = decoder.encode(payload)
        llr = awgn_llr(frame.transmitted_bits, noise, args.esn0_db)
        for label, siso, inclusive in (
            ("spc_only_exact_cavity", spc_siso, False),
            ("score_spc_exact_cavity", score_siso, False),
            ("score_spc_inclusive_app", score_siso, True),
        ):
            payload_hat, crc_valid = decode_batch(
                decoder, llr, siso, inclusive_source_input=inclusive
            )
            bit_error_per_block = np.sum(payload_hat != payload, axis=-1)
            block_error = bit_error_per_block > 0
            totals[label][0] += int(bit_error_per_block.sum())
            totals[label][1] += int(block_error.sum())
            totals[label][2] += int((~crc_valid).sum())
            totals[label][3] += int((crc_valid & block_error).sum())
            totals[label][4] += int((~crc_valid & ~block_error).sum())

    rows = []
    for label, values in totals.items():
        bit_errors, true_errors, crc_errors, undetected, false_crc = values
        row = completed_row(
            arm=label,
            esn0_db=args.esn0_db,
            blocks=args.blocks,
            bit_errors=bit_errors,
            true_block_errors=true_errors,
            crc_block_errors=crc_errors,
            undetected_errors=undetected,
            crc_fail_payload_correct=false_crc,
            elapsed_seconds=0.0,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    output = {
        "experiment": "source_message_mode_probe",
        "paired": True,
        "interpretation": {
            "exact_cavity": "BCJR APP minus the exact source prior injected this pass",
            "inclusive_app": (
                "legacy-like denoiser input retaining the injected source prior"
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
