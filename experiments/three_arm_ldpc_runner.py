#!/usr/bin/env python3
"""Read-only practical_sigma adapter for the two 5G-LDPC waterfall arms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf

# The stable repository recipe is TF/BP on CPU and the torch denoiser on CUDA.
tf.config.set_visible_devices([], "GPU")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

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
    parser.add_argument("--arm", choices=["ldpc_only", "ldpc_score_legacy"], required=True)
    parser.add_argument("--esn0-db", type=float, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--dataset-root", default="/tmp/fmnist")
    parser.add_argument(
        "--practical-worktree",
        type=Path,
        default=Path("/home/LJH/onlyextrinsic_ada_sigma_practical_sigma"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/LJH/onlyextrinsic_ada_sigma/checkpoints/denoiser.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.practical_worktree.is_dir():
        raise FileNotFoundError(args.practical_worktree)
    if args.arm == "ldpc_score_legacy" and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
    from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
    from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder

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
    if args.arm == "ldpc_only":
        decoder = LDPC5GDecoder(encoder, num_iter=100, **common)
        torch_device = None
    else:
        sys.path.insert(0, str(args.practical_worktree.resolve()))
        import torch
        from decoder import LDPC5GDecoder_soft

        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        decoder = LDPC5GDecoder_soft(
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
        decoder.denoiser.load_weights_pt(str(args.checkpoint))
        decoder.denoiser.sigma = 0.3

    bank = load_fashion_mnist_bits(args.dataset_root)
    rows = []
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
            payload_tf = tf.constant(payload, dtype=encoder.rdtype)
            information = crc(payload_tf)
            codeword = encoder(information).numpy().astype(np.uint8)
            if codeword.shape[-1] != TRANSMITTED_BITS:
                raise AssertionError("LDPC encoder violated N=12600")
            logits = tf.constant(
                awgn_llr(codeword, standard_noise, esn0_db),
                dtype=encoder.rdtype,
            )
            decoded_logits = decoder(logits)
            _, crc_valid_tf = crc_decoder(decoded_logits)
            crc_valid = np.asarray(crc_valid_tf.numpy(), dtype=bool).reshape(-1)
            payload_hat = (
                np.asarray(decoded_logits[:, :PAYLOAD_BITS].numpy()) > 0.0
            ).astype(np.uint8)
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
                    f"{args.arm} Es/N0={esn0_db:g} processed={processed}/{args.blocks}",
                    flush=True,
                )
        row = completed_row(
            arm=args.arm,
            esn0_db=esn0_db,
            blocks=args.blocks,
            bit_errors=bit_errors,
            true_block_errors=true_errors,
            crc_block_errors=crc_errors,
            undetected_errors=undetected,
            crc_fail_payload_correct=false_crc,
            elapsed_seconds=time.perf_counter() - started,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    output = {
        "experiment": "three_arm_ldpc_waterfall",
        "arm": args.arm,
        "branch": "practical_sigma",
        "branch_worktree": str(args.practical_worktree),
        "branch_read_only": True,
        "pipeline_source": str(args.practical_worktree / "decoder.py"),
        "payload_bits": PAYLOAD_BITS,
        "transmitted_bits": TRANSMITTED_BITS,
        "payload_rate": PAYLOAD_RATE,
        "energy_axis": "Es/N0",
        "bpsk_mapping": "0 -> -1, 1 -> +1",
        "llr_convention": "log P(1)/P(0); positive -> bit 1",
        "crc": "CRC24A",
        "crc_success_rule": "CRC pass",
        "true_payload_bler_separately_measured": True,
        "paired_plan_seed": args.seed,
        "torch_denoiser_device": torch_device,
        "decoder_config": (
            {"bp_iterations": 100, "score": False}
            if args.arm == "ldpc_only"
            else {
                "bp_schedule": [5] * 20,
                "bp_budget": 100,
                "score": True,
                "ep_mode": False,
                "alpha": 0.1,
                "beta": 0.1,
                "sigma": 0.3,
                "sigma_post": 3.0,
            }
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
