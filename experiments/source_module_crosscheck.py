#!/usr/bin/env python3
"""Compare the practical_sigma LDPC wrapper and no-LDPC source module."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from coding.bcjr import bcjr_decode
from decoders import (
    NoLDPCConfig,
    RSCSourceIterativeDecoder,
    SourceCategoricalSISO,
)
from experiments.no_ldpc_smoke import load_score_provider
from experiments.waterfall_common import (
    awgn_llr,
    load_fashion_mnist_bits,
    paired_batches,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esn0-db", type=float, default=2.6)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--dataset-root", default="/tmp/fmnist")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/denoiser.pt"))
    parser.add_argument(
        "--practical-worktree",
        type=Path,
        default=Path("/home/LJH/onlyextrinsic_ada_sigma_practical_sigma"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/source_wiring_diagnostic/ldpc_source_crosscheck.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.reshape(-1).astype(np.float64)
    b = right.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    args = parse_args()
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
    bank = load_fashion_mnist_bits(args.dataset_root)
    _, payload, noise = next(
        paired_batches(
            bank,
            blocks=1,
            batch_size=1,
            esn0_db=args.esn0_db,
            seed=args.seed,
        )
    )
    frame = decoder.encode(payload)
    transmitted_llr = awgn_llr(frame.transmitted_bits, noise, args.esn0_db)
    systematic_llr, parity_llr = decoder.rate_matcher.depuncture_llr(transmitted_llr)
    first_bcjr = bcjr_decode(
        systematic_llr,
        parity_llr,
        decoder.trellis,
        a_priori_llr=np.zeros_like(systematic_llr),
        mode="logmap",
        start_state=0,
        end_state=0,
    )
    source_order = decoder.interleaver.deinterleave(
        first_bcjr.extrinsic_llr[..., : config.rsc_information_length]
    )
    spc_cavity = source_order[..., : config.source_coded_length]
    payload_cavity = spc_cavity.reshape(1, 784, 9)[..., :8].reshape(1, -1)

    # no-LDPC exact categorical adapter.
    provider = load_score_provider(config, args.checkpoint, "cpu")
    no_ldpc_result = SourceCategoricalSISO(8, provider)(payload_cavity)

    # Import the actual practical_sigma TF wrapper without modifying its worktree.
    denoiser_path = args.practical_worktree / "denoiser.py"
    spec = importlib.util.spec_from_file_location("practical_sigma_denoiser", denoiser_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {denoiser_path}")
    practical_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(practical_module)

    import tensorflow as tf

    ldpc_wrapper = practical_module.SoftDenoiser(
        img_h=28,
        img_w=28,
        bits_per_pixel=8,
        sigma_post=3.0,
        device="cpu",
    )
    ldpc_wrapper.load_weights_pt(str(args.checkpoint))
    ldpc_extrinsic = ldpc_wrapper(
        tf.constant(payload_cavity.astype(np.float32)),
        sigma=tf.constant([0.3], dtype=tf.float32),
    ).numpy()
    ldpc_posterior = payload_cavity + ldpc_extrinsic

    difference = no_ldpc_result.extrinsic_llr - ldpc_extrinsic
    posterior_difference = no_ldpc_result.posterior_llr - ldpc_posterior
    current_source = REPOSITORY_ROOT / "source_prior.py"
    practical_source = args.practical_worktree / "source_prior.py"
    output = {
        "experiment": "source_module_crosscheck",
        "input": {
            "same_payload": True,
            "same_systematic_llr_object_copied_to_both_modules": True,
            "shape": list(payload_cavity.shape),
            "esn0_db": args.esn0_db,
            "cavity_hard_bit_accuracy": float(
                np.mean((payload_cavity > 0.0) == payload)
            ),
        },
        "read_only_ldpc_import": str(denoiser_path),
        "source_prior_files_byte_identical": (
            sha256(current_source) == sha256(practical_source)
        ),
        "source_prior_sha256": sha256(current_source),
        "extrinsic": {
            "max_abs_difference": float(np.max(np.abs(difference))),
            "mean_abs_difference": float(np.mean(np.abs(difference))),
            "cosine_similarity": cosine(
                no_ldpc_result.extrinsic_llr, ldpc_extrinsic
            ),
            "sign_equal_fraction": float(
                np.mean(
                    np.sign(no_ldpc_result.extrinsic_llr)
                    == np.sign(ldpc_extrinsic)
                )
            ),
        },
        "posterior": {
            "max_abs_difference": float(np.max(np.abs(posterior_difference))),
            "mean_abs_difference": float(np.mean(np.abs(posterior_difference))),
            "cosine_similarity": cosine(
                no_ldpc_result.posterior_llr, ldpc_posterior
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
