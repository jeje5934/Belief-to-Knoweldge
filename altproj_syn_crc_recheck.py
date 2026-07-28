"""Recheck the historical altproj ``syn=0 != CRC pass`` claim with hard CRC."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from sionna.phy.utils import ebnodb2no

import decoder as decoder_module
from altproj_screen import (
    BATCH,
    BPP,
    CKPT,
    DEV,
    EBNO,
    IMG_H,
    IMG_W,
    K,
    NBPS,
    SEED,
    bank,
    make,
)
from crc_utils import hard_crc_valid
from decoder import LDPC5GDecoder_soft


def dump(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def first_round(mask_by_round):
    stacked = np.stack(mask_by_round, axis=1)
    any_hit = np.any(stacked, axis=1)
    first = np.full(stacked.shape[0], -1, dtype=np.int64)
    first[any_hit] = np.argmax(stacked[any_hit], axis=1) + 1
    return first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.blocks % BATCH:
        raise ValueError(f"blocks must be divisible by historical batch {BATCH}")

    crc, crcd, ldpc, mapper, demapper, awgn = make()
    bit_bank = bank()
    decoder = LDPC5GDecoder_soft(
        ldpc,
        k_payload=K,
        num_iter=100,
        bp_schedule=[5] * 20,
        adaptive_sigma=False,
        denoiser_kwargs=dict(device=DEV),
        cn_update="boxplus-phi",
        vn_update="sum",
        cn_schedule="flooding",
        hard_out=False,
        return_infobits=True,
        llr_max=30.0,
    )
    checkpoint = Path(CKPT)
    if not checkpoint.exists():
        checkpoint = Path(
            "/home/LJH/onlyextrinsic_ada_sigma/checkpoints/denoiser.pt"
        )
    decoder.denoiser.load_weights_pt(str(checkpoint))
    decoder.denoiser.sigma = 0.3
    decoder.denoiser.sigma_post = 3.0
    decoder.altproj = True
    decoder.altproj_delta = 0.02
    decoder.altproj_rho = 0.9
    decoder.altproj_sigma_den = 0.3
    decoder.altproj_warm_start = True
    decoder.altproj_early_stop = False
    decoder.altproj_crc_check = None
    decoder.track_u_hat = True

    original_syndrome = decoder_module.compute_syndrome_weight_and_ratio
    captured_counts = []

    def capture_syndrome(*call_args, **call_kwargs):
        result = original_syndrome(*call_args, **call_kwargs)
        captured_counts.append(np.asarray(result[0].numpy()).reshape(-1))
        return result

    per_round = [
        {
            "syn0": 0,
            "hard_crc_pass": 0,
            "soft_crc_pass": 0,
            "syn0_hard_crc_fail": 0,
            "syn0_soft_crc_fail": 0,
            "syn0_payload_wrong": 0,
            "syn0_info_wrong": 0,
        }
        for _ in range(20)
    ]
    first_syn_all = []
    first_hard_all = []
    first_soft_all = []

    no = ebnodb2no(EBNO, NBPS, ldpc.coderate)
    rounds = args.blocks // BATCH
    try:
        decoder_module.compute_syndrome_weight_and_ratio = capture_syndrome
        for batch_index in range(rounds):
            tf.random.set_seed(SEED + batch_index)
            indices = tf.random.uniform(
                [BATCH], 0, tf.shape(bit_bank)[0], dtype=tf.int32
            )
            payload = tf.gather(bit_bank, indices)
            info_truth = crc(tf.cast(payload, ldpc.rdtype))
            received = awgn(mapper(ldpc(info_truth)), no)
            llr = demapper(received, no)
            captured_counts.clear()
            decoder(llr)
            history = decoder.last_u_hat_hist
            if len(history) != 20 or len(captured_counts) != 20:
                raise RuntimeError(
                    f"expected 20 rounds, got info={len(history)}, "
                    f"syndrome={len(captured_counts)}"
                )

            syn_masks = []
            hard_masks = []
            soft_masks = []
            truth_np = np.asarray(info_truth.numpy(), dtype=np.uint8)
            payload_np = np.asarray(payload.numpy(), dtype=np.uint8)
            for round_index, (info_logits, syndrome) in enumerate(
                zip(history, captured_counts)
            ):
                syn0 = syndrome == 0
                hard_crc = np.asarray(
                    hard_crc_valid(crcd, info_logits).numpy()
                ).reshape(-1).astype(bool)
                # Historical invalid check, retained only to reproduce direction.
                _, soft_tensor = crcd(info_logits)
                soft_crc = np.asarray(soft_tensor.numpy()).reshape(-1).astype(bool)
                hard_info = (np.asarray(info_logits.numpy()) > 0).astype(np.uint8)
                payload_wrong = np.any(
                    hard_info[:, :K] != payload_np, axis=1
                )
                info_wrong = np.any(hard_info != truth_np, axis=1)

                row = per_round[round_index]
                row["syn0"] += int(syn0.sum())
                row["hard_crc_pass"] += int(hard_crc.sum())
                row["soft_crc_pass"] += int(soft_crc.sum())
                row["syn0_hard_crc_fail"] += int((syn0 & ~hard_crc).sum())
                row["syn0_soft_crc_fail"] += int((syn0 & ~soft_crc).sum())
                row["syn0_payload_wrong"] += int((syn0 & payload_wrong).sum())
                row["syn0_info_wrong"] += int((syn0 & info_wrong).sum())
                syn_masks.append(syn0)
                hard_masks.append(hard_crc)
                soft_masks.append(soft_crc)

            first_syn_all.append(first_round(syn_masks))
            first_hard_all.append(first_round(hard_masks))
            first_soft_all.append(first_round(soft_masks))
    finally:
        decoder_module.compute_syndrome_weight_and_ratio = original_syndrome

    first_syn = np.concatenate(first_syn_all)
    first_hard = np.concatenate(first_hard_all)
    first_soft = np.concatenate(first_soft_all)
    both = (first_syn > 0) & (first_hard > 0)
    comparison = {
        "both_reached": int(both.sum()),
        "same_round": int((both & (first_syn == first_hard)).sum()),
        "syndrome_earlier": int((both & (first_syn < first_hard)).sum()),
        "hard_crc_earlier": int((both & (first_hard < first_syn)).sum()),
        "syndrome_never": int((first_syn < 0).sum()),
        "hard_crc_never": int((first_hard < 0).sum()),
        "soft_crc_never": int((first_soft < 0).sum()),
    }
    result = {
        "kind": "altproj_syn0_hard_crc_recheck",
        "configuration": {
            "blocks": args.blocks,
            "payload_bits": K,
            "n": int(ldpc.n),
            "ldpc_rate": float(ldpc.coderate),
            "ebn0_db": EBNO,
            "equivalent_esn0_db": float(
                EBNO + 10.0 * math.log10(float(ldpc.coderate) * NBPS)
            ),
            "delta": 0.02,
            "rho": 0.9,
            "sigma_den": 0.3,
            "sigma_post": 3.0,
            "schedule": [5] * 20,
            "warm_start": True,
            "early_stop_during_probe": False,
            "batch": BATCH,
            "seed": SEED,
            "checkpoint": str(checkpoint),
        },
        "per_round": {
            str(index + 1): row for index, row in enumerate(per_round)
        },
        "first_event_comparison": comparison,
        "first_syn0_round_counts": {
            str(value): int((first_syn == value).sum()) for value in range(1, 21)
        },
        "first_hard_crc_round_counts": {
            str(value): int((first_hard == value).sum()) for value in range(1, 21)
        },
        "first_soft_crc_round_counts": {
            str(value): int((first_soft == value).sum()) for value in range(1, 21)
        },
    }
    dump(args.output, result)


if __name__ == "__main__":
    main()
