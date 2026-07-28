"""Small characterization of invalid soft-logit CRC versus correct hard CRC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder, LDPCBPDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder

from bp20_failure_diagnostic import (
    CHUNK,
    CRC_LENGTH,
    K_PAYLOAD,
    N_CODEWORD,
    crc_valid,
    load_raw,
    rate_recover,
    stateless_channel,
)


def dump(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esn0", type=float, default=-1.92)
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    images, bits = load_raw()
    del images
    crc_encoder = CRCEncoder("CRC24A")
    crc_decoder = CRCDecoder(crc_encoder)
    encoder = LDPC5GEncoder(
        K_PAYLOAD + CRC_LENGTH, N_CODEWORD, num_bits_per_symbol=1
    )
    decoder = LDPC5GDecoder(
        encoder,
        cn_update="boxplus-phi",
        vn_update="sum",
        cn_schedule="flooding",
        hard_out=False,
        return_infobits=True,
        num_iter=CHUNK,
        llr_max=30.0,
        return_state=True,
    )
    snapshots = {
        iteration: {
            "hard_pass": 0,
            "soft_pass": 0,
            "hard_pass_soft_fail": 0,
            "hard_fail_soft_pass": 0,
            "cumulative_hard_pass": 0,
            "cumulative_soft_pass": 0,
            "cumulative_hard_pass_soft_fail": 0,
            "cumulative_hard_fail_soft_pass": 0,
        }
        for iteration in range(CHUNK, 41, CHUNK)
    }
    first_hard = np.full(args.blocks, -1, dtype=np.int64)
    first_soft = np.full(args.blocks, -1, dtype=np.int64)

    for start in range(0, args.blocks, args.batch):
        stop = min(start + args.batch, args.blocks)
        payload = tf.constant(bits[start:stop], dtype=encoder.rdtype)
        codeword = encoder(crc_encoder(payload))
        llr = stateless_channel(
            codeword, args.esn0, args.seed, start // args.batch
        )
        llr_graph = rate_recover(decoder, llr)
        message = None
        captured_hard = np.zeros(stop - start, dtype=bool)
        captured_soft = np.zeros(stop - start, dtype=bool)
        for iteration in range(CHUNK, 41, CHUNK):
            graph_logits, message = LDPCBPDecoder.call(
                decoder, llr_graph, num_iter=CHUNK, msg_v2c=message
            )
            hard = crc_valid(crc_decoder, graph_logits, encoder.k)
            # Intentionally invalid legacy path: raw logits are truncated to
            # integers inside CRCEncoder before the modulo-2 check.
            _, soft_tensor = crc_decoder(graph_logits[:, : encoder.k])
            soft = np.asarray(soft_tensor.numpy()).reshape(-1).astype(bool)

            hard_new = hard & ~captured_hard
            soft_new = soft & ~captured_soft
            first_hard[start:stop][hard_new] = iteration
            first_soft[start:stop][soft_new] = iteration
            captured_hard |= hard
            captured_soft |= soft

            row = snapshots[iteration]
            row["hard_pass"] += int(hard.sum())
            row["soft_pass"] += int(soft.sum())
            row["hard_pass_soft_fail"] += int((hard & ~soft).sum())
            row["hard_fail_soft_pass"] += int((~hard & soft).sum())
            row["cumulative_hard_pass"] += int(captured_hard.sum())
            row["cumulative_soft_pass"] += int(captured_soft.sum())
            row["cumulative_hard_pass_soft_fail"] += int(
                (captured_hard & ~captured_soft).sum()
            )
            row["cumulative_hard_fail_soft_pass"] += int(
                (~captured_hard & captured_soft).sum()
            )

    for row in snapshots.values():
        row["fake_failure_rate_given_hard_pass"] = rate(
            row["hard_pass_soft_fail"], row["hard_pass"]
        )
        hard_fail = args.blocks - row["hard_pass"]
        row["fake_pass_rate_given_hard_fail"] = rate(
            row["hard_fail_soft_pass"], hard_fail
        )
        row["cumulative_fake_failure_rate_given_hard_pass"] = rate(
            row["cumulative_hard_pass_soft_fail"],
            row["cumulative_hard_pass"],
        )
        cumulative_hard_fail = args.blocks - row["cumulative_hard_pass"]
        row["cumulative_fake_pass_rate_given_hard_fail"] = rate(
            row["cumulative_hard_fail_soft_pass"], cumulative_hard_fail
        )

    both = (first_hard > 0) & (first_soft > 0)
    summary = {
        "kind": "soft_vs_hard_crc_characterization",
        "configuration": vars(args),
        "snapshots": {str(key): value for key, value in snapshots.items()},
        "early_stop": {
            "hard_never_passed_by_40": int((first_hard < 0).sum()),
            "soft_never_passed_by_40": int((first_soft < 0).sum()),
            "both_passed_by_40": int(both.sum()),
            "soft_earlier": int((both & (first_soft < first_hard)).sum()),
            "same_iteration": int((both & (first_soft == first_hard)).sum()),
            "soft_later": int((both & (first_soft > first_hard)).sum()),
            "hard_pass_soft_never": int(((first_hard > 0) & (first_soft < 0)).sum()),
            "hard_never_soft_pass": int(((first_hard < 0) & (first_soft > 0)).sum()),
            "hard_first_iteration_counts": {
                str(value): int((first_hard == value).sum())
                for value in range(CHUNK, 41, CHUNK)
            },
            "soft_first_iteration_counts": {
                str(value): int((first_soft == value).sum())
                for value in range(CHUNK, 41, CHUNK)
            },
        },
    }
    dump(args.output, summary)


if __name__ == "__main__":
    main()
