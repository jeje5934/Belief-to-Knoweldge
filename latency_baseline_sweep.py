"""PixelCNN/WebP/raw BP trajectories for latency matching.

The runner is TensorFlow/Sionna-only so it stays in a process separate from the
PyTorch source decoder.  One warm BP trajectory supplies snapshots at every
requested budget and records the first CRC-passing iteration.
"""

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

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.mapping import Demapper, Mapper


N_CODEWORD = 12600
CHUNK = 5
SYSTEMS = {
    "pixelcnn": ("channel_streams.npz", 4872),
    "webp": ("webp_streams.npz", 5808),
    "raw": (None, 6272),
}


def wilson(k, n, z=1.96):
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def dump(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_raw_bits():
    (_, _), (images, _) = tf.keras.datasets.fashion_mnist.load_data()
    flat = np.asarray(images, dtype=np.uint8).reshape(len(images), -1)
    return np.unpackbits(flat, axis=1, bitorder="big").astype(np.float32)


def stateless_channel(codeword, esn0_db, seed, round_index):
    mapper = Mapper("pam", num_bits_per_symbol=1)
    demapper = Demapper("app", "pam", num_bits_per_symbol=1)
    transmitted = mapper(codeword)
    noise_variance = tf.cast(
        10.0 ** (-float(esn0_db) / 10.0), codeword.dtype
    )
    if transmitted.dtype.is_complex:
        dtype = transmitted.dtype.real_dtype
        real = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=[seed, round_index],
            dtype=dtype,
        )
        imag = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=[seed + 1, round_index],
            dtype=dtype,
        )
        unit_noise = tf.complex(real, imag) / tf.cast(
            math.sqrt(2.0), transmitted.dtype
        )
    else:
        unit_noise = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=[seed, round_index],
            dtype=transmitted.dtype,
        )
    received = transmitted + unit_noise * tf.cast(
        tf.sqrt(noise_variance), transmitted.dtype
    )
    return demapper(received, noise_variance)


def run_point(*, bits, payload_k, esn0_db, blocks, batch, budgets, seed):
    crc_encoder = CRCEncoder("CRC24A")
    crc_decoder = CRCDecoder(crc_encoder)
    encoder = LDPC5GEncoder(payload_k + 24, N_CODEWORD, num_bits_per_symbol=1)
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
    successes = {
        budget: np.zeros(blocks, dtype=bool) for budget in budgets
    }
    first_iterations = np.full(blocks, float(max(budgets)))
    for start in range(0, blocks, batch):
        stop = min(start + batch, blocks)
        payload = tf.constant(bits[start:stop, :payload_k], dtype=encoder.rdtype)
        codeword = encoder(crc_encoder(payload))
        llr = stateless_channel(
            codeword, esn0_db, seed, start // batch
        )
        message = None
        captured = np.zeros(stop - start, dtype=bool)
        snapshots = {}
        for iteration in range(CHUNK, max(budgets) + 1, CHUNK):
            logits, message = decoder(
                llr, num_iter=CHUNK, msg_v2c=message
            )
            _, valid = hard_crc_decode(crc_decoder, logits)
            valid_np = np.asarray(valid.numpy()).reshape(-1).astype(bool)
            newly = valid_np & ~captured
            first_iterations[start:stop][newly] = float(iteration)
            captured |= valid_np
            if iteration in budgets:
                snapshots[iteration] = captured.copy()
        for budget in budgets:
            successes[budget][start:stop] = snapshots[budget]
        print(
            f"baseline K={payload_k} Es/N0={esn0_db:+.2f} "
            f"batch {stop}/{blocks}",
            flush=True,
        )

    rows = {}
    for budget in budgets:
        success = successes[budget]
        failures = int(np.sum(~success))
        used = np.minimum(first_iterations, float(budget))
        used[~success] = float(budget)
        rows[str(budget)] = {
            "blocks": int(blocks),
            "failures": failures,
            "crc_bler": float(failures / blocks),
            "wilson_95": wilson(failures, blocks),
            "actual_mean_bp_iters": float(np.mean(used)),
            "actual_iteration_distribution": {
                "median": float(np.median(used)),
                "q10": float(np.quantile(used, 0.1)),
                "q90": float(np.quantile(used, 0.9)),
            },
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams-dir", required=True)
    parser.add_argument("--systems", default="pixelcnn,webp,raw")
    parser.add_argument("--budgets", default="20,30,50,100,200")
    parser.add_argument("--esn0", type=float, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    budgets = tuple(
        sorted(int(value) for value in args.budgets.split(",") if value)
    )
    if any(value % CHUNK for value in budgets):
        raise ValueError("budgets must be multiples of five")
    systems = [value for value in args.systems.split(",") if value]
    streams_root = Path(args.streams_dir)
    raw_bits = None

    result = {
        "kind": "latency_matching_bp_baselines",
        "configuration": {
            "channel": "BPSK/AWGN/perfect_CSI",
            "n": N_CODEWORD,
            "crc": "CRC24A",
            "budgets": budgets,
            "bp_chunk": CHUNK,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "paired_unit_noise_across_systems": True,
            "crc_early_stop": "first CRC pass on one warm BP trajectory",
        },
        "systems": {},
    }

    for system in systems:
        filename, expected_k = SYSTEMS[system]
        if filename is None:
            if raw_bits is None:
                raw_bits = load_raw_bits()
            bits = raw_bits
            observed_k = expected_k
        else:
            archive = np.load(streams_root / filename)
            bits = archive["bits"]
            observed_k = int(np.max(archive["lengths"]))
        if observed_k != expected_k:
            raise RuntimeError(
                f"{system}: expected K={expected_k}, observed {observed_k}"
            )
        if args.blocks > len(bits):
            raise ValueError(f"{system}: only {len(bits)} streams available")
        system_result = {
            "payload_bits": observed_k,
            "ldpc_k": observed_k + 24,
            "ldpc_rate": (observed_k + 24) / N_CODEWORD,
            "points": {},
        }
        for point_index, esn0_db in enumerate(args.esn0):
            rows = run_point(
                bits=bits,
                payload_k=observed_k,
                esn0_db=esn0_db,
                blocks=args.blocks,
                batch=args.batch,
                budgets=budgets,
                # Same seed for all systems at a given SNR.
                seed=args.seed + point_index * 10000,
            )
            system_result["points"][f"{esn0_db:.3f}"] = {
                "esn0_db": float(esn0_db),
                "budgets": rows,
            }
            result["systems"][system] = system_result
            dump(args.output, result)


if __name__ == "__main__":
    main()
