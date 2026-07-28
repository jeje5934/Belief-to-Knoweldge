"""Equal-BP-budget commercial compression baselines.

TensorFlow/Sionna-only process. It deliberately does not import torch or the
production source decoder. For every codec and channel realization, one warm
BP trajectory is run in five-iteration chunks and CRC24A success is recorded at
budgets 20, 30, 50, and 100.
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
BUDGETS = (20, 30, 50, 100)
CHUNK = 5
CODECS = {
    "pixelcnn": ("channel_streams.npz", 4872, 0.38857142857142857),
    "webp": ("webp_streams.npz", 5808, 0.46285714285714286),
    "gzip": ("gzip_streams.npz", 6088, 0.48507936507936505),
    "png": ("png_streams.npz", 6560, 0.5225396825396825),
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


def linear_knee(rows, target):
    rows = sorted(rows, key=lambda row: row["esn0_db"])
    for left, right in zip(rows[:-1], rows[1:]):
        y0, y1 = left["crc_bler"], right["crc_bler"]
        if (y0 - target) * (y1 - target) > 0 or y0 == y1:
            continue
        x0, x1 = left["esn0_db"], right["esn0_db"]
        return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return None


def dump(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def stateless_channel(codeword, esn0_db, seed, round_index):
    mapper = Mapper("pam", num_bits_per_symbol=1)
    demapper = Demapper("app", "pam", num_bits_per_symbol=1)
    transmitted = mapper(codeword)
    noise_variance = tf.cast(
        10.0 ** (-float(esn0_db) / 10.0),
        codeword.dtype,
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
            math.sqrt(2.0),
            transmitted.dtype,
        )
    else:
        unit_noise = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=[seed, round_index],
            dtype=transmitted.dtype,
        )
    received = transmitted + unit_noise * tf.cast(
        tf.sqrt(noise_variance),
        transmitted.dtype,
    )
    return demapper(received, noise_variance)


def run_codec_point(
    *,
    bits,
    k_container,
    esn0_db,
    blocks,
    batch,
    seed,
):
    crc_encoder = CRCEncoder("CRC24A")
    crc_decoder = CRCDecoder(crc_encoder)
    encoder = LDPC5GEncoder(
        k_container + 24,
        N_CODEWORD,
        num_bits_per_symbol=1,
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
    successes = {
        budget: np.zeros(blocks, dtype=bool) for budget in BUDGETS
    }
    first_iterations = np.full(blocks, 100.0, dtype=np.float64)
    for start in range(0, blocks, batch):
        stop = min(start + batch, blocks)
        payload = tf.constant(
            bits[start:stop, :k_container],
            dtype=encoder.rdtype,
        )
        codeword = encoder(crc_encoder(payload))
        llr = stateless_channel(
            codeword,
            esn0_db,
            seed,
            start // batch,
        )
        message = None
        captured = np.zeros(stop - start, dtype=bool)
        snapshots = {}
        for iteration in range(CHUNK, max(BUDGETS) + 1, CHUNK):
            logits, message = decoder(
                llr,
                num_iter=CHUNK,
                msg_v2c=message,
            )
            _, valid = hard_crc_decode(crc_decoder, logits)
            valid_np = np.asarray(valid.numpy()).reshape(-1).astype(bool)
            newly = valid_np & ~captured
            first_iterations[start:stop][newly] = float(iteration)
            captured |= valid_np
            if iteration in BUDGETS:
                snapshots[iteration] = captured.copy()
        for budget in BUDGETS:
            successes[budget][start:stop] = snapshots[budget]
        print(
            f"codec point Es/N0={esn0_db:+.2f} "
            f"batch {stop}/{blocks}",
            flush=True,
        )
    rows = {}
    for budget in BUDGETS:
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
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--codecs",
        default="webp,gzip,png",
        help="comma-separated subset of pixelcnn,webp,gzip,png",
    )
    parser.add_argument("--esn0", type=float, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, nargs="+", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.esn0) != len(args.blocks):
        raise ValueError("esn0 and blocks lengths differ")
    codecs = [value.strip() for value in args.codecs.split(",") if value.strip()]
    result = {
        "kind": "equal_budget_codec_waterfalls",
        "configuration": {
            "n": N_CODEWORD,
            "crc": "CRC24A",
            "budgets": BUDGETS,
            "bp_chunk": CHUNK,
            "crc_early_stop": (
                "universal first CRC pass on one warm BP trajectory"
            ),
            "channel": "BPSK/AWGN/perfect_CSI",
            "batch": args.batch,
            "seed": args.seed,
        },
        "codecs": {},
    }
    for codec_index, codec in enumerate(codecs):
        filename, expected_k, expected_rate = CODECS[codec]
        archive = np.load(Path(args.results_dir) / filename)
        bits = archive["bits"]
        lengths = archive["lengths"]
        observed_k = int(np.max(lengths))
        if observed_k != expected_k:
            raise RuntimeError(
                f"{codec}: expected K={expected_k}, observed {observed_k}"
            )
        codec_result = {
            "container_bits": observed_k,
            "ldpc_k": observed_k + 24,
            "ldpc_rate": (observed_k + 24) / N_CODEWORD,
            "expected_rate": expected_rate,
            "points": {},
            "knees": {
                str(budget): {"bler_0.1": None, "bler_0.01": None}
                for budget in BUDGETS
            },
        }
        for point_index, (esn0_db, blocks) in enumerate(
            zip(args.esn0, args.blocks)
        ):
            if blocks > len(bits):
                raise ValueError(
                    f"{codec}: requested {blocks}, only {len(bits)} streams"
                )
            rows = run_codec_point(
                bits=bits,
                k_container=observed_k,
                esn0_db=esn0_db,
                blocks=blocks,
                batch=args.batch,
                seed=args.seed + codec_index * 100000 + point_index * 10000,
            )
            codec_result["points"][f"{esn0_db:.3f}"] = {
                "esn0_db": float(esn0_db),
                "budgets": rows,
            }
            for budget in BUDGETS:
                budget_rows = [
                    {
                        "esn0_db": point["esn0_db"],
                        **point["budgets"][str(budget)],
                    }
                    for point in codec_result["points"].values()
                ]
                codec_result["knees"][str(budget)] = {
                    "bler_0.1": linear_knee(budget_rows, 0.1),
                    "bler_0.01": linear_knee(budget_rows, 0.01),
                }
            result["codecs"][codec] = codec_result
            dump(args.output, result)
    dump(args.output, result)


if __name__ == "__main__":
    main()
