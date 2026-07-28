"""Classify raw+LDPC BP-20 CRC failures and continue the same BP state to 40.

This is a read-only diagnostic harness around Sionna.  It deliberately avoids
the production ``decoder.py`` and never invokes the source denoiser.
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
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder, LDPCBPDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.mapping import Demapper, Mapper

from syndrome_sigma_schedule import (
    compute_syndrome_weight_and_ratio,
    csr_matrix_to_sparse_tensor,
)


N_CODEWORD = 12600
K_PAYLOAD = 6272
CRC_LENGTH = 24
CHUNK = 5


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
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


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "q10": None,
            "q90": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "q10": float(np.quantile(values, 0.1)),
        "q90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def describe_psnr(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result = describe(finite)
    result["exact_reconstruction_count"] = int(np.sum(~np.isfinite(values)))
    result["total_count"] = int(values.size)
    return result


def load_raw():
    (_, _), (images, _) = tf.keras.datasets.fashion_mnist.load_data()
    images = np.asarray(images, dtype=np.uint8)
    bits = np.unpackbits(
        images.reshape(len(images), -1), axis=1, bitorder="big"
    ).astype(np.float32)
    return images, bits


def stateless_channel(codeword, esn0_db, seed, batch_index):
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
            seed=[seed, batch_index],
            dtype=dtype,
        )
        imag = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=[seed + 1, batch_index],
            dtype=dtype,
        )
        unit_noise = tf.complex(real, imag) / tf.cast(
            math.sqrt(2.0), transmitted.dtype
        )
    else:
        unit_noise = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=[seed, batch_index],
            dtype=transmitted.dtype,
        )
    received = transmitted + unit_noise * tf.cast(
        tf.sqrt(noise_variance), transmitted.dtype
    )
    return demapper(received, noise_variance)


def rate_recover(decoder, llr_ch):
    """Copy Sionna's rate recovery, stopping before the core BP call."""
    llr = tf.reshape(llr_ch, [-1, decoder.encoder.n])
    batch_size = tf.shape(llr)[0]
    if decoder.encoder.num_bits_per_symbol is not None:
        llr = tf.gather(llr, decoder.encoder.out_int_inv, axis=-1)
    llr_5g = tf.concat(
        [tf.zeros([batch_size, 2 * decoder.encoder.z], decoder.rdtype), llr],
        axis=1,
    )
    k_filler = decoder.encoder.k_ldpc - decoder.encoder.k
    nb_punc_bits = (
        decoder.encoder.n_ldpc
        - k_filler
        - decoder.encoder.n
        - 2 * decoder.encoder.z
    )
    llr_5g = tf.concat(
        [
            llr_5g,
            tf.zeros(
                [batch_size, nb_punc_bits - decoder._nb_pruned_nodes],
                decoder.rdtype,
            ),
        ],
        axis=1,
    )
    x1 = tf.slice(llr_5g, [0, 0], [batch_size, decoder.encoder.k])
    nb_par_bits = (
        decoder.encoder.n_ldpc
        - k_filler
        - decoder.encoder.k
        - decoder._nb_pruned_nodes
    )
    x2 = tf.slice(
        llr_5g, [0, decoder.encoder.k], [batch_size, nb_par_bits]
    )
    filler = -tf.cast(decoder.llr_max, decoder.rdtype) * tf.ones(
        [batch_size, k_filler], decoder.rdtype
    )
    return tf.concat([x1, filler, x2], axis=1)


def crc_valid(crc_decoder, graph_logits, k_info):
    # Sionna CRCDecoder expects binary 0/1 values, not soft logits.  Passing
    # logits makes CRCEncoder truncate them to integers before modulo-2 and is
    # therefore not a CRC check.  This repository's established hard-decision
    # convention is logit > 0 -> bit 1.
    hard_info = tf.cast(graph_logits[:, :k_info] > 0.0, tf.float32)
    _, valid = crc_decoder(hard_info)
    return np.asarray(valid.numpy()).reshape(-1).astype(bool)


def payload_metrics(payload_logits, truth_bits, truth_images):
    hard = (np.asarray(payload_logits) > 0.0).astype(np.uint8)
    truth = np.asarray(truth_bits, dtype=np.uint8)
    errors = np.sum(hard != truth, axis=1).astype(np.int64)
    images = np.packbits(hard, axis=1, bitorder="big").reshape(-1, 28, 28)
    delta = images.astype(np.float64) - truth_images.astype(np.float64)
    mse = np.mean(delta * delta, axis=(1, 2))
    psnr = np.full(len(mse), np.inf, dtype=np.float64)
    nonzero = mse > 0.0
    psnr[nonzero] = 20.0 * np.log10(255.0 / np.sqrt(mse[nonzero]))
    return errors, psnr


def category_name(syndrome_weight):
    if syndrome_weight == 0:
        return "valid_but_wrong"
    if syndrome_weight <= 50:
        return "near_convergence"
    return "far_from_convergence"


def summarize_categories(records):
    total = len(records)
    result = {}
    for name in (
        "valid_but_wrong",
        "near_convergence",
        "far_from_convergence",
    ):
        rows = [row for row in records if row["category"] == name]
        rescued = [row for row in rows if row["bp40_rescued"]]
        result[name] = {
            "count": len(rows),
            "fraction_of_bp20_failures": len(rows) / total if total else 0.0,
            "payload_bit_errors": describe(
                [row["payload_bit_errors"] for row in rows]
            ),
            "crc_bit_errors": describe(
                [row["crc_bit_errors"] for row in rows]
            ),
            "info_bit_errors": describe(
                [row["info_bit_errors"] for row in rows]
            ),
            "payload_exact_count": int(
                sum(row["payload_bit_errors"] == 0 for row in rows)
            ),
            "payload_damaged_count": int(
                sum(row["payload_bit_errors"] > 0 for row in rows)
            ),
            "image_psnr_db": describe_psnr(
                [row["image_psnr_db"] for row in rows]
            ),
            "bp40_rescued": len(rescued),
            "bp40_rescue_fraction": len(rescued) / len(rows) if rows else 0.0,
            "bp40_first_pass_iteration": describe(
                [row["bp40_first_pass_iteration"] for row in rescued]
            ),
        }
    return result


def run_point(*, images, bits, esn0_db, blocks, batch, seed, diagnose):
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
    h_sparse = csr_matrix_to_sparse_tensor(decoder.pcm)
    max_iter = 40 if diagnose else 20
    success20 = np.zeros(blocks, dtype=bool)
    success40 = np.zeros(blocks, dtype=bool)
    first_pass = np.full(blocks, -1, dtype=np.int64)
    records = []
    undetected20 = 0

    for start in range(0, blocks, batch):
        stop = min(start + batch, blocks)
        payload_np = bits[start:stop, :K_PAYLOAD]
        payload = tf.constant(payload_np, dtype=encoder.rdtype)
        info_bits = crc_encoder(payload)
        info_np = np.asarray(info_bits.numpy(), dtype=np.uint8)
        codeword = encoder(info_bits)
        llr = stateless_channel(codeword, esn0_db, seed, start // batch)
        llr_graph = rate_recover(decoder, llr)
        message = None
        captured = np.zeros(stop - start, dtype=bool)
        graph20 = None
        fail20 = None

        for iteration in range(CHUNK, max_iter + 1, CHUNK):
            graph_logits, message = LDPCBPDecoder.call(
                decoder,
                llr_graph,
                num_iter=CHUNK,
                msg_v2c=message,
            )
            tf.debugging.assert_equal(
                tf.shape(graph_logits)[-1],
                tf.cast(h_sparse.dense_shape[1], tf.int32),
            )
            valid_now = crc_valid(crc_decoder, graph_logits, encoder.k)
            newly = valid_now & ~captured
            first_pass[start:stop][newly] = iteration
            captured |= valid_now

            if iteration == 20:
                graph20 = np.asarray(graph_logits.numpy())
                success20[start:stop] = captured
                fail20 = ~captured.copy()
                hard_payload = graph20[:, :K_PAYLOAD] > 0.0
                undetected20 += int(
                    np.sum(captured & np.any(hard_payload != payload_np, axis=1))
                )

        success40[start:stop] = captured

        if diagnose:
            counts, _, _ = compute_syndrome_weight_and_ratio(
                tf.constant(graph20), h_sparse
            )
            syndrome = np.asarray(counts.numpy()).reshape(-1).astype(np.int64)
            errors, psnr = payload_metrics(
                graph20[:, :K_PAYLOAD],
                payload_np,
                images[start:stop],
            )
            hard_info = (graph20[:, :encoder.k] > 0.0).astype(np.uint8)
            info_errors = np.sum(hard_info != info_np, axis=1).astype(np.int64)
            crc_errors = np.sum(
                hard_info[:, K_PAYLOAD:] != info_np[:, K_PAYLOAD:], axis=1
            ).astype(np.int64)
            for local_index in np.flatnonzero(fail20):
                global_index = start + int(local_index)
                rescue_iteration = int(first_pass[global_index])
                rescued = rescue_iteration > 20
                records.append(
                    {
                        "block_index": global_index,
                        "syndrome_weight": int(syndrome[local_index]),
                        "category": category_name(int(syndrome[local_index])),
                        "payload_bit_errors": int(errors[local_index]),
                        "crc_bit_errors": int(crc_errors[local_index]),
                        "info_bit_errors": int(info_errors[local_index]),
                        "image_psnr_db": (
                            None
                            if not np.isfinite(psnr[local_index])
                            else float(psnr[local_index])
                        ),
                        "image_exact": bool(not np.isfinite(psnr[local_index])),
                        "bp40_rescued": bool(rescued),
                        "bp40_first_pass_iteration": (
                            rescue_iteration if rescued else None
                        ),
                    }
                )
        print(
            f"BP Es/N0={esn0_db:+.2f} batch {stop}/{blocks}", flush=True
        )

    failures20 = int(np.sum(~success20))
    failures40 = int(np.sum(~success40))
    result = {
        "esn0_db": float(esn0_db),
        "blocks": int(blocks),
        "bp20": {
            "failures": failures20,
            "crc_bler": failures20 / blocks,
            "wilson_95": wilson(failures20, blocks),
            "crc_undetected_payload_errors": int(undetected20),
        },
        "bp40": {
            "failures": failures40,
            "crc_bler": failures40 / blocks,
            "wilson_95": wilson(failures40, blocks),
            "rescued_from_bp20_failures": failures20 - failures40,
            "rescue_fraction_of_bp20_failures": (
                (failures20 - failures40) / failures20 if failures20 else 0.0
            ),
        },
    }
    if diagnose:
        # JSON stores exact image recovery as null PSNR; restore infinity for stats.
        stat_records = []
        for row in records:
            restored = dict(row)
            if restored["image_exact"]:
                restored["image_psnr_db"] = float("inf")
            stat_records.append(restored)
        categories = summarize_categories(stat_records)
        valid_unresolved = (
            categories["valid_but_wrong"]["count"]
            - categories["valid_but_wrong"]["bp40_rescued"]
        )
        far_unresolved = (
            categories["far_from_convergence"]["count"]
            - categories["far_from_convergence"]["bp40_rescued"]
        )
        proxy_count = valid_unresolved + far_unresolved
        result["failure_composition"] = categories
        payload_damaged = sum(
            row["payload_bit_errors"] > 0 for row in records
        )
        result["source_visible_payload_damage"] = {
            "definition": (
                "BP20 CRC failures whose hard payload differs from the "
                "transmitted 6272-bit image"
            ),
            "count": int(payload_damaged),
            "fraction_of_bp20_failures": (
                payload_damaged / failures20 if failures20 else 0.0
            ),
        }
        result["source_only_recoverable_proxy"] = {
            "definition": (
                "BP40-unresolved valid-but-wrong plus BP40-unresolved "
                "far-from-convergence, divided by BP20 CRC failures"
            ),
            "count": int(proxy_count),
            "fraction_of_bp20_failures": (
                proxy_count / failures20 if failures20 else 0.0
            ),
        }
        result["failure_records"] = records
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("scan", "diagnose"), required=True)
    parser.add_argument("--esn0", type=float, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, default=512)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.blocks % args.batch:
        raise ValueError("blocks must be divisible by batch")
    images, bits = load_raw()
    if args.blocks > len(bits):
        raise ValueError(f"only {len(bits)} Fashion-MNIST test blocks available")

    payload = {
        "kind": "bp20_failure_composition_diagnostic",
        "configuration": {
            "mode": args.mode,
            "channel": "BPSK/AWGN/perfect_CSI",
            "payload_bits": K_PAYLOAD,
            "n": N_CODEWORD,
            "crc": "CRC24A",
            "bp": "Sionna 5G LDPC boxplus-phi/flooding, warm state",
            "bp_schedule": "[5]x4 to BP20; failed blocks continue [5]x4 to BP40",
            "crc_early_stop": "check every 5 BP iterations",
            "noise_pairing": "same stateless unit noise across SNR points",
            "syndrome_domain": "full pruned LDPC graph posterior",
            "hard_decision": "graph logit > 0 -> bit 1",
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "production_decoder_modified": False,
        },
        "points": {},
    }
    for esn0_db in args.esn0:
        row = run_point(
            images=images,
            bits=bits,
            esn0_db=esn0_db,
            blocks=args.blocks,
            batch=args.batch,
            seed=args.seed,
            diagnose=args.mode == "diagnose",
        )
        payload["points"][f"{esn0_db:.3f}"] = row
        dump(args.output, payload)


if __name__ == "__main__":
    main()
