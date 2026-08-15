#!/usr/bin/env python3
"""Compression + post-decode multiview fusion under the raw multiview pairing.

The target image indices, Es/N0, BPSK unit-noise tensor, synthetic side view,
and receiver-estimated registration match multiview_altproj_study.py.  The raw
5-iteration BP image used for registration is deliberately shared with the
compression baselines: a compressed receiver cannot form that image on a CRC
failure, so this is a generous (baseline-favouring) registration grant.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
from zipfile import BadZipFile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import torch
from PIL import Image, UnidentifiedImageError
from torchvision.datasets import FashionMNIST

from crc_utils import hard_crc_decode
from denoiser_sigma_alignment_diag import (
    BPP,
    IMG_H,
    IMG_W,
    K_PAYLOAD,
    N_CODEWORD,
    load_fashion_mnist,
)
from denoiser_sigma_conditional_diag import make_channel_batch, make_system, wilson
from denoiser_sigma_three_scheme import DEFAULT_CHECKPOINT
from multiview_altproj_study import BIT_WEIGHTS, estimate_alignment_grid
from multiview_gate import STRENGTHS, make_views
from source_prior import SourcePriorDenoiser
from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.mapping import Demapper, Mapper


PIXELS = IMG_H * IMG_W
CHUNK = 5
BP_BUDGET = 100
SOURCE_CALLS = 19
DEFAULT_STREAM_ROOT = Path(
    "/home/LJH/onlyextrinsic_ada_sigma/compression_baseline/results"
)
STREAMS = {
    "webp": ("webp_streams.npz", 5808),
    "pixelcnn": ("channel_streams.npz", 4872),
}
POST_CALLS = {
    "side_direct": 0,
    "side_denoise_once_s0.1": 1,
    "side_denoise_once_s0.05": 1,
    "side_denoise_once_s0.02": 1,
    "side_denoise_iter4": 4,
    "side_denoise_iter8": 8,
    "side_denoise_iter19": 19,
}


def dump(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n")


def bootstrap_mean_ci(values, seed, draws=1200):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(draws, len(x)))
    means = x[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def mse_summary(values, seed):
    x = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(x))
    return {
        "blocks": int(len(x)),
        "mean_mse_01": mean,
        "bootstrap_mean_95": bootstrap_mean_ci(x, seed),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "aggregate_psnr_db": None if mean == 0.0 else float(-10.0 * np.log10(mean)),
        "per_block_mse_01": x.tolist(),
    }


def decode_webp_bytes(raw):
    try:
        with Image.open(io.BytesIO(raw)) as image:
            decoded = np.asarray(image.convert("L"), dtype=np.uint8)
        return decoded if decoded.shape == (IMG_H, IMG_W) else None
    except (OSError, ValueError, UnidentifiedImageError):
        return None


def load_stream_archive(path, attempts=3):
    """Eagerly materialize NPZ arrays to avoid repeated remote-filesystem reads."""
    last_error = None
    for _ in range(attempts):
        try:
            with np.load(path) as loaded:
                return {name: np.array(loaded[name], copy=True) for name in loaded.files}
        except BadZipFile as error:
            last_error = error
    raise last_error


def channel_llr(codeword, esn0_db, point_seed, round_index, mapper, demapper):
    transmitted = mapper(codeword)
    noise_variance = tf.cast(10.0 ** (-float(esn0_db) / 10.0), codeword.dtype)
    if transmitted.dtype.is_complex:
        dtype = transmitted.dtype.real_dtype
        real = tf.random.stateless_normal(
            tf.shape(transmitted), seed=[point_seed + 1, round_index], dtype=dtype
        )
        imag = tf.random.stateless_normal(
            tf.shape(transmitted), seed=[point_seed + 2, round_index], dtype=dtype
        )
        unit_noise = tf.complex(real, imag) / tf.cast(
            math.sqrt(2.0), transmitted.dtype
        )
    else:
        unit_noise = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=[point_seed + 1, round_index],
            dtype=transmitted.dtype,
        )
    received = transmitted + unit_noise * tf.cast(
        tf.sqrt(noise_variance), transmitted.dtype
    )
    return demapper(received, noise_variance)


def build_codec_system(k_container):
    crc_encoder = CRCEncoder("CRC24A")
    crc_decoder = CRCDecoder(crc_encoder)
    encoder = LDPC5GEncoder(
        k_container + 24, N_CODEWORD, num_bits_per_symbol=1
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
    return crc_encoder, crc_decoder, encoder, decoder


def decode_codec_batch(
    archive,
    indices,
    system,
    esn0_db,
    point_seed,
    round_index,
    mapper,
    demapper,
):
    crc_encoder, crc_decoder, encoder, decoder = system
    k_container = int(archive["bits"].shape[1])
    payload_np = archive["bits"][indices, :k_container].astype(np.float32)
    payload = tf.constant(payload_np, dtype=encoder.rdtype)
    llr = channel_llr(
        encoder(crc_encoder(payload)), esn0_db, point_seed, round_index,
        mapper, demapper,
    )
    batch = len(indices)
    captured = np.zeros(batch, dtype=bool)
    captured_bits = np.zeros((batch, k_container), dtype=bool)
    message = None
    final_hard = None
    first_iteration = np.full(batch, BP_BUDGET, dtype=np.int32)
    for iteration in range(CHUNK, BP_BUDGET + 1, CHUNK):
        logits, message = decoder(llr, num_iter=CHUNK, msg_v2c=message)
        _, valid_tensor = hard_crc_decode(crc_decoder, logits)
        valid = np.asarray(valid_tensor.numpy()).reshape(-1).astype(bool)
        hard = np.asarray(logits.numpy())[:, :k_container] > 0.0
        newly = valid & ~captured
        captured_bits[newly] = hard[newly]
        first_iteration[newly] = iteration
        captured |= valid
        final_hard = hard
    captured_bits[~captured] = final_hard[~captured]
    lengths = archive["lengths"][indices].astype(int)
    stream_exact = np.asarray(
        [
            np.array_equal(
                captured_bits[local, :length],
                archive["bits"][index, :length].astype(bool),
            )
            for local, (index, length) in enumerate(zip(indices, lengths))
        ],
        dtype=bool,
    )
    return {
        "crc_success": captured,
        "stream_exact": stream_exact,
        "delivered": captured & stream_exact,
        "recovered_bits": captured_bits,
        "lengths": lengths,
        "first_iteration": first_iteration,
    }


def build_post_candidates(prior, aligned):
    device = next(prior.parameters()).device
    base = torch.from_numpy(aligned[:, None]).float().to(device)
    outputs = {"side_direct": aligned.astype(np.float32)}
    with torch.no_grad():
        for sigma in (0.1, 0.05, 0.02):
            sigma_t = torch.full((len(base),), sigma, device=device)
            value = prior.net(base, sigma_t).clamp(0.0, 1.0)
            outputs[f"side_denoise_once_s{sigma}"] = value[:, 0].cpu().numpy()
        state = base
        path = np.geomspace(0.3, 0.02, SOURCE_CALLS)
        for index, sigma in enumerate(path, start=1):
            sigma_t = torch.full((len(base),), float(sigma), device=device)
            state = prior.net(state, sigma_t).clamp(0.0, 1.0)
            if index in (4, 8, 19):
                outputs[f"side_denoise_iter{index}"] = state[:, 0].cpu().numpy()
    return outputs


def init_codec_accum():
    return {
        "crc_success": [],
        "stream_exact": [],
        "delivered": [],
        "first_iteration": [],
        "d_mse": {
            "all_zero": [],
            "global_constant": [],
            "dataset_mean_image": [],
            "best_effort_webp_then_mean": [],
        },
        "e_mse": {name: [] for name in POST_CALLS},
        "e_exact": {name: [] for name in POST_CALLS},
        "best_effort_webp_decode": [],
    }


def summarize_codec(accum, seed, primary_post):
    crc = np.asarray(accum["crc_success"], dtype=bool)
    stream = np.asarray(accum["stream_exact"], dtype=bool)
    delivered = np.asarray(accum["delivered"], dtype=bool)
    failures = int(np.sum(~crc))
    d_stats = {
        name: mse_summary(values, seed + i * 101)
        for i, (name, values) in enumerate(accum["d_mse"].items())
        if len(values)
    }
    e_stats = {
        name: {
            **mse_summary(values, seed + 1000 + i * 101),
            "denoiser_calls": POST_CALLS[name],
            "exact_output_rate": float(np.mean(accum["e_exact"][name])),
        }
        for i, (name, values) in enumerate(accum["e_mse"].items())
    }
    d_best = min(d_stats, key=lambda name: d_stats[name]["mean_mse_01"])
    e_candidates = {**d_stats, **e_stats}
    e_best = min(e_candidates, key=lambda name: e_candidates[name]["mean_mse_01"])
    return {
        "blocks": int(len(crc)),
        "crc_failures": failures,
        "crc_bler": float(np.mean(~crc)),
        "wilson_95": wilson(failures, len(crc)),
        "true_delivery_failures": int(np.sum(~delivered)),
        "crc_pass_stream_wrong": int(np.sum(crc & ~stream)),
        "crc_fail_stream_exact": int(np.sum(~crc & stream)),
        "crc_success_mask": crc.tolist(),
        "delivered_mask": delivered.tolist(),
        "mean_bp_iters": float(np.mean(accum["first_iteration"])),
        "D_independent_policies": d_stats,
        "D_lower_envelope_policy": d_best,
        "D_lower_envelope": d_stats[d_best],
        "E_postprocess_policies": e_stats,
        "E_lower_envelope_policy": e_best,
        "E_lower_envelope": e_candidates[e_best],
        "E_primary_policy": primary_post,
        "E_primary": e_stats[primary_post],
        "best_effort_webp_decode_rate": (
            None if not len(accum["best_effort_webp_decode"])
            else float(np.mean(accum["best_effort_webp_decode"]))
        ),
    }


def run(args):
    if args.blocks % args.batch:
        raise ValueError("blocks must be divisible by batch")
    images, bit_bank = load_fashion_mnist()
    images = images[:args.paired_pool]
    bit_bank = bit_bank[:args.paired_pool]
    stream_root = Path(args.stream_root)
    codec_names = [name for name in args.codecs.split(",") if name]
    archives = {
        name: load_stream_archive(stream_root / STREAMS[name][0])
        for name in codec_names
    }
    for name, archive in archives.items():
        expected = STREAMS[name][1]
        if archive["bits"].shape[1] != expected:
            raise RuntimeError(f"{name}: unexpected container length")
        if not np.array_equal(
            archive["imgs"][:args.paired_pool].reshape(-1, IMG_H, IMG_W),
            images[:args.paired_pool],
        ):
            raise RuntimeError(f"{name}: Fashion-MNIST order mismatch")

    crc_encoder, crc_decoder, raw_encoder, mapper, demapper, awgn = make_system()
    raw_bp5 = LDPC5GDecoder(
        raw_encoder,
        cn_update="boxplus-phi",
        vn_update="sum",
        cn_schedule="flooding",
        hard_out=False,
        return_infobits=True,
        num_iter=5,
        llr_max=30.0,
    )
    codec_systems = {
        name: build_codec_system(STREAMS[name][1]) for name in codec_names
    }
    codec_mapper = Mapper("pam", num_bits_per_symbol=1)
    codec_demapper = Demapper("app", "pam", num_bits_per_symbol=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = SourcePriorDenoiser(sigma_post=3.0).to(device)
    prior.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    prior.eval()
    train = FashionMNIST(root="/tmp/fmnist", train=True, download=False)
    train_np = train.data.numpy().astype(np.float32) / 255.0
    global_mean = float(np.mean(train_np))
    mean_image = np.mean(train_np, axis=0)
    strength = next(value for value in STRENGTHS if value.name == args.strength)

    result = {
        "kind": "multiview_compression_postdecode_fusion",
        "configuration": {
            "branch": "codex/multiview",
            "channel": "BPSK/AWGN/perfect_CSI",
            "n": N_CODEWORD,
            "bp_budget": BP_BUDGET,
            "bp_chunk": CHUNK,
            "blocks_per_snr": args.blocks,
            "batch": args.batch,
            "snrs_db": args.esn0,
            "seed": args.seed,
            "paired_pool": args.paired_pool,
            "strength": strength.__dict__,
            "registration": (
                "receiver grid estimate from the shared raw first-5-BP image; "
                "granted to compression baselines although unavailable on a "
                "compressed CRC failure"
            ),
            "side_assumption": "lossless at receiver, zero channel cost",
            "postprocess_calls": POST_CALLS,
            "primary_post": args.primary_post,
            "production_decoder_modified": False,
        },
        "systems": {},
        "points": {},
    }

    for snr_index, snr in enumerate(args.esn0):
        point_seed = args.seed + snr_index * 10000
        accum = {name: init_codec_accum() for name in codec_names}
        reg_mse = {"raw_side": [], "estimated": []}
        for round_index in range(args.blocks // args.batch):
            indices, payload, truth_images, raw_llr = make_channel_batch(
                round_index,
                args.batch,
                point_seed,
                snr,
                images,
                bit_bank,
                crc_encoder,
                raw_encoder,
                mapper,
                demapper,
                awgn,
            )
            truth_u8 = np.rint(truth_images * 255.0).astype(np.uint8)
            first_logits = np.asarray(raw_bp5(raw_llr).numpy())[:, :K_PAYLOAD]
            first_bits = first_logits.reshape(-1, PIXELS, BPP) > 0.0
            first_image = np.sum(
                first_bits * BIT_WEIGHTS[None, None, :], axis=2
            ).reshape(-1, IMG_H, IMG_W) / 255.0
            side, _, _ = make_views(
                truth_images,
                strength,
                point_seed + 500000 + round_index * args.batch,
            )
            aligned, _ = estimate_alignment_grid(side, first_image, strength)
            reg_mse["raw_side"].extend(
                np.mean((side - truth_images) ** 2, axis=(1, 2)).tolist()
            )
            reg_mse["estimated"].extend(
                np.mean((aligned - truth_images) ** 2, axis=(1, 2)).tolist()
            )
            post_candidates = build_post_candidates(prior, aligned)

            for name in codec_names:
                archive = archives[name]
                decoded = decode_codec_batch(
                    archive,
                    indices,
                    codec_systems[name],
                    snr,
                    point_seed,
                    round_index,
                    codec_mapper,
                    codec_demapper,
                )
                delivered = decoded["delivered"]
                acc = accum[name]
                for key in ("crc_success", "stream_exact", "delivered", "first_iteration"):
                    acc[key].extend(np.asarray(decoded[key]).tolist())

                fallback = {
                    "all_zero": np.mean(truth_images**2, axis=(1, 2)),
                    "global_constant": np.mean(
                        (truth_images - global_mean) ** 2, axis=(1, 2)
                    ),
                    "dataset_mean_image": np.mean(
                        (truth_images - mean_image[None]) ** 2, axis=(1, 2)
                    ),
                }
                for policy, values in fallback.items():
                    acc["d_mse"][policy].extend(
                        np.where(delivered, 0.0, values).tolist()
                    )

                if name == "webp":
                    best_effort = []
                    for local, index in enumerate(indices):
                        if delivered[local]:
                            best_effort.append(0.0)
                            acc["best_effort_webp_decode"].append(True)
                            continue
                        nbytes = int(decoded["lengths"][local] // 8)
                        raw = np.packbits(decoded["recovered_bits"][local])[:nbytes].tobytes()
                        image = decode_webp_bytes(raw)
                        ok = image is not None
                        acc["best_effort_webp_decode"].append(ok)
                        estimate = mean_image if image is None else image.astype(np.float32) / 255.0
                        best_effort.append(float(np.mean((estimate - truth_images[local]) ** 2)))
                    acc["d_mse"]["best_effort_webp_then_mean"].extend(best_effort)
                else:
                    acc["d_mse"].pop("best_effort_webp_then_mean", None)

                for policy, estimate in post_candidates.items():
                    per_block = np.mean((estimate - truth_images) ** 2, axis=(1, 2))
                    values = np.where(delivered, 0.0, per_block)
                    quantized = np.rint(np.clip(estimate, 0.0, 1.0) * 255.0).astype(np.uint8)
                    post_exact = np.all(quantized == truth_u8, axis=(1, 2))
                    output_exact = delivered | (~delivered & post_exact)
                    acc["e_mse"][policy].extend(values.tolist())
                    acc["e_exact"][policy].extend(output_exact.tolist())

            print(
                f"postdecode Es/N0={snr:+.2f} round "
                f"{round_index + 1}/{args.blocks // args.batch}",
                flush=True,
            )

        point = {
            "esn0_db": float(snr),
            "seed": point_seed,
            "registration_mse_01": {
                key: mse_summary(values, point_seed + i * 17)
                for i, (key, values) in enumerate(reg_mse.items())
            },
            "codecs": {
                name: summarize_codec(acc, point_seed + i * 1000, args.primary_post)
                for i, (name, acc) in enumerate(accum.items())
            },
        }
        result["points"][f"{snr:.3f}"] = point
        dump(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--esn0", type=float, nargs="+", default=[-3.05])
    parser.add_argument("--seed", type=int, default=20263001)
    parser.add_argument("--paired-pool", type=int, default=3200)
    parser.add_argument("--strength", default="strong")
    parser.add_argument("--codecs", default="webp,pixelcnn")
    parser.add_argument("--stream-root", default=str(DEFAULT_STREAM_ROOT))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--primary-post", choices=tuple(POST_CALLS),
        default="side_denoise_iter19",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    for key, point in result["points"].items():
        print("\nSNR", key)
        for name, row in point["codecs"].items():
            print(
                name,
                "BLER", f"{row['crc_bler']:.4f}",
                "D", row["D_lower_envelope_policy"],
                f"{row['D_lower_envelope']['mean_mse_01']:.6g}",
                "E", row["E_lower_envelope_policy"],
                f"{row['E_lower_envelope']['mean_mse_01']:.6g}",
            )


if __name__ == "__main__":
    main()
