"""Small-Monte-Carlo BP-50 image-MSE optimization study.

The source arm and WebP arm are separate subcommands so TensorFlow and PyTorch
never compete for the same CUDA context.  Production decoder.py is imported
unchanged; all parameter changes live in this experiment harness.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path

import numpy as np


IMG_H = 28
IMG_W = 28
BPP = 8
NPIX = IMG_H * IMG_W
K_PAYLOAD = NPIX * BPP
N_CODEWORD = 12600
SCHEDULE = [10] * 5
SOURCE_CALLS = len(SCHEDULE) - 1
DEFAULT_CHECKPOINT = "/home/LJH/onlyextrinsic_ada_sigma/checkpoints/denoiser.pt"
DEFAULT_WEBP = (
    "/home/LJH/onlyextrinsic_ada_sigma/compression_baseline/results/"
    "webp_streams.npz"
)
BIT_WEIGHTS = np.asarray([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.float64)


def dump_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def bootstrap_mean_ci(values, seed, draws=2000):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def describe(values, seed):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "bootstrap_mean_95": bootstrap_mean_ci(values, seed),
    }


def psnr_from_mse(mse_255):
    mse_255 = np.asarray(mse_255, dtype=np.float64)
    return np.where(
        mse_255 <= 0.0,
        99.0,
        10.0 * np.log10((255.0**2) / np.maximum(mse_255, 1e-12)),
    )


def summarize_image_metrics(metrics, seed):
    hard = np.asarray(metrics["hard_mse_255"], dtype=np.float64)
    soft = np.asarray(metrics["soft_mse_255"], dtype=np.float64)
    crc = np.asarray(metrics["crc_valid"], dtype=bool)
    bit_errors = np.asarray(metrics["bit_errors"], dtype=np.int64)
    return {
        "blocks": int(len(hard)),
        "primary_hard_mse_01": describe(hard / (255.0**2), seed + 1),
        "hard_mse_255": describe(hard, seed + 2),
        "soft_posterior_mean_mse_01": describe(
            soft / (255.0**2), seed + 3
        ),
        "soft_posterior_mean_mse_255": describe(soft, seed + 4),
        "hard_psnr_db": describe(psnr_from_mse(hard), seed + 5),
        "crc_failures": int(np.sum(~crc)),
        "crc_bler": float(np.mean(~crc)),
        "exact_image_rate": float(np.mean(bit_errors == 0)),
        "payload_ber": float(np.sum(bit_errors) / (len(bit_errors) * K_PAYLOAD)),
        "per_block": {
            "hard_mse_255": hard.astype(float).tolist(),
            "soft_mse_255": soft.astype(float).tolist(),
            "crc_valid": crc.tolist(),
            "bit_errors": bit_errors.astype(int).tolist(),
        },
    }


def config_id(config):
    scheme = config["scheme"]
    if scheme == "legacy":
        strength = f"a{config['strength']:g}"
    elif scheme == "ep":
        strength = f"aep{config['strength']:g}"
    else:
        strength = f"d{config['strength']:g}_r{config['rho']:g}"
    sigma = (
        f"sfixed{config['sigma_start']:g}"
        if config["sigma_mode"] == "fixed"
        else f"sgeom{config['sigma_start']:g}to{config['sigma_end']:g}"
    )
    return f"{scheme}_{strength}_{sigma}_sp{config['sigma_post']:g}"


def make_config(
    scheme,
    strength,
    *,
    rho=0.9,
    sigma_mode="geom",
    sigma_start=0.3,
    sigma_end=0.05,
    sigma_post=3.0,
):
    config = {
        "scheme": str(scheme),
        "strength": float(strength),
        "rho": float(rho),
        "sigma_mode": str(sigma_mode),
        "sigma_start": float(sigma_start),
        "sigma_end": float(sigma_end),
        "sigma_post": float(sigma_post),
    }
    config["name"] = config_id(config)
    return config


def default_config(scheme, strength, rho=0.9):
    return make_config(scheme, strength, rho=rho)


def scheme_strength_grid():
    configs = [default_config("legacy", 0.0)]
    configs += [
        default_config("legacy", value)
        for value in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
    ]
    configs += [
        default_config("ep", value)
        for value in (0.005, 0.01, 0.02, 0.05, 0.1)
    ]
    configs += [
        default_config("altproj", delta, rho)
        for delta in (0.01, 0.02, 0.05, 0.1, 0.2)
        for rho in (0.85, 0.9, 0.95, 1.0)
    ]
    return configs


def sigma_grid(base):
    configs = []
    for value in (0.02, 0.05, 0.1, 0.3):
        configs.append(
            make_config(
                base["scheme"], base["strength"], rho=base["rho"],
                sigma_mode="fixed", sigma_start=value, sigma_end=value,
            )
        )
    for end in (0.01, 0.02, 0.05, 0.1, 0.2):
        configs.append(
            make_config(
                base["scheme"], base["strength"], rho=base["rho"],
                sigma_mode="geom", sigma_start=0.3, sigma_end=end,
            )
        )
    configs.append(
        make_config(
            base["scheme"], base["strength"], rho=base["rho"],
            sigma_mode="geom", sigma_start=0.5, sigma_end=0.05,
        )
    )
    return unique_configs(configs)


def sigma_post_grid(base):
    return [
        make_config(
            base["scheme"], base["strength"], rho=base["rho"],
            sigma_mode=base["sigma_mode"], sigma_start=base["sigma_start"],
            sigma_end=base["sigma_end"], sigma_post=value,
        )
        for value in (0.75, 1.5, 3.0, 6.0, 12.0, 24.0)
    ]


def unique_configs(configs):
    return list({config["name"]: config for config in configs}.values())


def candidate_for_decoder(config):
    if config["scheme"] == "legacy":
        return {
            "name": config["name"],
            "scheme": "legacy",
            "alpha": config["strength"],
        }
    if config["scheme"] == "ep":
        return {
            "name": config["name"],
            "scheme": "ep",
            "alpha_ep": config["strength"],
        }
    return {
        "name": config["name"],
        "scheme": "altproj",
        "delta": config["strength"],
        "rho": config["rho"],
    }


def choose_best(rows):
    return min(
        rows,
        key=lambda name: (
            rows[name]["primary_hard_mse_01"]["mean"],
            rows[name]["soft_posterior_mean_mse_01"]["mean"],
            rows[name]["crc_bler"],
        ),
    )


def run_source(args):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import torch

    from crc_utils import hard_crc_decode
    from denoiser_sigma_alignment_diag import load_fashion_mnist
    from denoiser_sigma_conditional_diag import make_system
    from denoiser_sigma_three_scheme import configure_candidate
    from low_budget_source_study import build_experiment_decoder
    from syndrome_sigma_schedule import AnnealingSigmaScheduler

    images, bit_bank = load_fashion_mnist()
    crc_encoder, crc_decoder, ldpc, mapper, demapper, _ = make_system()
    decoder, proxy = build_experiment_decoder(ldpc, SCHEDULE, args.checkpoint)
    decoder._ep_config_logged = True
    decoder.track_u_hat = False

    def channel_batch(round_index, batch, seed, esn0_db):
        indices = tf.random.stateless_uniform(
            [batch], seed=[seed, round_index], minval=0,
            maxval=args.paired_pool, dtype=tf.int32,
        )
        payload = tf.gather(bit_bank, indices)
        noise_variance = tf.cast(10.0 ** (-esn0_db / 10.0), ldpc.rdtype)
        codeword = ldpc(crc_encoder(tf.cast(payload, ldpc.rdtype)))
        transmitted = mapper(codeword)
        if transmitted.dtype.is_complex:
            real_dtype = transmitted.dtype.real_dtype
            noise_real = tf.random.stateless_normal(
                tf.shape(transmitted), seed=[seed + 1, round_index],
                dtype=real_dtype,
            )
            noise_imag = tf.random.stateless_normal(
                tf.shape(transmitted), seed=[seed + 2, round_index],
                dtype=real_dtype,
            )
            unit_noise = tf.complex(noise_real, noise_imag) / tf.cast(
                math.sqrt(2.0), transmitted.dtype
            )
        else:
            unit_noise = tf.random.stateless_normal(
                tf.shape(transmitted), seed=[seed + 1, round_index],
                dtype=transmitted.dtype,
            )
        received = transmitted + unit_noise * tf.cast(
            tf.sqrt(noise_variance), transmitted.dtype
        )
        llr = demapper(received, noise_variance)
        truth = images[indices.numpy()].astype(np.float64)
        return indices.numpy(), payload.numpy().astype(bool), truth, llr

    def sigma_objects(config):
        if config["sigma_mode"] == "fixed":
            start = end = config["sigma_start"]
        else:
            start, end = config["sigma_start"], config["sigma_end"]
        scheduler = AnnealingSigmaScheduler(
            start, end, SOURCE_CALLS, mode="geom"
        )
        path = tuple(float(value) for value in scheduler.path)
        if len(path) != SOURCE_CALLS:
            raise RuntimeError(f"sigma path length {len(path)} != {SOURCE_CALLS}")
        return scheduler, path

    def run_configs(configs, esn0_db, blocks, seed, tag):
        if blocks % args.batch:
            raise ValueError("blocks must be divisible by batch")
        configs = unique_configs(configs)
        accum = {
            config["name"]: {
                "hard_mse_255": [], "soft_mse_255": [],
                "crc_valid": [], "bit_errors": [],
            }
            for config in configs
        }
        for round_index in range(blocks // args.batch):
            _, truth_bits, truth_image, channel_llr = channel_batch(
                round_index, args.batch, seed, esn0_db
            )
            for config in configs:
                scheduler, path = sigma_objects(config)
                configure_candidate(
                    decoder, proxy, scheduler, path,
                    candidate_for_decoder(config),
                )
                proxy.set_readout("global", config["sigma_post"])
                decoder.altproj_early_stop = False
                logits = decoder(channel_llr)
                payload_logits = np.asarray(logits.numpy())[:, :K_PAYLOAD]
                shaped = payload_logits.reshape(-1, NPIX, BPP)
                hard_bits = shaped > 0.0
                probabilities = 1.0 / (
                    1.0 + np.exp(-np.clip(shaped, -60.0, 60.0))
                )
                hard_image = np.sum(hard_bits * BIT_WEIGHTS, axis=2)
                soft_image = np.sum(probabilities * BIT_WEIGHTS, axis=2)
                truth_flat = truth_image.reshape(-1, NPIX)
                hard_mse = np.mean((hard_image - truth_flat) ** 2, axis=1)
                soft_mse = np.mean((soft_image - truth_flat) ** 2, axis=1)
                bit_errors = np.sum(
                    hard_bits.reshape(-1, K_PAYLOAD) != truth_bits,
                    axis=1,
                )
                _, valid = hard_crc_decode(crc_decoder, logits)
                bucket = accum[config["name"]]
                bucket["hard_mse_255"].extend(hard_mse.tolist())
                bucket["soft_mse_255"].extend(soft_mse.tolist())
                bucket["crc_valid"].extend(
                    np.asarray(valid.numpy()).reshape(-1).astype(bool).tolist()
                )
                bucket["bit_errors"].extend(bit_errors.astype(int).tolist())
            print(
                f"{tag}: Es/N0={esn0_db:+.2f} round "
                f"{round_index + 1}/{blocks // args.batch} "
                f"configs={len(configs)}",
                flush=True,
            )
        rows = {}
        for position, config in enumerate(configs):
            row = summarize_image_metrics(accum[config["name"]], seed + position * 17)
            row["config"] = config
            rows[config["name"]] = row
        bp_name = default_config("legacy", 0.0)["name"]
        if bp_name in rows:
            reference = np.asarray(
                rows[bp_name]["per_block"]["hard_mse_255"], dtype=np.float64
            )
            for position, row in enumerate(rows.values()):
                values = np.asarray(row["per_block"]["hard_mse_255"])
                delta = (values - reference) / (255.0**2)
                row["paired_hard_mse_delta_vs_bp"] = {
                    "mean": float(np.mean(delta)),
                    "bootstrap_mean_95": bootstrap_mean_ci(
                        delta, seed + 5000 + position
                    ),
                    "improved_blocks": int(np.sum(delta < 0)),
                    "degraded_blocks": int(np.sum(delta > 0)),
                    "tied_blocks": int(np.sum(delta == 0)),
                }
        return rows

    result = {
        "kind": "bp50_mse_parameter_estimate",
        "configuration": {
            "branch": "codex/mse-optimization",
            "channel": "BPSK/AWGN/perfect_CSI",
            "payload_bits": K_PAYLOAD,
            "n": N_CODEWORD,
            "bp_schedule": SCHEDULE,
            "bp_budget": sum(SCHEDULE),
            "source_calls": SOURCE_CALLS,
            "screen_blocks": args.screen_blocks,
            "validation_blocks": args.validation_blocks,
            "batch": args.batch,
            "screen_esn0_db": args.screen_esn0,
            "validation_esn0_db": args.validation_esn0,
            "screen_seed": args.seed,
            "validation_seed_base": args.validation_seed,
            "paired_pool": args.paired_pool,
            "primary_objective": "hard 8-bit pixel MSE normalized by 255^2",
            "secondary_objective": "soft posterior-mean pixel MSE",
            "legacy_beta": 0.0,
            "ep_beta_ep": 1.0,
            "fixed_iteration_count": True,
            "crc_early_stop": False,
            "selection_warning": "small-MC estimate; not a final claim",
        },
        "stages": {},
        "validation": {},
    }

    stage1_configs = scheme_strength_grid()
    stage1 = run_configs(
        stage1_configs, args.screen_esn0, args.screen_blocks, args.seed,
        "stage1-scheme-strength",
    )
    best1_name = choose_best(stage1)
    best1 = stage1[best1_name]["config"]
    result["stages"]["scheme_strength"] = {
        "rows": stage1, "selected": best1_name,
    }
    dump_json(args.output, result)

    stage2_configs = sigma_grid(best1) + [default_config("legacy", 0.0)]
    stage2 = run_configs(
        stage2_configs, args.screen_esn0, args.screen_blocks, args.seed,
        "stage2-sigma",
    )
    source_stage2 = {k: v for k, v in stage2.items() if v["config"]["strength"] != 0.0}
    best2_name = choose_best(source_stage2)
    best2 = stage2[best2_name]["config"]
    result["stages"]["sigma"] = {"rows": stage2, "selected": best2_name}
    dump_json(args.output, result)

    stage3_configs = sigma_post_grid(best2) + [default_config("legacy", 0.0)]
    stage3 = run_configs(
        stage3_configs, args.screen_esn0, args.screen_blocks, args.seed,
        "stage3-sigma-post",
    )
    source_stage3 = {k: v for k, v in stage3.items() if v["config"]["strength"] != 0.0}
    best3_name = choose_best(source_stage3)
    best3 = stage3[best3_name]["config"]
    result["stages"]["sigma_post"] = {"rows": stage3, "selected": best3_name}
    dump_json(args.output, result)

    winning_scheme = best1["scheme"]
    strength_rows = [
        row for row in stage1.values()
        if row["config"]["scheme"] == winning_scheme
        and row["config"]["strength"] > 0.0
    ]
    top_strength = sorted(
        strength_rows,
        key=lambda row: row["primary_hard_mse_01"]["mean"],
    )[:3]
    sigma_rows = [
        row for row in stage2.values() if row["config"]["strength"] > 0.0
    ]
    top_sigma = sorted(
        sigma_rows,
        key=lambda row: row["primary_hard_mse_01"]["mean"],
    )[:3]
    sp_rows = [
        row for row in stage3.values() if row["config"]["strength"] > 0.0
    ]
    top_sp = sorted(
        sp_rows,
        key=lambda row: row["primary_hard_mse_01"]["mean"],
    )[:3]
    interaction = []
    for strength_row in top_strength:
        for sigma_row in top_sigma:
            for sp_row in top_sp:
                s = strength_row["config"]
                q = sigma_row["config"]
                p = sp_row["config"]
                interaction.append(
                    make_config(
                        winning_scheme, s["strength"], rho=s["rho"],
                        sigma_mode=q["sigma_mode"],
                        sigma_start=q["sigma_start"], sigma_end=q["sigma_end"],
                        sigma_post=p["sigma_post"],
                    )
                )
    interaction += [best3, default_config("legacy", 0.0)]
    stage4 = run_configs(
        unique_configs(interaction), args.screen_esn0, args.screen_blocks,
        args.seed, "stage4-interaction",
    )
    source_stage4 = {k: v for k, v in stage4.items() if v["config"]["strength"] != 0.0}
    best4_name = choose_best(source_stage4)
    result["stages"]["interaction"] = {
        "rows": stage4,
        "selected": best4_name,
        "construction": "top-3 strength x top-3 sigma x top-3 sigma_post",
    }
    dump_json(args.output, result)

    ranked = sorted(
        source_stage4.values(),
        key=lambda row: (
            row["primary_hard_mse_01"]["mean"],
            row["soft_posterior_mean_mse_01"]["mean"],
        ),
    )
    final_configs = unique_configs(
        [row["config"] for row in ranked[:3]] + [default_config("legacy", 0.0)]
    )
    for snr_index, snr in enumerate(args.validation_esn0):
        point_seed = args.validation_seed + snr_index * 10000
        rows = run_configs(
            final_configs, snr, args.validation_blocks, point_seed,
            "validation",
        )
        result["validation"][f"{snr:.3f}"] = {
            "seed": point_seed,
            "rows": rows,
        }
        dump_json(args.output, result)
    result["selection"] = {
        "screen_selected": best4_name,
        "validation_candidates": [config["name"] for config in final_configs],
        "status": "estimate only; independent larger-MC confirmation required",
    }
    dump_json(args.output, result)


def decode_webp_bytes(raw):
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(raw)) as image:
            decoded = np.asarray(image.convert("L"), dtype=np.uint8)
        if decoded.shape != (IMG_H, IMG_W):
            return None
        return decoded
    except (OSError, ValueError, UnidentifiedImageError):
        return None


def summarize_webp(metric, crc_valid, decoded, exact, seed):
    mse = np.asarray(metric, dtype=np.float64)
    return {
        "blocks": int(len(mse)),
        "mse_01": describe(mse / (255.0**2), seed),
        "mse_255": describe(mse, seed + 1),
        "psnr_db": describe(psnr_from_mse(mse), seed + 2),
        "crc_failures": int(np.sum(~np.asarray(crc_valid, dtype=bool))),
        "crc_bler": float(np.mean(~np.asarray(crc_valid, dtype=bool))),
        "webp_decode_rate": float(np.mean(decoded)),
        "exact_image_rate": float(np.mean(exact)),
        "per_block_mse_255": mse.tolist(),
    }


def run_webp(args):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    from crc_utils import hard_crc_decode
    from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
    from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
    from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
    from sionna.phy.mapping import Demapper, Mapper

    source = json.loads(Path(args.source_result).read_text(encoding="utf-8"))
    archive = np.load(args.webp_streams)
    bits = archive["bits"]
    lengths = archive["lengths"]
    images = archive["imgs"].reshape(-1, IMG_H, IMG_W).astype(np.float64)
    if args.paired_pool > len(bits):
        raise ValueError("paired pool exceeds available WebP streams")
    k_container = int(bits.shape[1])
    if k_container != int(np.max(lengths)):
        raise RuntimeError("WebP container is not MAX padded")
    mean_fallback = images[:args.paired_pool].mean(axis=0)
    zero_fallback = np.zeros((IMG_H, IMG_W), dtype=np.float64)

    crc_encoder = CRCEncoder("CRC24A")
    crc_decoder = CRCDecoder(crc_encoder)
    encoder = LDPC5GEncoder(k_container + 24, N_CODEWORD, num_bits_per_symbol=1)
    decoder = LDPC5GDecoder(
        encoder,
        cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
        hard_out=False, return_infobits=True, num_iter=10, llr_max=30.0,
        return_state=True,
    )
    mapper = Mapper("pam", num_bits_per_symbol=1)
    demapper = Demapper("app", "pam", num_bits_per_symbol=1)

    points = {}
    for snr_key, source_point in source["validation"].items():
        snr = float(snr_key)
        seed = int(source_point["seed"])
        blocks = int(source["configuration"]["validation_blocks"])
        if blocks % args.batch:
            raise ValueError("blocks must be divisible by batch")
        metrics = {
            "best_effort_mean": [], "best_effort_zero": [],
            "crc_gated_mean": [], "crc_gated_zero": [],
            "crc_valid": [], "decoded": [], "exact": [],
        }
        for round_index in range(blocks // args.batch):
            indices = tf.random.stateless_uniform(
                [args.batch], seed=[seed, round_index], minval=0,
                maxval=args.paired_pool, dtype=tf.int32,
            ).numpy()
            payload_np = bits[indices, :k_container].astype(np.float32)
            truth = images[indices]
            payload = tf.constant(payload_np, dtype=encoder.rdtype)
            noise_variance = tf.cast(10.0 ** (-snr / 10.0), encoder.rdtype)
            codeword = encoder(crc_encoder(payload))
            transmitted = mapper(codeword)
            if transmitted.dtype.is_complex:
                real_dtype = transmitted.dtype.real_dtype
                noise_real = tf.random.stateless_normal(
                    tf.shape(transmitted), seed=[seed + 1, round_index],
                    dtype=real_dtype,
                )
                noise_imag = tf.random.stateless_normal(
                    tf.shape(transmitted), seed=[seed + 2, round_index],
                    dtype=real_dtype,
                )
                unit_noise = tf.complex(noise_real, noise_imag) / tf.cast(
                    math.sqrt(2.0), transmitted.dtype
                )
            else:
                unit_noise = tf.random.stateless_normal(
                    tf.shape(transmitted), seed=[seed + 1, round_index],
                    dtype=transmitted.dtype,
                )
            received = transmitted + unit_noise * tf.cast(
                tf.sqrt(noise_variance), transmitted.dtype
            )
            llr = demapper(received, noise_variance)
            message = None
            logits = None
            for _ in range(5):
                logits, message = decoder(llr, num_iter=10, msg_v2c=message)
            _, valid_tensor = hard_crc_decode(crc_decoder, logits)
            valid = np.asarray(valid_tensor.numpy()).reshape(-1).astype(bool)
            recovered = np.asarray(logits.numpy())[:, :k_container] > 0.0
            for local in range(args.batch):
                nbytes = int(lengths[indices[local]] // 8)
                raw = np.packbits(recovered[local])[:nbytes].tobytes()
                decoded = decode_webp_bytes(raw)
                decode_ok = decoded is not None
                exact = bool(decode_ok and np.array_equal(decoded, truth[local]))
                best_mean = decoded if decode_ok else mean_fallback
                best_zero = decoded if decode_ok else zero_fallback
                gated_mean = decoded if valid[local] and decode_ok else mean_fallback
                gated_zero = decoded if valid[local] and decode_ok else zero_fallback
                for key, value in (
                    ("best_effort_mean", best_mean),
                    ("best_effort_zero", best_zero),
                    ("crc_gated_mean", gated_mean),
                    ("crc_gated_zero", gated_zero),
                ):
                    metrics[key].append(float(np.mean((value - truth[local]) ** 2)))
                metrics["crc_valid"].append(bool(valid[local]))
                metrics["decoded"].append(bool(decode_ok))
                metrics["exact"].append(exact)
            print(
                f"WebP: Es/N0={snr:+.2f} round "
                f"{round_index + 1}/{blocks // args.batch}", flush=True,
            )
        policies = {}
        for offset, key in enumerate(
            ("best_effort_mean", "best_effort_zero", "crc_gated_mean", "crc_gated_zero")
        ):
            policies[key] = summarize_webp(
                metrics[key], metrics["crc_valid"], metrics["decoded"],
                metrics["exact"], seed + offset * 101,
            )
        points[snr_key] = {"seed": seed, "policies": policies}

    result = {
        "kind": "webp_bp50_mse_baseline",
        "configuration": {
            "source_result": args.source_result,
            "webp_streams": args.webp_streams,
            "container_bits": k_container,
            "ldpc_k": k_container + 24,
            "ldpc_rate": (k_container + 24) / N_CODEWORD,
            "n": N_CODEWORD,
            "bp_schedule": SCHEDULE,
            "bp_budget": sum(SCHEDULE),
            "fixed_iteration_count": True,
            "crc_early_stop": False,
            "paired_payload_indices_and_unit_noise": True,
            "primary_policy": (
                "best_effort WebP decode; undecodable streams use the empirical "
                "per-pixel mean image over the paired pool (optimistic fallback)"
            ),
            "fallback_sensitivity": ["mean_image", "zero_image"],
            "mse_note": (
                "lossless WebP gives zero MSE when the stream is exact; CRC "
                "failure has no intrinsic image, so fallback policy is explicit"
            ),
        },
        "points": points,
    }
    dump_json(args.output, result)


def run_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = json.loads(Path(args.source_result).read_text(encoding="utf-8"))
    webp = json.loads(Path(args.webp_result).read_text(encoding="utf-8"))

    stages = source["stages"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), constrained_layout=True)
    stage_specs = [
        ("scheme_strength", "Scheme / injection strength"),
        ("sigma", "Denoiser sigma path"),
        ("sigma_post", "Pixel-to-bit sigma_post"),
        ("interaction", "Top-neighborhood interaction"),
    ]
    for ax, (key, title) in zip(axes.reshape(-1), stage_specs):
        rows = stages[key]["rows"]
        ordered = sorted(
            rows.items(), key=lambda item: item[1]["primary_hard_mse_01"]["mean"]
        )
        show = ordered[: min(12, len(ordered))]
        names = [name for name, _ in show][::-1]
        means = [row["primary_hard_mse_01"]["mean"] for _, row in show][::-1]
        lo = [
            row["primary_hard_mse_01"]["bootstrap_mean_95"][0]
            for _, row in show
        ][::-1]
        hi = [
            row["primary_hard_mse_01"]["bootstrap_mean_95"][1]
            for _, row in show
        ][::-1]
        y = np.arange(len(names))
        ax.barh(y, means, color="#33658A", alpha=0.85)
        ax.errorbar(
            means, y,
            xerr=[np.asarray(means) - np.asarray(lo), np.asarray(hi) - np.asarray(means)],
            fmt="none", ecolor="black", capsize=2, lw=0.8,
        )
        ax.set_yticks(y, [name.replace("_", " ") for name in names], fontsize=7)
        ax.set_xlabel("hard image MSE / 255²")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("BP-50 [10]×5 MSE screening (32 blocks, estimate only)", fontsize=14)
    fig.savefig(args.sweep_plot, dpi=180)
    plt.close(fig)

    validation_names = source["selection"]["validation_candidates"]
    bp_name = default_config("legacy", 0.0)["name"]
    snrs = sorted(float(key) for key in source["validation"])
    fig, ax = plt.subplots(figsize=(9.2, 5.7), constrained_layout=True)
    colors = ["#D1495B", "#EDAE49", "#30638E"]
    for index, name in enumerate([n for n in validation_names if n != bp_name]):
        means, lo, hi = [], [], []
        for snr in snrs:
            row = source["validation"][f"{snr:.3f}"]["rows"][name]
            stat = row["primary_hard_mse_01"]
            means.append(stat["mean"])
            lo.append(stat["bootstrap_mean_95"][0])
            hi.append(stat["bootstrap_mean_95"][1])
        ax.plot(snrs, means, "o-", color=colors[index], label=f"ours: {name}")
        ax.fill_between(snrs, lo, hi, color=colors[index], alpha=0.12)
    if bp_name in validation_names:
        means = [
            source["validation"][f"{snr:.3f}"]["rows"][bp_name]
            ["primary_hard_mse_01"]["mean"]
            for snr in snrs
        ]
        ax.plot(snrs, means, "D--", color="#555555", label="raw + BP-50")

    for policy, style, label in (
        ("best_effort_mean", "s-", "WebP + BP-50 (best effort, mean fallback)"),
        ("crc_gated_mean", "^:", "WebP + BP-50 (CRC-gated, mean fallback)"),
    ):
        means, lo, hi = [], [], []
        for snr in snrs:
            stat = webp["points"][f"{snr:.3f}"]["policies"][policy]["mse_01"]
            means.append(stat["mean"])
            lo.append(stat["bootstrap_mean_95"][0])
            hi.append(stat["bootstrap_mean_95"][1])
        # A lossless codec can yield exactly zero sample MSE when every tested
        # stream is recovered.  Log axes cannot display zero, so put those
        # measured-zero markers on an explicit plotting floor and annotate them.
        display_floor = 1.0e-6
        shown_means = np.maximum(means, display_floor)
        shown_lo = np.maximum(lo, display_floor)
        shown_hi = np.maximum(hi, display_floor)
        ax.plot(snrs, shown_means, style, color="#00798C", label=label)
        ax.fill_between(snrs, shown_lo, shown_hi, color="#00798C", alpha=0.08)
        for snr, measured, shown in zip(snrs, means, shown_means):
            if measured == 0.0:
                ax.annotate(
                    "measured MSE=0\n(64/64 exact)",
                    (snr, shown), xytext=(-78, 16), textcoords="offset points",
                    fontsize=7, color="#00798C",
                    arrowprops=dict(arrowstyle="->", color="#00798C", lw=0.8),
                )
    ax.set_yscale("log")
    ax.set_ylim(bottom=5.0e-7)
    ax.set_xlabel("Es/N0 (dB)")
    ax.set_ylabel("mean image MSE / 255²")
    ax.set_title("MSE objective: raw+source versus lossless WebP, BP budget 50")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5)
    fig.savefig(
        args.comparison_plot, dpi=180, bbox_inches="tight", pad_inches=0.16
    )
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    source = sub.add_parser("source")
    source.add_argument("--screen-blocks", type=int, default=32)
    source.add_argument("--validation-blocks", type=int, default=64)
    source.add_argument("--batch", type=int, default=32)
    source.add_argument("--screen-esn0", type=float, default=-2.85)
    source.add_argument(
        "--validation-esn0", type=float, nargs="+", default=[-3.0, -2.85, -2.7]
    )
    source.add_argument("--seed", type=int, default=20260901)
    source.add_argument("--validation-seed", type=int, default=20260911)
    source.add_argument("--paired-pool", type=int, default=3200)
    source.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    source.add_argument("--output", required=True)

    webp = sub.add_parser("webp")
    webp.add_argument("--source-result", required=True)
    webp.add_argument("--webp-streams", default=DEFAULT_WEBP)
    webp.add_argument("--paired-pool", type=int, default=3200)
    webp.add_argument("--batch", type=int, default=32)
    webp.add_argument("--output", required=True)

    plot = sub.add_parser("plot")
    plot.add_argument("--source-result", required=True)
    plot.add_argument("--webp-result", required=True)
    plot.add_argument("--sweep-plot", required=True)
    plot.add_argument("--comparison-plot", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "source":
        run_source(args)
    elif args.mode == "webp":
        run_webp(args)
    elif args.mode == "plot":
        run_plot(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
