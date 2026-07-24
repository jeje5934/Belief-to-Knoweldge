"""Altproj waterfall and compute-budget study on the compression Es/N0 axis.

Experiment harness only: production ``decoder.py`` is imported unchanged.
The first mode implemented here is the canonical altproj waterfall used as the
pre-SPC anchor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import torch

from decoder import LDPC5GDecoder_soft
from denoiser_sigma_alignment_diag import (
    FIXED_SIGMA,
    K_PAYLOAD,
    N_CODEWORD,
    load_fashion_mnist,
)
from denoiser_sigma_conditional_diag import (
    make_channel_batch,
    make_system,
    wilson,
)
from denoiser_sigma_three_scheme import (
    DEFAULT_CHECKPOINT,
    EP_BETA_CODE_SITE,
    LEGACY_BETA_BP_EXTRINSIC_BOOST,
    SIGMA_POST,
    build_decoder,
    configure_candidate,
    dump_json,
)
from syndrome_sigma_schedule import AnnealingSigmaScheduler
from ldpc_altproj_spc import (
    BITS_PER_PIXEL,
    PAYLOAD_BITS,
    PIXELS,
    SPC_BITS,
    SPCScoreExtrinsic,
    candidate_bits_numpy,
    decode_spc_systematic_numpy,
    encode_spc_numpy,
    parity_valid_numpy,
    spc_marginalize_torch,
)
from sionna.phy.channel.awgn import AWGN
from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.mapping import Demapper, Mapper


COMPRESSION_GRID = (-2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0)


def geom_scheduler(schedule, endpoint):
    scheduler = AnnealingSigmaScheduler(
        FIXED_SIGMA, float(endpoint), len(schedule) - 1, mode="geom"
    )
    return scheduler, tuple(float(v) for v in scheduler.path)


def altproj_candidate(endpoint, delta=0.02, rho=0.9):
    return {
        "name": f"altproj_d{delta:g}_r{rho:g}_s{endpoint:g}",
        "scheme": "altproj",
        "sigma_end": float(endpoint),
        "delta": float(delta),
        "rho": float(rho),
    }


def make_spc_system():
    crc_encoder = CRCEncoder("CRC24A")
    crc_decoder = CRCDecoder(crc_encoder)
    ldpc = LDPC5GEncoder(
        SPC_BITS + crc_encoder.crc_length,
        N_CODEWORD,
        num_bits_per_symbol=1,
    )
    mapper = Mapper("pam", num_bits_per_symbol=1)
    demapper = Demapper("app", "pam", num_bits_per_symbol=1)
    return crc_encoder, crc_decoder, ldpc, mapper, demapper, AWGN()


def build_spc_altproj_decoder(
    ldpc,
    checkpoint,
    schedule,
    sigma_end,
    delta,
    rho,
    device,
):
    scheduler, path = geom_scheduler(schedule, sigma_end)
    common = dict(
        cn_update="boxplus-phi",
        vn_update="sum",
        cn_schedule="flooding",
        hard_out=False,
        return_infobits=True,
        llr_max=30.0,
    )
    decoder = LDPC5GDecoder_soft(
        ldpc,
        k_payload=SPC_BITS,
        num_iter=sum(schedule),
        bp_schedule=schedule,
        alpha=0.1,
        beta=0.0,
        ep_mode=False,
        altproj=True,
        altproj_delta=float(delta),
        altproj_rho=float(rho),
        altproj_sigma_den=FIXED_SIGMA,
        altproj_warm_start=True,
        altproj_early_stop=False,
        sigma_scheduler=scheduler,
        adaptive_sigma=True,
        denoiser_kwargs=dict(device=device),
        **common,
    )
    decoder.denoiser.load_weights_pt(checkpoint)
    decoder.denoiser.sigma = FIXED_SIGMA
    decoder.denoiser.sigma_post = SIGMA_POST
    adapter = SPCScoreExtrinsic(decoder.denoiser, sigma_path=path)
    decoder._denoiser = adapter
    decoder._ep_config_logged = True
    return decoder, adapter, path


def make_spc_channel_batch(
    round_idx,
    batch,
    seed,
    esn0_db,
    images,
    bit_bank,
    crc_encoder,
    ldpc,
    mapper,
    demapper,
):
    indices = tf.random.stateless_uniform(
        [batch],
        seed=tf.constant([seed, round_idx], dtype=tf.int32),
        minval=0,
        maxval=tf.shape(bit_bank)[0],
        dtype=tf.int32,
    )
    payload = tf.gather(bit_bank, indices)
    spc_np = encode_spc_numpy(payload.numpy())
    spc_payload = tf.constant(spc_np, dtype=ldpc.rdtype)
    codeword = ldpc(crc_encoder(spc_payload))
    transmitted = mapper(codeword)
    noise_variance = tf.cast(
        10.0 ** (-float(esn0_db) / 10.0), ldpc.rdtype
    )
    if transmitted.dtype.is_complex:
        real_dtype = transmitted.dtype.real_dtype
        noise_real = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=tf.constant([seed + 1, round_idx], dtype=tf.int32),
            dtype=real_dtype,
        )
        noise_imag = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=tf.constant([seed + 2, round_idx], dtype=tf.int32),
            dtype=real_dtype,
        )
        unit_noise = tf.complex(noise_real, noise_imag) / tf.cast(
            math.sqrt(2.0), transmitted.dtype
        )
    else:
        unit_noise = tf.random.stateless_normal(
            tf.shape(transmitted),
            seed=tf.constant([seed + 1, round_idx], dtype=tf.int32),
            dtype=transmitted.dtype,
        )
    received = transmitted + unit_noise * tf.cast(
        tf.sqrt(noise_variance), transmitted.dtype
    )
    return (
        indices.numpy(),
        payload,
        spc_payload,
        codeword,
        demapper(received, noise_variance),
    )


def linear_knee(points, threshold=0.1):
    """Match compression_baseline/plot_channel.py's linear BLER interpolation."""
    ordered = sorted(points, key=lambda row: row["esn0_db"], reverse=True)
    for left, right in zip(ordered, ordered[1:]):
        if left["crc_bler"] < threshold <= right["crc_bler"]:
            fraction = (
                (threshold - left["crc_bler"])
                / (right["crc_bler"] - left["crc_bler"])
            )
            return float(
                left["esn0_db"]
                + fraction * (right["esn0_db"] - left["esn0_db"])
            )
    return None


def run_waterfall(args):
    if len(args.esn0) != len(args.blocks):
        raise ValueError("esn0 and blocks must have equal lengths")
    if any(blocks % args.batch for blocks in args.blocks):
        raise ValueError("each block allocation must be divisible by batch")

    schedule = [5] * 20
    (
        crc_encoder,
        crc_decoder,
        ldpc,
        mapper,
        demapper,
        awgn,
    ) = make_system()
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, proxy, _, _ = build_decoder(
        ldpc, args.checkpoint, schedule, device
    )
    decoder._ep_config_logged = True
    candidate = altproj_candidate(args.sigma_end, args.delta, args.rho)
    scheduler, path = geom_scheduler(schedule, args.sigma_end)

    result = {
        "kind": "altproj_compression_axis_waterfall",
        "configuration": {
            "branch": "practical_sigma",
            "payload_bits": 6272,
            "crc": "CRC24A",
            "ldpc_k": 6296,
            "n": 12600,
            "payload_rate": 6272 / 12600,
            "ldpc_rate": 6296 / 12600,
            "channel": "BPSK/AWGN/perfect_CSI",
            "axis": "Es/N0",
            "bp_schedule": schedule,
            "nominal_bp_budget": sum(schedule),
            "sigma_start": FIXED_SIGMA,
            "sigma_end": args.sigma_end,
            "sigma_mode": "geom",
            "sigma_post": SIGMA_POST,
            "delta": args.delta,
            "rho": args.rho,
            "crc_early_stop": False,
            "legacy_beta_contract": LEGACY_BETA_BP_EXTRINSIC_BOOST,
            "ep_beta_contract": EP_BETA_CODE_SITE,
            "batch": args.batch,
            "seed": args.seed,
            "knee_interpolation": (
                "linear in BLER, identical to compression_baseline/plot_channel.py"
            ),
        },
        "sigma_path": list(path),
        "points": {},
        "knee_bler_0p1_db": None,
        "pixelcnn_max_knee_db": -3.79,
        "gap_to_pixelcnn_max_db": None,
    }

    for point_index, (esn0_db, blocks) in enumerate(
        zip(args.esn0, args.blocks)
    ):
        failures = 0
        actual_iters = []
        rounds = blocks // args.batch
        for round_idx in range(rounds):
            _, _, _, channel_llr = make_channel_batch(
                round_idx,
                args.batch,
                args.seed + point_index * 10000,
                esn0_db,
                images,
                bit_bank,
                crc_encoder,
                ldpc,
                mapper,
                demapper,
                awgn,
            )
            configure_candidate(
                decoder, proxy, scheduler, path, candidate
            )
            decoder.altproj_early_stop = False
            final_logits = decoder(channel_llr)
            _, crc_valid = crc_decoder(final_logits)
            failures += (
                args.batch
                - int(
                    tf.reduce_sum(
                        tf.cast(crc_valid, tf.int32)
                    ).numpy()
                )
            )
            actual_iters.append(
                float(
                    sum(
                        row["bp_iters_used"]
                        for row in decoder.last_bp_stats
                    )
                )
            )
            if proxy.call_index != len(path):
                raise RuntimeError(
                    f"altproj source calls={proxy.call_index}, expected={len(path)}"
                )
            print(
                f"Es/N0={esn0_db:+.2f} round {round_idx + 1}/{rounds}",
                flush=True,
            )
        key = f"{esn0_db:.3f}"
        result["points"][key] = {
            "esn0_db": float(esn0_db),
            "blocks": int(blocks),
            "failures": int(failures),
            "crc_bler": float(failures / blocks),
            "wilson_95": wilson(failures, blocks),
            "nominal_bp_iters": 100,
            "actual_mean_bp_iters": float(np.mean(actual_iters)),
        }
        rows = list(result["points"].values())
        knee = linear_knee(rows)
        result["knee_bler_0p1_db"] = knee
        result["gap_to_pixelcnn_max_db"] = (
            None if knee is None else float(knee - (-3.79))
        )
        dump_json(args.output, result)
        print(f"saved {key} to {args.output}", flush=True)


def _numpy_spc_reference(log_categorical, parity_llr):
    bits = candidate_bits_numpy().astype(bool)
    parity = (bits.sum(axis=-1) % 2).astype(bool)
    log_categorical = np.asarray(log_categorical, dtype=np.float64)
    shifted = log_categorical - np.max(
        log_categorical, axis=-1, keepdims=True
    )
    categorical = np.exp(shifted)
    categorical /= np.sum(categorical, axis=-1, keepdims=True)
    p1 = 1.0 / (1.0 + np.exp(-np.asarray(parity_llr, dtype=np.float64)))
    parity_likelihood = np.where(
        parity, p1[..., None], (1.0 - p1)[..., None]
    )
    posterior = categorical * parity_likelihood
    posterior /= np.sum(posterior, axis=-1, keepdims=True)
    apps = []
    for bit_index in range(BITS_PER_PIXEL):
        mask = bits[:, bit_index]
        apps.append(
            np.log(np.sum(posterior[..., mask], axis=-1))
            - np.log(np.sum(posterior[..., ~mask], axis=-1))
        )
    systematic = np.stack(apps, axis=-1)
    parity_app = (
        np.log(np.sum(posterior[..., parity], axis=-1))
        - np.log(np.sum(posterior[..., ~parity], axis=-1))
    )
    return systematic, parity_app


def run_feasibility(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = [5] * 20
    support_error = None
    try:
        system = make_spc_system()
    except Exception as exc:
        support_error = f"{type(exc).__name__}: {exc}"
        system = None
    result = {
        "kind": "ldpc_altproj_spc_feasibility",
        "configuration": {
            "payload_bits": PAYLOAD_BITS,
            "spc_bits": SPC_BITS,
            "crc_bits": 24,
            "ldpc_k": SPC_BITS + 24,
            "ldpc_n": N_CODEWORD,
            "ldpc_rate": (SPC_BITS + 24) / N_CODEWORD,
            "payload_rate": PAYLOAD_BITS / N_CODEWORD,
            "layout": (
                "raster pixels; 8 systematic MSB-first bits then XOR parity; "
                "CRC24A follows the 7056 source positions"
            ),
            "source_correction_positions": (
                "all 7056 systematic+SPC bits; CRC24 positions excluded"
            ),
            "llr_convention": "log P(1)/P(0)",
            "sigma_post": SIGMA_POST,
            "device": device,
        },
        "ldpc_support": {
            "supported": support_error is None,
            "error": support_error,
        },
        "checks": {},
    }
    if system is None:
        dump_json(args.output, result)
        return

    (
        crc_encoder,
        crc_decoder,
        ldpc,
        mapper,
        demapper,
        _,
    ) = system
    result["ldpc_support"]["observed_coderate"] = float(ldpc.coderate)

    # (a) Exhaust every possible 8-bit source symbol and verify its XOR bit.
    exhaustive = candidate_bits_numpy()
    exhaustive_spc = encode_spc_numpy(exhaustive)
    result["checks"]["spc_exhaustive"] = {
        "symbols": int(len(exhaustive)),
        "all_parity_valid": bool(
            np.all(parity_valid_numpy(exhaustive_spc))
        ),
        "roundtrip_exact": bool(
            np.array_equal(
                decode_spc_systematic_numpy(exhaustive_spc), exhaustive
            )
        ),
    }

    # (b) Compare the vectorized 256-way log-domain APP with an independent
    # direct-probability brute-force enumeration.
    rng = np.random.default_rng(args.seed)
    logcat = rng.normal(size=(2, 5, 256)).astype(np.float32)
    parity_llr = rng.normal(size=(2, 5)).astype(np.float32)
    candidate_tensor = torch.as_tensor(
        candidate_bits_numpy(), dtype=torch.int64, device=device
    )
    with torch.no_grad():
        sys_t, par_t, _ = spc_marginalize_torch(
            torch.as_tensor(logcat, device=device),
            torch.as_tensor(parity_llr, device=device),
            candidate_tensor,
        )
    sys_ref, par_ref = _numpy_spc_reference(logcat, parity_llr)
    max_error = max(
        float(np.max(np.abs(sys_t.cpu().numpy() - sys_ref))),
        float(np.max(np.abs(par_t.cpu().numpy() - par_ref))),
    )
    result["checks"]["siso_vs_bruteforce"] = {
        "max_abs_llr_error": max_error,
        "passed_at_1e-5": bool(max_error < 1e-5),
    }

    decoder, adapter, path = build_spc_altproj_decoder(
        ldpc,
        args.checkpoint,
        schedule,
        args.sigma_end,
        args.delta,
        args.rho,
        device,
    )
    images, bit_bank = load_fashion_mnist()

    # (c) Noiseless/direct-LLR recovery.  This bypasses a zero-variance
    # demapper singularity while representing an exact channel observation.
    indices = tf.range(args.verify_blocks, dtype=tf.int32)
    payload = tf.gather(bit_bank, indices)
    spc_payload = tf.constant(
        encode_spc_numpy(payload.numpy()), dtype=ldpc.rdtype
    )
    codeword = ldpc(crc_encoder(spc_payload))
    direct_llr = (2.0 * codeword - 1.0) * 50.0
    adapter.reset_path(path)
    decoded_logits = decoder(direct_llr)
    _, crc_valid = crc_decoder(decoded_logits)
    hard_spc = (
        decoded_logits[:, :SPC_BITS].numpy() > 0
    ).astype(np.uint8)
    recovered = decode_spc_systematic_numpy(hard_spc)
    result["checks"]["noiseless"] = {
        "blocks": args.verify_blocks,
        "crc_passes": int(
            tf.reduce_sum(tf.cast(crc_valid, tf.int32)).numpy()
        ),
        "payload_exact_blocks": int(
            np.sum(np.all(recovered == payload.numpy(), axis=1))
        ),
        "spc_parity_valid_blocks": int(
            np.sum(np.all(parity_valid_numpy(hard_spc), axis=1))
        ),
    }

    # (d) delta=0 must be the channel-anchored pure BP path.
    delta0, adapter0, path0 = build_spc_altproj_decoder(
        ldpc,
        args.checkpoint,
        schedule,
        args.sigma_end,
        0.0,
        args.rho,
        device,
    )
    common = dict(
        cn_update="boxplus-phi",
        vn_update="sum",
        cn_schedule="flooding",
        hard_out=False,
        return_infobits=True,
        llr_max=30.0,
    )
    bp = LDPC5GDecoder(ldpc, num_iter=100, **common)
    _, payload_noise, _, _, channel_llr = make_spc_channel_batch(
        0,
        args.verify_blocks,
        args.seed + 10,
        -2.7,
        images,
        bit_bank,
        crc_encoder,
        ldpc,
        mapper,
        demapper,
    )
    adapter0.reset_path(path0)
    out_alt0 = delta0(channel_llr)
    out_bp = bp(channel_llr)
    hard_alt0 = out_alt0.numpy() > 0
    hard_bp = out_bp.numpy() > 0
    result["checks"]["delta0_vs_bp100"] = {
        "hard_bit_mismatches": int(np.sum(hard_alt0 != hard_bp)),
        "max_abs_logit_difference": float(
            np.max(np.abs(out_alt0.numpy() - out_bp.numpy()))
        ),
        "payload_bit_mismatches_to_truth_alt0": int(
            np.sum(
                decode_spc_systematic_numpy(
                    hard_alt0[:, :SPC_BITS].astype(np.uint8)
                )
                != payload_noise.numpy().astype(bool)
            )
        ),
    }

    # Runtime feasibility: one full batch and the source-call timing exposed by
    # the adapter.  Warm the graph/model once, then measure a second decode.
    _, _, _, _, runtime_llr = make_spc_channel_batch(
        1,
        args.runtime_batch,
        args.seed + 20,
        -2.7,
        images,
        bit_bank,
        crc_encoder,
        ldpc,
        mapper,
        demapper,
    )
    adapter.reset_path(path)
    _ = decoder(runtime_llr)
    adapter.reset_path(path)
    started = time.perf_counter()
    _ = decoder(runtime_llr)
    full_seconds = time.perf_counter() - started
    result["runtime"] = {
        "batch": args.runtime_batch,
        "source_calls": len(path),
        "full_decode_seconds": float(full_seconds),
        "seconds_per_block": float(full_seconds / args.runtime_batch),
        "last_source_call": adapter.last_runtime,
        "candidate_evaluations_per_decode": int(
            args.runtime_batch * PIXELS * 256 * len(path)
        ),
    }
    dump_json(args.output, result)
    print(f"wrote {args.output}", flush=True)


def _spc_success_masks(
    *,
    configs,
    esn0_db,
    blocks,
    batch,
    seed,
    checkpoint,
):
    if blocks % batch:
        raise ValueError("blocks must be divisible by batch")
    (
        crc_encoder,
        crc_decoder,
        ldpc,
        mapper,
        demapper,
        _,
    ) = make_spc_system()
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = [5] * 20
    decoder, adapter, _ = build_spc_altproj_decoder(
        ldpc,
        checkpoint,
        schedule,
        configs[0]["sigma_end"],
        configs[0]["delta"],
        configs[0]["rho"],
        device,
    )
    decoder.altproj_early_stop = False
    masks = {config["name"]: [] for config in configs}
    for round_idx in range(blocks // batch):
        _, _, _, _, channel_llr = make_spc_channel_batch(
            round_idx,
            batch,
            seed,
            esn0_db,
            images,
            bit_bank,
            crc_encoder,
            ldpc,
            mapper,
            demapper,
        )
        for config in configs:
            _, path = geom_scheduler(
                schedule, config["sigma_end"]
            )
            decoder.altproj_delta = float(config["delta"])
            decoder.altproj_rho = float(config["rho"])
            decoder.altproj_early_stop = False
            adapter.reset_path(path)
            logits = decoder(channel_llr)
            _, valid = crc_decoder(logits)
            masks[config["name"]].extend(
                np.asarray(valid.numpy()).reshape(-1).astype(bool).tolist()
            )
            if adapter.call_index != len(path):
                raise RuntimeError(
                    f"{config['name']} source calls={adapter.call_index}, "
                    f"expected={len(path)}"
                )
        print(
            f"SPC Es/N0={esn0_db:+.2f} "
            f"round {round_idx + 1}/{blocks // batch}",
            flush=True,
        )
    return masks


def _mask_summary(mask, reference=None):
    success = np.asarray(mask, dtype=bool)
    failures = int(np.sum(~success))
    row = {
        "blocks": int(len(success)),
        "failures": failures,
        "crc_bler": float(failures / len(success)),
        "wilson_95": wilson(failures, len(success)),
        "success_mask": success.tolist(),
    }
    if reference is not None:
        reference = np.asarray(reference, dtype=bool)
        row["paired_vs_reference"] = {
            "broken": int(np.sum(reference & ~success)),
            "rescued": int(np.sum(~reference & success)),
            "both_failure": int(np.sum(~reference & ~success)),
            "both_success": int(np.sum(reference & success)),
        }
    return row


def run_spc_screen(args):
    sigma_configs = [
        {
            "name": f"sigma_end_{endpoint:g}",
            "sigma_end": float(endpoint),
            "delta": 0.02,
            "rho": 0.9,
        }
        for endpoint in (0.05, 0.02, 0.01)
    ]
    sigma_masks = _spc_success_masks(
        configs=sigma_configs,
        esn0_db=args.esn0_db,
        blocks=args.blocks,
        batch=args.batch,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
    sigma_reference = sigma_masks["sigma_end_0.02"]
    sigma_rows = {
        config["name"]: {
            **_mask_summary(
                sigma_masks[config["name"]],
                None
                if config["name"] == "sigma_end_0.02"
                else sigma_reference,
            ),
            "config": config,
        }
        for config in sigma_configs
    }
    preference = {0.02: 0, 0.05: 1, 0.01: 2}
    best_sigma_config = min(
        sigma_configs,
        key=lambda config: (
            sigma_rows[config["name"]]["failures"],
            preference[config["sigma_end"]],
        ),
    )
    best_sigma = best_sigma_config["sigma_end"]

    dr_configs = [
        {
            "name": f"delta_{delta:g}_rho_{rho:g}",
            "sigma_end": best_sigma,
            "delta": delta,
            "rho": rho,
        }
        for delta in (0.02, 0.05)
        for rho in (0.9, 0.95)
    ]
    dr_masks = _spc_success_masks(
        configs=dr_configs,
        esn0_db=args.esn0_db,
        blocks=args.blocks,
        batch=args.batch,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
    dr_reference = dr_masks["delta_0.02_rho_0.9"]
    dr_rows = {
        config["name"]: {
            **_mask_summary(
                dr_masks[config["name"]],
                None
                if config["name"] == "delta_0.02_rho_0.9"
                else dr_reference,
            ),
            "config": config,
        }
        for config in dr_configs
    }
    dr_preference = {
        (0.02, 0.9): 0,
        (0.02, 0.95): 1,
        (0.05, 0.9): 2,
        (0.05, 0.95): 3,
    }
    best_dr = min(
        dr_configs,
        key=lambda config: (
            dr_rows[config["name"]]["failures"],
            dr_preference[(config["delta"], config["rho"])],
        ),
    )
    # The canonical delta=0.02 sigma phase can saturate at this higher LDPC
    # rate.  Re-evaluate sigma after choosing delta/rho so endpoint selection
    # is not made from an all-failure tie.
    final_sigma_configs = [
        {
            "name": f"final_sigma_end_{endpoint:g}",
            "sigma_end": float(endpoint),
            "delta": best_dr["delta"],
            "rho": best_dr["rho"],
        }
        for endpoint in (0.05, 0.02, 0.01)
    ]
    final_sigma_masks = _spc_success_masks(
        configs=final_sigma_configs,
        esn0_db=args.esn0_db,
        blocks=args.blocks,
        batch=args.batch,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
    final_reference = final_sigma_masks["final_sigma_end_0.02"]
    final_sigma_rows = {
        config["name"]: {
            **_mask_summary(
                final_sigma_masks[config["name"]],
                None
                if config["name"] == "final_sigma_end_0.02"
                else final_reference,
            ),
            "config": config,
        }
        for config in final_sigma_configs
    }
    final_sigma = min(
        final_sigma_configs,
        key=lambda config: (
            final_sigma_rows[config["name"]]["failures"],
            preference[config["sigma_end"]],
        ),
    )
    selected = {
        **best_dr,
        "sigma_end": final_sigma["sigma_end"],
    }
    result = {
        "kind": "altproj_spc_parameter_screen",
        "configuration": {
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "bp_schedule": [5] * 20,
            "crc_early_stop": False,
            "sigma_post": SIGMA_POST,
            "paired_payload_and_noise": True,
        },
        "sigma_screen": sigma_rows,
        "selected_sigma_end": best_sigma,
        "delta_rho_screen": dr_rows,
        "selected_delta_rho": best_dr,
        "final_sigma_screen": final_sigma_rows,
        "selected": selected,
    }
    dump_json(args.output, result)
    print(f"wrote {args.output}", flush=True)


def run_spc_waterfall(args):
    if len(args.esn0) != len(args.blocks):
        raise ValueError("esn0 and blocks must have equal lengths")
    result = {
        "kind": "altproj_spc_compression_axis_waterfall",
        "configuration": {
            "payload_bits": PAYLOAD_BITS,
            "spc_bits": SPC_BITS,
            "crc_bits": 24,
            "ldpc_k": SPC_BITS + 24,
            "ldpc_n": N_CODEWORD,
            "ldpc_rate": (SPC_BITS + 24) / N_CODEWORD,
            "payload_rate": PAYLOAD_BITS / N_CODEWORD,
            "bp_schedule": [5] * 20,
            "nominal_bp_budget": 100,
            "sigma_start": FIXED_SIGMA,
            "sigma_end": args.sigma_end,
            "sigma_post": SIGMA_POST,
            "delta": args.delta,
            "rho": args.rho,
            "crc_early_stop": False,
            "batch": args.batch,
            "seed": args.seed,
            "knee_interpolation": (
                "linear in BLER, identical to compression baseline"
            ),
        },
        "points": {},
        "knee_bler_0p1_db": None,
        "pixelcnn_max_knee_db": -3.79,
        "gap_to_pixelcnn_max_db": None,
    }
    config = {
        "name": "altproj_spc",
        "sigma_end": args.sigma_end,
        "delta": args.delta,
        "rho": args.rho,
    }
    for point_index, (esn0_db, blocks) in enumerate(
        zip(args.esn0, args.blocks)
    ):
        masks = _spc_success_masks(
            configs=[config],
            esn0_db=esn0_db,
            blocks=blocks,
            batch=args.batch,
            seed=args.seed + point_index * 10000,
            checkpoint=args.checkpoint,
        )
        row = _mask_summary(masks["altproj_spc"])
        row.pop("success_mask")
        row["esn0_db"] = float(esn0_db)
        result["points"][f"{esn0_db:.3f}"] = row
        knee = linear_knee(list(result["points"].values()))
        result["knee_bler_0p1_db"] = knee
        result["gap_to_pixelcnn_max_db"] = (
            None if knee is None else float(knee - (-3.79))
        )
        dump_json(args.output, result)
        print(f"saved SPC {esn0_db:+.2f}", flush=True)


def _crc_check_callable(crc_decoder):
    def check(info_logits):
        _, valid = crc_decoder(info_logits)
        return tf.reshape(tf.cast(valid, tf.bool), [-1])

    return check


def _crc_capture_iterations(history, crc_decoder, schedule, batch):
    captured = np.zeros(batch, dtype=bool)
    iterations = np.full(batch, float(sum(schedule)), dtype=np.float64)
    cumulative = np.cumsum(schedule)
    for chunk_index, info_logits in enumerate(history):
        _, valid = crc_decoder(info_logits)
        valid_np = np.asarray(valid.numpy()).reshape(-1).astype(bool)
        newly = valid_np & ~captured
        iterations[newly] = float(cumulative[chunk_index])
        captured |= valid_np
    return iterations


def _schedule_variants():
    return [
        {"name": "b100_5x20", "budget": 100, "schedule": [5] * 20},
        {"name": "b100_10x10", "budget": 100, "schedule": [10] * 10},
        {"name": "b50_5x10", "budget": 50, "schedule": [5] * 10},
        {"name": "b50_10x5", "budget": 50, "schedule": [10] * 5},
        {"name": "b30_5x6", "budget": 30, "schedule": [5] * 6},
        {"name": "b30_10x3", "budget": 30, "schedule": [10] * 3},
    ]


def run_budget_screen(args):
    if args.blocks % args.batch:
        raise ValueError("blocks must be divisible by batch")
    (
        crc_encoder,
        crc_decoder,
        ldpc,
        mapper,
        demapper,
        awgn,
    ) = make_system()
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    endpoints = (0.05, 0.02, 0.01)
    result = {
        "kind": "altproj_budget_sigma_screen",
        "configuration": {
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "delta": 0.02,
            "rho": 0.9,
            "sigma_start": 0.3,
            "sigma_endpoints": list(endpoints),
            "sigma_post": SIGMA_POST,
            "crc_early_stop": True,
            "actual_iterations": (
                "per-codeword cumulative BP iterations at first CRC pass; "
                "nominal budget if never passing"
            ),
            "paired_payload_and_noise_across_all_configs": True,
        },
        "schedules": {},
        "best_by_budget": {},
    }
    crc_check = _crc_check_callable(crc_decoder)
    all_masks = {}

    for variant in _schedule_variants():
        schedule = variant["schedule"]
        decoder, proxy, _, _ = build_decoder(
            ldpc, args.checkpoint, schedule, device
        )
        decoder._ep_config_logged = True
        decoder.track_u_hat = True
        masks = {
            endpoint: [] for endpoint in endpoints
        }
        iterations = {
            endpoint: [] for endpoint in endpoints
        }
        for round_index in range(args.blocks // args.batch):
            _, _, _, channel_llr = make_channel_batch(
                round_index,
                args.batch,
                args.seed,
                args.esn0_db,
                images,
                bit_bank,
                crc_encoder,
                ldpc,
                mapper,
                demapper,
                awgn,
            )
            for endpoint in endpoints:
                scheduler, path = geom_scheduler(schedule, endpoint)
                config = altproj_candidate(endpoint)
                configure_candidate(
                    decoder, proxy, scheduler, path, config
                )
                decoder.altproj_early_stop = True
                decoder.altproj_crc_check = crc_check
                decoder.track_u_hat = True
                logits = decoder(channel_llr)
                _, valid = crc_decoder(logits)
                masks[endpoint].extend(
                    np.asarray(valid.numpy())
                    .reshape(-1)
                    .astype(bool)
                    .tolist()
                )
                iterations[endpoint].extend(
                    _crc_capture_iterations(
                        decoder.last_u_hat_hist,
                        crc_decoder,
                        schedule,
                        args.batch,
                    ).tolist()
                )
                if proxy.call_index > len(path):
                    raise RuntimeError("sigma schedule over-consumed")
            print(
                f"{variant['name']} round "
                f"{round_index + 1}/{args.blocks // args.batch}",
                flush=True,
            )
        rows = {}
        for endpoint in endpoints:
            name = f"sigma_end_{endpoint:g}"
            row = _mask_summary(masks[endpoint])
            row["actual_mean_bp_iters"] = float(
                np.mean(iterations[endpoint])
            )
            row["actual_iteration_distribution"] = {
                "median": float(np.median(iterations[endpoint])),
                "q10": float(np.quantile(iterations[endpoint], 0.1)),
                "q90": float(np.quantile(iterations[endpoint], 0.9)),
                "min": float(np.min(iterations[endpoint])),
                "max": float(np.max(iterations[endpoint])),
            }
            rows[name] = row
            all_masks[(variant["name"], endpoint)] = np.asarray(
                masks[endpoint], dtype=bool
            )
        endpoint_preference = {0.02: 0, 0.05: 1, 0.01: 2}
        best_endpoint = min(
            endpoints,
            key=lambda endpoint: (
                rows[f"sigma_end_{endpoint:g}"]["failures"],
                rows[f"sigma_end_{endpoint:g}"]["actual_mean_bp_iters"],
                endpoint_preference[endpoint],
            ),
        )
        result["schedules"][variant["name"]] = {
            "budget": variant["budget"],
            "schedule": schedule,
            "endpoints": rows,
            "selected_sigma_end": best_endpoint,
            "selected_result": rows[f"sigma_end_{best_endpoint:g}"],
        }
        dump_json(args.output, result)

    for budget in (100, 50, 30):
        variants = [
            variant for variant in _schedule_variants()
            if variant["budget"] == budget
        ]
        best = min(
            variants,
            key=lambda variant: (
                result["schedules"][variant["name"]][
                    "selected_result"
                ]["failures"],
                result["schedules"][variant["name"]][
                    "selected_result"
                ]["actual_mean_bp_iters"],
                0 if variant["schedule"][0] == 5 else 1,
            ),
        )
        selected = result["schedules"][best["name"]]
        endpoint = selected["selected_sigma_end"]
        result["best_by_budget"][str(budget)] = {
            "variant": best["name"],
            "schedule": best["schedule"],
            "sigma_end": endpoint,
            "result": selected["selected_result"],
        }

    reference_info = result["best_by_budget"]["100"]
    reference_mask = all_masks[
        (reference_info["variant"], reference_info["sigma_end"])
    ]
    for budget in (50, 30):
        info = result["best_by_budget"][str(budget)]
        mask = all_masks[(info["variant"], info["sigma_end"])]
        info["paired_vs_budget100"] = {
            "budget100_success_to_candidate_failure": int(
                np.sum(reference_mask & ~mask)
            ),
            "budget100_failure_to_candidate_success": int(
                np.sum(~reference_mask & mask)
            ),
            "both_failure": int(np.sum(~reference_mask & ~mask)),
            "both_success": int(np.sum(reference_mask & mask)),
        }
    dump_json(args.output, result)
    print(f"wrote {args.output}", flush=True)


def run_budget_waterfall(args):
    if len(args.esn0) != len(args.blocks):
        raise ValueError("esn0 and blocks must have equal lengths")
    if any(blocks % args.batch for blocks in args.blocks):
        raise ValueError("each block allocation must be divisible by batch")
    schedule = [args.chunk] * args.chunks
    nominal_budget = sum(schedule)
    (
        crc_encoder,
        crc_decoder,
        ldpc,
        mapper,
        demapper,
        awgn,
    ) = make_system()
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, proxy, _, _ = build_decoder(
        ldpc, args.checkpoint, schedule, device
    )
    decoder._ep_config_logged = True
    decoder.track_u_hat = True
    decoder.altproj_crc_check = _crc_check_callable(crc_decoder)
    config = altproj_candidate(args.sigma_end)
    result = {
        "kind": "altproj_budget_waterfall",
        "configuration": {
            "bp_schedule": schedule,
            "nominal_bp_budget": nominal_budget,
            "sigma_start": 0.3,
            "sigma_end": args.sigma_end,
            "sigma_post": SIGMA_POST,
            "delta": 0.02,
            "rho": 0.9,
            "crc_early_stop": True,
            "batch": args.batch,
            "seed": args.seed,
            "actual_iterations": (
                "per-codeword cumulative BP iterations at first CRC pass; "
                "nominal budget if never passing"
            ),
        },
        "points": {},
        "knee_bler_0p1_db": None,
        "pixelcnn_max_knee_db": -3.79,
        "gap_to_pixelcnn_max_db": None,
    }
    for point_index, (esn0_db, blocks) in enumerate(
        zip(args.esn0, args.blocks)
    ):
        failures = 0
        iteration_values = []
        for round_index in range(blocks // args.batch):
            _, _, _, channel_llr = make_channel_batch(
                round_index,
                args.batch,
                args.seed + point_index * 10000,
                esn0_db,
                images,
                bit_bank,
                crc_encoder,
                ldpc,
                mapper,
                demapper,
                awgn,
            )
            scheduler, path = geom_scheduler(schedule, args.sigma_end)
            configure_candidate(
                decoder, proxy, scheduler, path, config
            )
            decoder.altproj_early_stop = True
            decoder.altproj_crc_check = _crc_check_callable(crc_decoder)
            decoder.track_u_hat = True
            logits = decoder(channel_llr)
            _, valid = crc_decoder(logits)
            failures += (
                args.batch
                - int(
                    tf.reduce_sum(
                        tf.cast(valid, tf.int32)
                    ).numpy()
                )
            )
            iteration_values.extend(
                _crc_capture_iterations(
                    decoder.last_u_hat_hist,
                    crc_decoder,
                    schedule,
                    args.batch,
                ).tolist()
            )
            print(
                f"budget={nominal_budget} {args.chunk}x{args.chunks} "
                f"Es/N0={esn0_db:+.2f} "
                f"round {round_index + 1}/{blocks // args.batch}",
                flush=True,
            )
        row = {
            "esn0_db": float(esn0_db),
            "blocks": int(blocks),
            "failures": int(failures),
            "crc_bler": float(failures / blocks),
            "wilson_95": wilson(failures, blocks),
            "nominal_bp_iters": nominal_budget,
            "actual_mean_bp_iters": float(np.mean(iteration_values)),
            "actual_iteration_distribution": {
                "median": float(np.median(iteration_values)),
                "q10": float(np.quantile(iteration_values, 0.1)),
                "q90": float(np.quantile(iteration_values, 0.9)),
            },
        }
        result["points"][f"{esn0_db:.3f}"] = row
        knee = linear_knee(list(result["points"].values()))
        result["knee_bler_0p1_db"] = knee
        result["gap_to_pixelcnn_max_db"] = (
            None if knee is None else float(knee - (-3.79))
        )
        dump_json(args.output, result)
        print(f"saved budget waterfall {esn0_db:+.2f}", flush=True)


def run_rate_probe(args):
    crc_encoder = CRCEncoder("CRC24A")
    crc_decoder = CRCDecoder(crc_encoder)
    support_error = None
    try:
        ldpc = LDPC5GEncoder(
            K_PAYLOAD + crc_encoder.crc_length,
            args.n,
            num_bits_per_symbol=1,
        )
    except Exception as exc:
        support_error = f"{type(exc).__name__}: {exc}"
        dump_json(
            args.output,
            {
                "kind": "altproj_rate_probe",
                "supported": False,
                "error": support_error,
                "n": args.n,
            },
        )
        return
    mapper = Mapper("pam", num_bits_per_symbol=1)
    demapper = Demapper("app", "pam", num_bits_per_symbol=1)
    awgn = AWGN()
    images, bit_bank = load_fashion_mnist()
    schedule = [5] * 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, proxy, _, _ = build_decoder(
        ldpc, args.checkpoint, schedule, device
    )
    decoder._ep_config_logged = True
    decoder.track_u_hat = True
    decoder.altproj_crc_check = _crc_check_callable(crc_decoder)
    failures = 0
    iteration_values = []
    config = altproj_candidate(args.sigma_end)
    for round_index in range(args.blocks // args.batch):
        _, _, _, channel_llr = make_channel_batch(
            round_index,
            args.batch,
            args.seed,
            args.esn0_db,
            images,
            bit_bank,
            crc_encoder,
            ldpc,
            mapper,
            demapper,
            awgn,
        )
        scheduler, path = geom_scheduler(schedule, args.sigma_end)
        configure_candidate(
            decoder, proxy, scheduler, path, config
        )
        decoder.altproj_early_stop = True
        decoder.altproj_crc_check = _crc_check_callable(crc_decoder)
        decoder.track_u_hat = True
        logits = decoder(channel_llr)
        _, valid = crc_decoder(logits)
        failures += (
            args.batch
            - int(tf.reduce_sum(tf.cast(valid, tf.int32)).numpy())
        )
        iteration_values.extend(
            _crc_capture_iterations(
                decoder.last_u_hat_hist,
                crc_decoder,
                schedule,
                args.batch,
            ).tolist()
        )
        print(
            f"rate probe N={args.n} Es/N0={args.esn0_db:+.2f} "
            f"round {round_index + 1}/{args.blocks // args.batch}",
            flush=True,
        )
    result = {
        "kind": "altproj_rate_probe",
        "supported": True,
        "configuration": {
            "payload_bits": K_PAYLOAD,
            "crc_bits": int(crc_encoder.crc_length),
            "ldpc_k": int(K_PAYLOAD + crc_encoder.crc_length),
            "n": args.n,
            "ldpc_rate": float(ldpc.coderate),
            "payload_rate": float(K_PAYLOAD / args.n),
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "bp_schedule": schedule,
            "sigma_end": args.sigma_end,
            "crc_early_stop": True,
            "selection_reason": (
                "single exploratory point near the rate-ratio shifted knee; "
                "not a same-resource comparison"
            ),
        },
        "failures": failures,
        "crc_bler": float(failures / args.blocks),
        "wilson_95": wilson(failures, args.blocks),
        "actual_mean_bp_iters": float(np.mean(iteration_values)),
    }
    dump_json(args.output, result)
    print(f"wrote {args.output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    waterfall = sub.add_parser("waterfall")
    waterfall.add_argument(
        "--esn0", type=float, nargs="+", default=COMPRESSION_GRID
    )
    waterfall.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=(3200, 3200, 1024, 1024, 1024, 1024, 1024),
    )
    waterfall.add_argument("--batch", type=int, default=64)
    waterfall.add_argument("--seed", type=int, default=20260808)
    waterfall.add_argument("--sigma-end", type=float, default=0.02)
    waterfall.add_argument("--delta", type=float, default=0.02)
    waterfall.add_argument("--rho", type=float, default=0.9)
    waterfall.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    waterfall.add_argument("--output", required=True)

    feasibility = sub.add_parser("feasibility")
    feasibility.add_argument("--seed", type=int, default=20260809)
    feasibility.add_argument("--verify-blocks", type=int, default=8)
    feasibility.add_argument("--runtime-batch", type=int, default=32)
    feasibility.add_argument("--sigma-end", type=float, default=0.02)
    feasibility.add_argument("--delta", type=float, default=0.02)
    feasibility.add_argument("--rho", type=float, default=0.9)
    feasibility.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    feasibility.add_argument("--output", required=True)

    screen = sub.add_parser("spc-screen")
    screen.add_argument("--esn0-db", type=float, default=-2.7)
    screen.add_argument("--blocks", type=int, default=256)
    screen.add_argument("--batch", type=int, default=32)
    screen.add_argument("--seed", type=int, default=20260810)
    screen.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    screen.add_argument("--output", required=True)

    spc_waterfall = sub.add_parser("spc-waterfall")
    spc_waterfall.add_argument(
        "--esn0", type=float, nargs="+", default=COMPRESSION_GRID
    )
    spc_waterfall.add_argument(
        "--blocks", type=int, nargs="+",
        default=(3200, 1024, 1024, 1024, 1024, 1024, 1024),
    )
    spc_waterfall.add_argument("--batch", type=int, default=32)
    spc_waterfall.add_argument("--seed", type=int, default=20260811)
    spc_waterfall.add_argument("--sigma-end", type=float, required=True)
    spc_waterfall.add_argument("--delta", type=float, required=True)
    spc_waterfall.add_argument("--rho", type=float, required=True)
    spc_waterfall.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    spc_waterfall.add_argument("--output", required=True)

    budget = sub.add_parser("budget-screen")
    budget.add_argument("--esn0-db", type=float, default=-2.825)
    budget.add_argument("--blocks", type=int, default=512)
    budget.add_argument("--batch", type=int, default=64)
    budget.add_argument("--seed", type=int, default=20260812)
    budget.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    budget.add_argument("--output", required=True)

    budget_waterfall = sub.add_parser("budget-waterfall")
    budget_waterfall.add_argument(
        "--esn0", type=float, nargs="+", default=COMPRESSION_GRID
    )
    budget_waterfall.add_argument(
        "--blocks", type=int, nargs="+",
        default=(3200, 3200, 1024, 1024, 1024, 1024, 1024),
    )
    budget_waterfall.add_argument("--chunk", type=int, choices=(5, 10), required=True)
    budget_waterfall.add_argument("--chunks", type=int, required=True)
    budget_waterfall.add_argument("--sigma-end", type=float, required=True)
    budget_waterfall.add_argument("--batch", type=int, default=64)
    budget_waterfall.add_argument("--seed", type=int, default=20260813)
    budget_waterfall.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    budget_waterfall.add_argument("--output", required=True)

    rate_probe = sub.add_parser("rate-probe")
    rate_probe.add_argument("--n", type=int, default=9000)
    rate_probe.add_argument("--esn0-db", type=float, default=-1.5)
    rate_probe.add_argument("--blocks", type=int, default=512)
    rate_probe.add_argument("--batch", type=int, default=64)
    rate_probe.add_argument("--sigma-end", type=float, default=0.05)
    rate_probe.add_argument("--seed", type=int, default=20260814)
    rate_probe.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    rate_probe.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "waterfall":
        run_waterfall(arguments)
    elif arguments.mode == "feasibility":
        run_feasibility(arguments)
    elif arguments.mode == "spc-screen":
        run_spc_screen(arguments)
    elif arguments.mode == "spc-waterfall":
        run_spc_waterfall(arguments)
    elif arguments.mode == "budget-screen":
        run_budget_screen(arguments)
    elif arguments.mode == "budget-waterfall":
        run_budget_waterfall(arguments)
    elif arguments.mode == "rate-probe":
        run_rate_probe(arguments)
