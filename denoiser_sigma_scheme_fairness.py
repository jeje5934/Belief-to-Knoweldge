"""Fair sigma/schedule diagnostics for legacy, EP, and altproj.

This is an experiment harness only.  Production decoder.py is imported
unchanged.  It provides:

* ``profile``: capture each scheme's actual denoiser-input pixel RMSE;
* ``endpoint``: paired screening of geometric sigma endpoints;
* ``awgn``: paired fair 3-way comparison for one BP chunk schedule.

All modes deliberately disable CRC early stopping for every scheme.
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

tf.config.set_visible_devices([], "GPU")

import torch

from channel_models import ChannelModel, QPSK_FADING
from denoiser_sigma_alignment_diag import (
    FIXED_SIGMA,
    K_PAYLOAD,
    describe,
    load_fashion_mnist,
    rmse_per_block,
)
from denoiser_sigma_conditional_diag import make_channel_batch, make_system, wilson
from denoiser_sigma_three_scheme import (
    DEFAULT_CHECKPOINT,
    EP_BETA_CODE_SITE,
    LEGACY_BETA_BP_EXTRINSIC_BOOST,
    SIGMA_POST,
    SigmaScheduleProxy,
    build_decoder,
    configure_candidate,
    dump_json,
    summarize_mask,
)
from syndrome_sigma_schedule import AnnealingSigmaScheduler
from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.utils import ebnodb2no


def parse_schedule(text):
    """Parse ``5x20`` or ``10x10`` into a chunk schedule."""
    left, right = text.lower().split("x", 1)
    per_chunk, chunks = int(left), int(right)
    if per_chunk <= 0 or chunks <= 1:
        raise ValueError("schedule must have positive chunk size and >=2 chunks")
    if per_chunk * chunks != 100:
        raise ValueError("this experiment fixes the total BP budget at 100")
    return [per_chunk] * chunks


def geom_schedule(schedule, endpoint):
    scheduler = AnnealingSigmaScheduler(
        FIXED_SIGMA, float(endpoint), len(schedule) - 1, mode="geom"
    )
    return scheduler, tuple(float(value) for value in scheduler.path)


def candidate(scheme, endpoint, *, alpha=0.1, alpha_ep=0.01,
              delta=0.02, rho=0.9):
    row = {
        "name": f"{scheme}_s{endpoint:g}",
        "scheme": scheme,
        "sigma_end": float(endpoint),
    }
    if scheme == "legacy":
        row["alpha"] = float(alpha)
    elif scheme == "ep":
        row["alpha_ep"] = float(alpha_ep)
    elif scheme == "altproj":
        row["delta"] = float(delta)
        row["rho"] = float(rho)
    else:
        raise ValueError(scheme)
    return row


def selected_candidates(endpoints):
    return [
        candidate("legacy", endpoints["legacy"]),
        candidate("ep", endpoints["ep"]),
        candidate("altproj", endpoints["altproj"]),
    ]


def replace_proxy(decoder, old_proxy, *, capture):
    inner = old_proxy.inner
    if capture:
        proxy = CaptureSigmaScheduleProxy(inner)
    else:
        proxy = SigmaScheduleProxy(inner)
    decoder._denoiser = proxy
    return proxy


class CaptureSigmaScheduleProxy(tf.keras.layers.Layer):
    """Sigma override plus non-invasive capture of the true denoiser input."""

    def __init__(self, inner):
        super().__init__(name="sigma_actual_capture_proxy")
        self.inner = inner
        self.override = False
        self.path = ()
        self.call_index = 0
        self.truth = None
        self.capture_enabled = False
        self.actual_rmse = []
        self.reported_sigma = []

    def reset(self, override, path):
        self.override = bool(override)
        self.path = tuple(float(value) for value in path)
        self.call_index = 0
        self.actual_rmse = []
        self.reported_sigma = []

    def set_truth(self, truth_image):
        device = self.inner._device
        self.truth = torch.from_numpy(
            np.asarray(truth_image, dtype=np.float32)
        ).reshape(-1, 1, 28, 28).to(device)

    def call(self, input_llr, sigma=None):
        if self.override:
            if self.call_index >= len(self.path):
                raise RuntimeError(
                    "altproj made more denoiser calls than the sigma path"
                )
            sigma = self.path[self.call_index]
            self.call_index += 1

        if self.capture_enabled:
            if self.truth is None:
                raise RuntimeError("truth image was not set before capture")
            llr = torch.from_numpy(input_llr.numpy()).float().to(
                self.inner._device
            )
            with torch.no_grad():
                mu = self.inner.prior_model.llr_to_soft_field(llr)
                actual = rmse_per_block(mu, self.truth)
            self.actual_rmse.append(
                actual.detach().cpu().numpy().astype(np.float64)
            )
            if sigma is None:
                reported = np.full(
                    input_llr.shape[0], float(self.inner.sigma), dtype=np.float64
                )
            elif isinstance(sigma, tf.Tensor):
                reported = np.asarray(sigma.numpy(), dtype=np.float64).reshape(-1)
            else:
                reported = np.asarray(sigma, dtype=np.float64).reshape(-1)
            if reported.size == 1:
                reported = np.full(
                    input_llr.shape[0], float(reported[0]), dtype=np.float64
                )
            self.reported_sigma.append(reported)

        return self.inner(input_llr, sigma=sigma)


def run_profile(args):
    schedule = parse_schedule(args.schedule)
    system = make_system()
    crc_encoder, crc_decoder, ldpc, mapper, demapper, awgn = system
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, old_proxy, _, _ = build_decoder(
        ldpc, args.checkpoint, schedule, device
    )
    proxy = replace_proxy(decoder, old_proxy, capture=True)
    decoder._ep_config_logged = True
    decoder.altproj_early_stop = False

    _, _, truth_image, channel_llr = make_channel_batch(
        0,
        args.blocks,
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
    profile_candidates = selected_candidates(
        {"legacy": args.endpoint, "ep": args.endpoint, "altproj": args.endpoint}
    )
    profiles = {}
    for item in profile_candidates:
        scheduler, path = geom_schedule(schedule, item["sigma_end"])
        configure_candidate(decoder, proxy, scheduler, path, item)
        decoder.altproj_early_stop = False
        proxy.set_truth(truth_image)
        proxy.capture_enabled = True
        logits = decoder(channel_llr)
        _, crc_valid = hard_crc_decode(crc_decoder, logits)
        if len(proxy.actual_rmse) != len(schedule) - 1:
            raise RuntimeError(
                f"{item['scheme']} captured {len(proxy.actual_rmse)} calls, "
                f"expected {len(schedule) - 1}"
            )
        actual = np.stack(proxy.actual_rmse, axis=1)
        reported = np.stack(proxy.reported_sigma, axis=1)
        block_mean = np.mean(actual, axis=1)
        profiles[item["scheme"]] = {
            "candidate": item,
            "crc_failures": int(
                args.blocks
                - np.sum(np.asarray(crc_valid.numpy()).reshape(-1).astype(bool))
            ),
            "trajectory": [
                {
                    "source_call": idx + 1,
                    "reported_sigma": float(np.mean(reported[:, idx])),
                    "actual": describe(actual[:, idx]),
                    "reported_over_actual_median": float(
                        np.median(
                            reported[:, idx]
                            / np.maximum(actual[:, idx], 1e-12)
                        )
                    ),
                    "fraction_underreported": float(
                        np.mean(reported[:, idx] < actual[:, idx])
                    ),
                }
                for idx in range(actual.shape[1])
            ],
            "block_mean_actual": describe(block_mean),
            "actual_matrix": actual.tolist(),
            "reported_matrix": reported.tolist(),
        }

    pairwise = {}
    for left, right in (
        ("ep", "legacy"),
        ("altproj", "legacy"),
        ("ep", "altproj"),
    ):
        a = np.mean(np.asarray(profiles[left]["actual_matrix"]), axis=1)
        b = np.mean(np.asarray(profiles[right]["actual_matrix"]), axis=1)
        diff = a - b
        se = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
        pairwise[f"{left}_minus_{right}"] = {
            "paired_block_mean_difference": describe(diff),
            "normal_95": [
                float(np.mean(diff) - 1.96 * se),
                float(np.mean(diff) + 1.96 * se),
            ],
            "relative_to_right_mean": float(np.mean(diff) / np.mean(b)),
        }

    payload = {
        "kind": "three_scheme_sigma_actual_profile",
        "configuration": {
            "branch": "practical_sigma",
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "seed": args.seed,
            "bp_schedule": schedule,
            "source_calls": len(schedule) - 1,
            "early_stop": "disabled_for_all_schemes",
            "legacy_beta": LEGACY_BETA_BP_EXTRINSIC_BOOST,
            "ep_beta_ep": EP_BETA_CODE_SITE,
            "sigma_post": SIGMA_POST,
            "paired_payload_and_noise": True,
        },
        "profiles": profiles,
        "paired_comparisons": pairwise,
    }
    dump_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


def run_paired(
    *,
    args,
    candidates,
    schedule,
    comparison_reference,
):
    system = make_system()
    crc_encoder, crc_decoder, ldpc, mapper, demapper, awgn = system
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, old_proxy, _, _ = build_decoder(
        ldpc, args.checkpoint, schedule, device
    )
    proxy = replace_proxy(decoder, old_proxy, capture=False)
    decoder._ep_config_logged = True
    decoder.altproj_early_stop = False
    masks = {item["name"]: [] for item in candidates}
    rounds = args.blocks // args.batch
    if rounds * args.batch != args.blocks:
        raise ValueError("blocks must be divisible by batch")

    for round_idx in range(rounds):
        _, _, _, channel_llr = make_channel_batch(
            round_idx,
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
        for item in candidates:
            scheduler, path = geom_schedule(schedule, item["sigma_end"])
            configure_candidate(decoder, proxy, scheduler, path, item)
            decoder.altproj_early_stop = False
            logits = decoder(channel_llr)
            _, crc_valid = hard_crc_decode(crc_decoder, logits)
            masks[item["name"]].extend(
                np.asarray(crc_valid.numpy()).reshape(-1).astype(bool).tolist()
            )
            if item["scheme"] == "altproj" and proxy.call_index != len(path):
                raise RuntimeError(
                    f"altproj source calls={proxy.call_index}, expected={len(path)}"
                )
        print(
            f"{args.schedule} {args.esn0_db:+.2f} "
            f"round {round_idx + 1}/{rounds}",
            flush=True,
        )

    ref = np.asarray(masks[comparison_reference], dtype=bool)
    rows = {}
    for item in candidates:
        name = item["name"]
        rows[name] = summarize_mask(
            masks[name], None if name == comparison_reference else ref
        )
        rows[name]["candidate"] = item
    return rows


def run_endpoint(args):
    schedule = parse_schedule(args.schedule)
    candidates = []
    for scheme in ("legacy", "ep", "altproj"):
        for endpoint in args.endpoints:
            candidates.append(candidate(scheme, endpoint))
    reference = f"legacy_s{args.reference_endpoint:g}"
    rows = run_paired(
        args=args,
        candidates=candidates,
        schedule=schedule,
        comparison_reference=reference,
    )
    # Add within-scheme transitions relative to each scheme's 0.02 arm.
    for item in candidates:
        name = item["name"]
        scheme_ref = f"{item['scheme']}_s{args.reference_endpoint:g}"
        if name == scheme_ref:
            continue
        success = np.asarray(rows[name]["success_mask"], dtype=bool)
        ref = np.asarray(rows[scheme_ref]["success_mask"], dtype=bool)
        rows[name]["vs_same_scheme_reference"] = {
            "reference": scheme_ref,
            "reference_success_to_candidate_failure": int(
                np.sum(ref & ~success)
            ),
            "reference_failure_to_candidate_success": int(
                np.sum(~ref & success)
            ),
        }
    payload = {
        "kind": "three_scheme_sigma_endpoint_screen",
        "configuration": {
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "seed": args.seed,
            "bp_schedule": schedule,
            "early_stop": "disabled_for_all_schemes",
            "legacy_beta": LEGACY_BETA_BP_EXTRINSIC_BOOST,
            "ep_beta_ep": EP_BETA_CODE_SITE,
            "sigma_post": SIGMA_POST,
            "paired_payload_and_noise": True,
        },
        "results": rows,
    }
    dump_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


def run_awgn(args):
    schedule = parse_schedule(args.schedule)
    candidates = selected_candidates(
        {
            "legacy": args.legacy_endpoint,
            "ep": args.ep_endpoint,
            "altproj": args.altproj_endpoint,
        }
    )
    reference = candidates[0]["name"]
    rows = run_paired(
        args=args,
        candidates=candidates,
        schedule=schedule,
        comparison_reference=reference,
    )
    payload = {
        "kind": "fair_three_scheme_awgn",
        "configuration": {
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "seed": args.seed,
            "bp_schedule": schedule,
            "bp_budget": sum(schedule),
            "early_stop": "disabled_for_all_schemes",
            "legacy_beta": LEGACY_BETA_BP_EXTRINSIC_BOOST,
            "ep_beta_ep": EP_BETA_CODE_SITE,
            "sigma_post": SIGMA_POST,
            "paired_payload_and_noise": True,
        },
        "results": rows,
    }
    dump_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


def run_fading(args):
    """Re-evaluate the A-selected arms on the historical method-B grid."""
    schedule = parse_schedule(args.schedule)
    crc_encoder = CRCEncoder("CRC24A")
    crc_decoder = CRCDecoder(crc_encoder)
    ldpc = LDPC5GEncoder(
        K_PAYLOAD + crc_encoder.crc_length,
        12600,
        num_bits_per_symbol=2,
    )
    _, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, old_proxy, _, _ = build_decoder(
        ldpc, args.checkpoint, schedule, device
    )
    proxy = replace_proxy(decoder, old_proxy, capture=False)
    decoder._ep_config_logged = True
    decoder.altproj_early_stop = False

    candidates = [
        candidate(
            "legacy",
            args.legacy_endpoint,
            alpha=args.legacy_alpha,
        ),
        candidate(
            "ep",
            args.ep_endpoint,
            alpha_ep=args.ep_alpha,
        ),
        candidate(
            "altproj",
            args.altproj_endpoint,
            delta=args.altproj_delta,
            rho=args.altproj_rho,
        ),
    ]
    # Keep names self-describing because the fading legacy alpha differs from
    # its AWGN value.
    candidates[0]["name"] = (
        f"legacy_a{args.legacy_alpha:g}_s{args.legacy_endpoint:g}"
    )
    candidates[1]["name"] = (
        f"ep_a{args.ep_alpha:g}_s{args.ep_endpoint:g}"
    )
    candidates[2]["name"] = (
        f"altproj_d{args.altproj_delta:g}_r{args.altproj_rho:g}"
        f"_s{args.altproj_endpoint:g}"
    )
    reference_name = candidates[0]["name"]

    if args.blocks % args.batch:
        raise ValueError("blocks must be divisible by batch")
    rounds = args.blocks // args.batch
    points = [
        (float(sigma_e2), float(ebno))
        for sigma_e2 in args.sigma_e2
        for ebno in args.ebno
    ]
    result = {
        "kind": "sigma_corrected_fading_three_scheme",
        "configuration": {
            "branch": "practical_sigma",
            "channel": QPSK_FADING,
            "llr_method": "B",
            "ar_coeff": 0.0,
            "blocks_per_point": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "bp_schedule": schedule,
            "bp_budget": sum(schedule),
            "early_stop": "disabled_for_all_schemes",
            "legacy_beta": LEGACY_BETA_BP_EXTRINSIC_BOOST,
            "ep_beta_ep": EP_BETA_CODE_SITE,
            "sigma_post": SIGMA_POST,
            "channel_llr_generated_once_per_batch": True,
            "paired_payload_noise_fading_and_csi": True,
            "historical_ebno_grid": list(args.ebno),
        },
        "candidates": candidates,
        "points": {},
    }

    for point_index, (sigma_e2, ebno) in enumerate(points):
        channel = ChannelModel(
            QPSK_FADING,
            sigma_e2=sigma_e2,
            perfect_csi=(sigma_e2 == 0.0),
            llr_method="B",
            ar_coeff=0.0,
        )
        no = ebnodb2no(ebno, channel.num_bps, ldpc.coderate)
        masks = {item["name"]: [] for item in candidates}
        for round_idx in range(rounds):
            sample_seed = tf.constant(
                [args.seed, point_index * 1000 + round_idx],
                dtype=tf.int32,
            )
            indices = tf.random.stateless_uniform(
                [args.batch],
                seed=sample_seed,
                minval=0,
                maxval=tf.shape(bit_bank)[0],
                dtype=tf.int32,
            )
            payload = tf.gather(bit_bank, indices)
            codeword = ldpc(
                crc_encoder(tf.cast(payload, ldpc.rdtype))
            )
            # ChannelModel is stateful internally, but it is called exactly
            # once here.  Every decoder receives this same tensor.
            tf.random.set_seed(
                args.seed + point_index * 1000 + round_idx
            )
            channel_llr = channel.transmit(codeword, no)
            for item in candidates:
                scheduler, path = geom_schedule(
                    schedule, item["sigma_end"]
                )
                configure_candidate(
                    decoder, proxy, scheduler, path, item
                )
                decoder.altproj_early_stop = False
                logits = decoder(channel_llr)
                _, crc_valid = hard_crc_decode(crc_decoder, logits)
                masks[item["name"]].extend(
                    np.asarray(crc_valid.numpy())
                    .reshape(-1)
                    .astype(bool)
                    .tolist()
                )
                if (
                    item["scheme"] == "altproj"
                    and proxy.call_index != len(path)
                ):
                    raise RuntimeError(
                        "altproj did not consume the complete sigma schedule"
                    )
            print(
                f"se2={sigma_e2:.2f} Eb/N0={ebno:.1f} "
                f"round {round_idx + 1}/{rounds}",
                flush=True,
            )

        reference = np.asarray(masks[reference_name], dtype=bool)
        rows = {}
        for item in candidates:
            name = item["name"]
            rows[name] = summarize_mask(
                masks[name],
                None if name == reference_name else reference,
            )
            rows[name]["candidate"] = item
        key = f"se2={sigma_e2:.3f},ebno={ebno:.3f}"
        result["points"][key] = {
            "sigma_e2": sigma_e2,
            "perfect_csi": sigma_e2 == 0.0,
            "ebno_db": ebno,
            "results": rows,
        }
        dump_json(args.output, result)
        print(f"saved {key} to {args.output}", flush=True)


def common_performance_args(parser):
    parser.add_argument("--esn0-db", type=float, default=-2.7)
    parser.add_argument("--blocks", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--schedule", choices=("5x20", "10x10"), default="5x20")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", required=True)


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    profile = sub.add_parser("profile")
    profile.add_argument("--esn0-db", type=float, default=-2.7)
    profile.add_argument("--blocks", type=int, default=128)
    profile.add_argument("--seed", type=int, default=20260804)
    profile.add_argument("--schedule", choices=("5x20",), default="5x20")
    profile.add_argument("--endpoint", type=float, default=0.02)
    profile.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    profile.add_argument("--output", required=True)

    endpoint = sub.add_parser("endpoint")
    common_performance_args(endpoint)
    endpoint.set_defaults(blocks=256)
    endpoint.add_argument(
        "--endpoints", type=float, nargs="+", default=(0.05, 0.02, 0.01)
    )
    endpoint.add_argument("--reference-endpoint", type=float, default=0.02)

    awgn = sub.add_parser("awgn")
    common_performance_args(awgn)
    awgn.add_argument("--legacy-endpoint", type=float, required=True)
    awgn.add_argument("--ep-endpoint", type=float, required=True)
    awgn.add_argument("--altproj-endpoint", type=float, required=True)

    fading = sub.add_parser("fading")
    fading.add_argument("--sigma-e2", type=float, nargs="+",
                        default=(0.0, 0.1, 0.2))
    fading.add_argument("--ebno", type=float, nargs="+",
                        default=(0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0))
    fading.add_argument("--blocks", type=int, default=512)
    fading.add_argument("--batch", type=int, default=64)
    fading.add_argument("--seed", type=int, default=20260807)
    fading.add_argument("--schedule", choices=("5x20",), default="5x20")
    fading.add_argument("--legacy-endpoint", type=float, default=0.02)
    fading.add_argument("--ep-endpoint", type=float, default=0.05)
    fading.add_argument("--altproj-endpoint", type=float, default=0.02)
    fading.add_argument("--legacy-alpha", type=float, default=0.15)
    fading.add_argument("--ep-alpha", type=float, default=0.01)
    fading.add_argument("--altproj-delta", type=float, default=0.02)
    fading.add_argument("--altproj-rho", type=float, default=0.9)
    fading.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    fading.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "profile":
        run_profile(arguments)
    elif arguments.mode == "endpoint":
        run_endpoint(arguments)
    elif arguments.mode == "awgn":
        run_awgn(arguments)
    else:
        run_fading(arguments)
