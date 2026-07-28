"""
Re-evaluate legacy / damped EP / altproj after denoiser-sigma calibration.

Production ``decoder.py`` is imported unchanged.  Legacy and EP consume the
geometric schedule through the decoder's existing ``sigma_scheduler`` API.
Altproj currently exposes only a scalar ``altproj_sigma_den``; for a fair common
schedule this harness wraps the already-loaded SoftDenoiser and replaces only
the sigma argument on its source calls.  An exact source-call-count assertion is
checked after every altproj decode.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from crc_utils import hard_crc_decode

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
from syndrome_sigma_schedule import AnnealingSigmaScheduler


DEFAULT_CHECKPOINT = (
    "/home/LJH/onlyextrinsic_ada_sigma/checkpoints/denoiser.pt"
)
SIGMA_END = 0.02
SIGMA_POST = 3.0

# IMPORTANT: these two beta-like parameters have opposite semantics.
#
# * Legacy ``beta`` is an EXTRA BP-extrinsic booster in
#     channel + beta*bp_ext + alpha*src_ext.
#   beta=0 is the fair standard-LDPC-BP baseline and is fixed here.
# * EP ``ep_code_power`` (beta_ep) is the code-factor site damping.  beta_ep=0
#   removes the code factor and invalidates the EP formulation; beta_ep=1 is the
#   required default here.
LEGACY_BETA_BP_EXTRINSIC_BOOST = 0.0
EP_BETA_CODE_SITE = 1.0


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def dump_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def budget_schedule(budget):
    budget = int(budget)
    if budget not in (30, 50, 100):
        raise ValueError("budget must be one of 30, 50, 100")
    return [5] * (budget // 5)


class SigmaScheduleProxy(tf.keras.layers.Layer):
    """Pass-through denoiser with an altproj-only sigma-argument override."""

    def __init__(self, inner):
        super().__init__(name="altproj_sigma_schedule_proxy")
        self.inner = inner
        self.override = False
        self.path = ()
        self.call_index = 0

    def reset(self, override, path):
        self.override = bool(override)
        self.path = tuple(float(value) for value in path)
        self.call_index = 0

    def call(self, input_llr, sigma=None):
        if self.override:
            if self.call_index >= len(self.path):
                raise RuntimeError(
                    "altproj made more denoiser calls than the sigma path"
                )
            sigma = self.path[self.call_index]
            self.call_index += 1
        return self.inner(input_llr, sigma=sigma)


def common_path(schedule):
    source_calls = len(schedule) - 1
    scheduler = AnnealingSigmaScheduler(
        FIXED_SIGMA, SIGMA_END, source_calls, mode="geom"
    )
    return scheduler, tuple(float(value) for value in scheduler.path)


def build_decoder(ldpc, checkpoint, schedule, device):
    common = dict(
        cn_update="boxplus-phi",
        vn_update="sum",
        cn_schedule="flooding",
        hard_out=False,
        return_infobits=True,
        llr_max=30.0,
    )
    scheduler, path = common_path(schedule)
    decoder = LDPC5GDecoder_soft(
        ldpc,
        k_payload=K_PAYLOAD,
        num_iter=sum(schedule),
        bp_schedule=schedule,
        alpha=0.1,
        beta=LEGACY_BETA_BP_EXTRINSIC_BOOST,
        ep_mode=False,
        source_input_mode="bp_post",
        ep_update="damped_ep",
        ep_source_power=0.01,
        ep_code_power=EP_BETA_CODE_SITE,
        altproj=False,
        altproj_delta=0.02,
        altproj_rho=0.9,
        altproj_sigma_den=FIXED_SIGMA,
        altproj_warm_start=True,
        altproj_early_stop=False,
        sigma_scheduler=scheduler,
        adaptive_sigma=True,
        ep_track_payload_hist=False,
        denoiser_kwargs=dict(device=device),
        **common,
    )
    decoder.denoiser.load_weights_pt(checkpoint)
    decoder.denoiser.sigma = FIXED_SIGMA
    decoder.denoiser.sigma_post = SIGMA_POST
    inner = decoder.denoiser
    proxy = SigmaScheduleProxy(inner)
    # Replacement happens before the first decoder call.  The proxy delegates
    # every numerical operation to the unchanged, already-loaded SoftDenoiser.
    decoder._denoiser = proxy
    return decoder, proxy, scheduler, path


def grid_candidates():
    candidates = [
        {
            "name": "legacy_a0.1",
            "scheme": "legacy",
            "alpha": 0.1,
        }
    ]
    for alpha_ep in (0.01, 0.02, 0.05):
        candidates.append(
            {
                "name": f"ep_a{alpha_ep:g}",
                "scheme": "ep",
                "alpha_ep": alpha_ep,
            }
        )
    for delta in (0.02, 0.05):
        for rho in (0.9, 0.95, 1.0):
            candidates.append(
                {
                    "name": f"altproj_d{delta:g}_r{rho:g}",
                    "scheme": "altproj",
                    "delta": delta,
                    "rho": rho,
                }
            )
    return candidates


def selected_candidates(ep_alpha, delta, rho):
    return [
        {
            "name": "legacy_a0.1",
            "scheme": "legacy",
            "alpha": 0.1,
        },
        {
            "name": f"ep_a{ep_alpha:g}",
            "scheme": "ep",
            "alpha_ep": float(ep_alpha),
        },
        {
            "name": f"altproj_d{delta:g}_r{rho:g}",
            "scheme": "altproj",
            "delta": float(delta),
            "rho": float(rho),
        },
    ]


def ep_recheck_candidates(delta, rho):
    candidates = [
        {
            "name": "legacy_a0.1",
            "scheme": "legacy",
            "alpha": 0.1,
        }
    ]
    for alpha_ep in (0.01, 0.02, 0.05):
        candidates.append(
            {
                "name": f"ep_a{alpha_ep:g}",
                "scheme": "ep",
                "alpha_ep": alpha_ep,
            }
        )
    candidates.append(
        {
            "name": f"altproj_d{delta:g}_r{rho:g}",
            "scheme": "altproj",
            "delta": float(delta),
            "rho": float(rho),
        }
    )
    return candidates


def configure_candidate(
    decoder,
    proxy,
    scheduler,
    path,
    candidate,
):
    decoder.altproj = False
    decoder.ep_mode = False
    decoder.alpha = 0.1
    # Legacy beta and EP beta_ep are deliberately set separately.  Do not
    # "share" one beta knob across schemes: their meanings are opposite.
    decoder.beta = LEGACY_BETA_BP_EXTRINSIC_BOOST
    decoder.ep_update = "damped_ep"
    decoder.ep_source_power = 0.01
    decoder.ep_code_power = EP_BETA_CODE_SITE
    decoder.ep_source_power_schedule = None
    decoder.ep_source_off_tail = 0
    decoder.altproj_delta = 0.02
    decoder.altproj_rho = 0.9
    decoder.altproj_sigma_den = FIXED_SIGMA
    decoder.altproj_warm_start = True
    decoder.altproj_early_stop = False
    decoder.sigma_scheduler = scheduler
    proxy.reset(False, path)

    scheme = candidate["scheme"]
    if scheme == "legacy":
        decoder.alpha = float(candidate["alpha"])
    elif scheme == "ep":
        decoder.ep_mode = True
        decoder.ep_source_power = float(candidate["alpha_ep"])
        if decoder.ep_code_power != EP_BETA_CODE_SITE:
            raise RuntimeError(
                "EP beta_ep must remain 1.0; beta_ep=0 removes the code factor"
            )
    elif scheme == "altproj":
        decoder.altproj = True
        decoder.altproj_delta = float(candidate["delta"])
        decoder.altproj_rho = float(candidate["rho"])
        proxy.reset(True, path)
    else:
        raise ValueError(scheme)


def summarize_mask(success, reference=None):
    success = np.asarray(success, dtype=bool)
    failures = int(np.sum(~success))
    row = {
        "failures": failures,
        "blocks": int(len(success)),
        "crc_bler": float(failures / len(success)),
        "wilson_95": wilson(failures, len(success)),
        "success_mask": success.tolist(),
    }
    if reference is not None:
        reference = np.asarray(reference, dtype=bool)
        broken = int(np.sum(reference & ~success))
        rescued = int(np.sum(~reference & success))
        row["vs_same_budget_legacy"] = {
            "legacy_success_to_candidate_failure": broken,
            "legacy_failure_to_candidate_success": rescued,
            "net_failure_change": broken - rescued,
            "both_failure": int(np.sum(~reference & ~success)),
            "both_success": int(np.sum(reference & success)),
        }
    return row


def empty_diag():
    return {
        "decode_calls": 0,
        "ep_diverged_calls": 0,
        "ep_max_site_abs": 0.0,
        "ep_max_site_delta_l2": 0.0,
        "altproj_max_mean_abs_C": 0.0,
        "altproj_max_mean_abs_D": 0.0,
        "altproj_sigma_call_assertions": 0,
    }


def update_diag(aggregate, decoder, proxy, candidate, source_calls):
    aggregate["decode_calls"] += 1
    if candidate["scheme"] == "ep":
        rows = decoder.last_chunk_diagnostics
        aggregate["ep_diverged_calls"] += int(
            any(bool(row.get("ep_diverged", False)) for row in rows)
        )
        aggregate["ep_max_site_abs"] = max(
            aggregate["ep_max_site_abs"],
            max(
                (
                    float(row.get("src_site_max_abs", 0.0))
                    for row in rows
                ),
                default=0.0,
            ),
        )
        aggregate["ep_max_site_delta_l2"] = max(
            aggregate["ep_max_site_delta_l2"],
            max(
                (
                    float(row.get("src_site_delta_l2", 0.0))
                    for row in rows
                ),
                default=0.0,
            ),
        )
    elif candidate["scheme"] == "altproj":
        if proxy.call_index != source_calls:
            raise RuntimeError(
                f"altproj sigma schedule calls={proxy.call_index}, "
                f"expected={source_calls}"
            )
        aggregate["altproj_sigma_call_assertions"] += 1
        rows = decoder.last_altproj_diag
        aggregate["altproj_max_mean_abs_C"] = max(
            aggregate["altproj_max_mean_abs_C"],
            max(
                (float(row.get("mean_abs_C", 0.0)) for row in rows),
                default=0.0,
            ),
        )
        aggregate["altproj_max_mean_abs_D"] = max(
            aggregate["altproj_max_mean_abs_D"],
            max(
                (
                    float(row.get("mean_abs_D") or 0.0)
                    for row in rows
                ),
                default=0.0,
            ),
        )


def run_point(
    *,
    candidates,
    schedule,
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
        awgn,
    ) = make_system()
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, proxy, scheduler, path = build_decoder(
        ldpc, checkpoint, schedule, device
    )
    # Avoid a stale one-time configuration print when the same decoder instance
    # is intentionally reconfigured among paired arms.
    decoder._ep_config_logged = True
    source_calls = len(schedule) - 1
    masks = {candidate["name"]: [] for candidate in candidates}
    diagnostics = {candidate["name"]: empty_diag() for candidate in candidates}
    rounds = blocks // batch

    for round_idx in range(rounds):
        _, _, _, channel_llr = make_channel_batch(
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
            awgn,
        )
        for candidate in candidates:
            configure_candidate(
                decoder,
                proxy,
                scheduler,
                path,
                candidate,
            )
            final_logits = decoder(channel_llr)
            _, crc_valid = hard_crc_decode(crc_decoder, final_logits)
            masks[candidate["name"]].extend(
                np.asarray(crc_valid.numpy())
                .reshape(-1)
                .astype(bool)
                .tolist()
            )
            update_diag(
                diagnostics[candidate["name"]],
                decoder,
                proxy,
                candidate,
                source_calls,
            )
        print(
            f"budget={sum(schedule)} Es/N0={esn0_db:+.2f} "
            f"round {round_idx + 1}/{rounds}",
            flush=True,
        )

    reference = np.asarray(masks["legacy_a0.1"], dtype=bool)
    rows = {}
    for candidate in candidates:
        name = candidate["name"]
        rows[name] = summarize_mask(
            masks[name],
            None if name == "legacy_a0.1" else reference,
        )
        rows[name]["candidate"] = candidate
        rows[name]["diagnostics"] = diagnostics[name]
    return rows, path


def run_evaluate(args):
    schedule = budget_schedule(args.budget)
    snrs = parse_float_list(args.snrs)
    allocations = parse_int_list(args.blocks_per_snr)
    if len(snrs) != len(allocations):
        raise ValueError("snrs and blocks-per-snr must have equal lengths")
    if args.candidate_set == "grid":
        if args.budget != 100:
            raise ValueError("full grid is defined only for budget 100")
        candidates = grid_candidates()
    elif args.candidate_set == "ep_recheck":
        if args.budget != 100:
            raise ValueError("EP beta_ep recheck is defined only for budget 100")
        candidates = ep_recheck_candidates(args.delta, args.rho)
    else:
        candidates = selected_candidates(
            args.ep_alpha, args.delta, args.rho
        )

    _, path = common_path(schedule)
    result = {
        "kind": "sigma_corrected_three_scheme_evaluation",
        "configuration": {
            "branch": "practical_sigma",
            "candidate_set": args.candidate_set,
            "bp_schedule": schedule,
            "bp_budget": sum(schedule),
            "source_calls": len(schedule) - 1,
            "legacy_beta_bp_extrinsic_boost": (
                LEGACY_BETA_BP_EXTRINSIC_BOOST
            ),
            "legacy_alpha": 0.1,
            "ep_beta_code_site": EP_BETA_CODE_SITE,
            "beta_parameter_contract": {
                "legacy_beta": (
                    "extra BP-extrinsic feedback boost; fixed 0"
                ),
                "ep_beta_ep": (
                    "code-factor site damping; required 1; zero is invalid"
                ),
            },
            "sigma_start": FIXED_SIGMA,
            "sigma_end": SIGMA_END,
            "sigma_mode": "geom",
            "sigma_post": SIGMA_POST,
            "batch": args.batch,
            "seed": args.seed,
            "channel": "BPSK/AWGN/perfect_CSI",
            "paired_payload_and_noise": True,
            "altproj_sigma_injection": (
                "harness denoiser-call proxy; decoder.py unchanged; "
                "per-decode source-call count asserted"
            ),
        },
        "sigma_path": list(path),
        "candidates": candidates,
        "snr_results": {},
    }
    for snr, blocks in zip(snrs, allocations):
        rows, observed_path = run_point(
            candidates=candidates,
            schedule=schedule,
            esn0_db=snr,
            blocks=blocks,
            batch=args.batch,
            seed=args.seed,
            checkpoint=args.checkpoint,
        )
        if tuple(result["sigma_path"]) != tuple(observed_path):
            raise RuntimeError("sigma path changed across SNR points")
        result["snr_results"][f"{snr:.3f}"] = rows
        dump_json(args.output, result)
        print(f"saved completed SNR {snr:+.2f} to {args.output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument(
        "--candidate-set",
        choices=("grid", "selected", "ep_recheck"),
        required=True,
    )
    evaluate.add_argument("--budget", type=int, required=True)
    evaluate.add_argument("--snrs", required=True)
    evaluate.add_argument("--blocks-per-snr", required=True)
    evaluate.add_argument("--batch", type=int, default=64)
    evaluate.add_argument("--seed", type=int, default=20260802)
    evaluate.add_argument("--ep-alpha", type=float, default=0.01)
    evaluate.add_argument("--delta", type=float, default=0.02)
    evaluate.add_argument("--rho", type=float, default=0.9)
    evaluate.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    evaluate.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_evaluate(arguments)
