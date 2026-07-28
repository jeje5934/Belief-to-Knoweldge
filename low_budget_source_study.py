"""Readout calibration and budget-aware source-decoder tuning.

Experiment harness only. Production ``decoder.py`` is imported unchanged.

Fair early-stop rule used by the budget study:
    a codeword succeeds at the first chunk whose full information block passes
    CRC24A.  The same rule is applied to legacy, EP, and altproj from the
    decoder's existing ``last_u_hat_hist`` diagnostic hook.
"""

from __future__ import annotations

import argparse
import gc
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

from altproj_compression_study import (
    _crc_check_callable,
)
from denoiser_sigma_alignment_diag import (
    FIXED_SIGMA,
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
    build_decoder,
    configure_candidate,
    dump_json,
)
from syndrome_sigma_schedule import AnnealingSigmaScheduler


ACCURACY_MSB_TO_LSB = np.asarray(
    [0.949, 0.814, 0.683, 0.677, 0.666, 0.635, 0.680, 0.731],
    dtype=np.float64,
)
MASKED_PLANES = (2, 3, 4, 5)

SCHEDULES = {
    100: {
        "5x20": [5] * 20,
        "10x10": [10] * 10,
    },
    50: {
        "5x10": [5] * 10,
        "10x5": [10] * 5,
    },
    30: {
        "5x6": [5] * 6,
        "10x3": [10] * 3,
        "15x2": [15] * 2,
    },
    20: {
        "5x4": [5] * 4,
        "10x2": [10] * 2,
    },
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def geom_scheduler(schedule, endpoint):
    scheduler = AnnealingSigmaScheduler(
        FIXED_SIGMA,
        float(endpoint),
        len(schedule) - 1,
        mode="geom",
    )
    return scheduler, tuple(float(value) for value in scheduler.path)


def exact_paired_binomial(broken, rescued):
    """Two-sided exact sign-test p-value for discordant paired outcomes."""
    n = int(broken + rescued)
    if n == 0:
        return 1.0
    tail = min(int(broken), int(rescued))
    probability = sum(math.comb(n, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2.0**n))


def linear_knee(rows, target=0.1):
    """Linear-in-BLER interpolation used by the compression baseline."""
    ordered = sorted(rows, key=lambda row: row["esn0_db"])
    for left, right in zip(ordered[:-1], ordered[1:]):
        y0, y1 = left["crc_bler"], right["crc_bler"]
        if (y0 - target) * (y1 - target) > 0 or y0 == y1:
            continue
        x0, x1 = left["esn0_db"], right["esn0_db"]
        return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return None


def mask_summary(success, reference=None):
    success = np.asarray(success, dtype=bool)
    failures = int(np.sum(~success))
    result = {
        "blocks": int(len(success)),
        "failures": failures,
        "crc_bler": float(failures / len(success)),
        "wilson_95": wilson(failures, len(success)),
        "success_mask": success.tolist(),
    }
    if reference is not None:
        reference = np.asarray(reference, dtype=bool)
        broken = int(np.sum(reference & ~success))
        rescued = int(np.sum(~reference & success))
        result["paired_vs_reference"] = {
            "broken": broken,
            "rescued": rescued,
            "net_failure_change": broken - rescued,
            "both_failure": int(np.sum(~reference & ~success)),
            "both_success": int(np.sum(reference & success)),
            "exact_two_sided_p": exact_paired_binomial(broken, rescued),
        }
    return result


class ReadoutScheduleProxy(tf.keras.layers.Layer):
    """Altproj sigma schedule plus experiment-only bit-plane readout controls."""

    def __init__(self, inner):
        super().__init__(name="readout_sigma_schedule_proxy")
        self.inner = inner
        self.override = False
        self.path = ()
        self.call_index = 0
        self.mode = "global"
        self.global_sigma_post = 3.0
        self.plane_stds = tuple(
            3.0 * float(np.max(ACCURACY_MSB_TO_LSB)) / ACCURACY_MSB_TO_LSB
        )

    def reset(self, override, path):
        self.override = bool(override)
        self.path = tuple(float(value) for value in path)
        self.call_index = 0

    def set_readout(self, mode, sigma_post=3.0):
        if mode not in ("global", "accuracy_inverse", "mask_b2_b5"):
            raise ValueError(mode)
        self.mode = mode
        self.global_sigma_post = float(sigma_post)
        self.inner.sigma_post = float(sigma_post)

    @property
    def sigma_post(self):
        return self.inner.sigma_post

    def _sigma_arg(self, sigma, batch):
        if sigma is None:
            value = self.inner.sigma
        else:
            value = sigma
        if isinstance(value, tf.Tensor):
            value = value.numpy()
        tensor = torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self.inner._device,
        ).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(batch)
        if tensor.numel() != batch:
            raise ValueError("sigma must be scalar or have one value per block")
        return tensor

    def _accuracy_inverse_call(self, input_llr, sigma):
        llr = torch.from_numpy(input_llr.numpy()).float().to(
            self.inner._device
        )
        prior = self.inner.prior_model
        with torch.no_grad():
            mu_cavity = prior.llr_to_soft_field(llr)
            sigma_tensor = self._sigma_arg(sigma, llr.shape[0])
            mu_projected = prior.net(mu_cavity, sigma_tensor).clamp(0.0, 1.0)
            selected = []
            for plane, plane_std in enumerate(self.plane_stds):
                logits = prior.soft_field_to_posterior_logits(
                    mu_projected,
                    posterior_pixel_std=float(plane_std),
                ).reshape(llr.shape[0], prior.n_pixels, prior.bpp)
                selected.append(logits[..., plane])
            projected = torch.stack(selected, dim=-1).reshape_as(llr)
            extrinsic = projected - llr
        return tf.constant(
            extrinsic.cpu().numpy(),
            dtype=input_llr.dtype,
        )

    def call(self, input_llr, sigma=None):
        if self.override:
            if self.call_index >= len(self.path):
                raise RuntimeError("altproj consumed too many sigma values")
            sigma = self.path[self.call_index]
            self.call_index += 1
        if self.mode == "accuracy_inverse":
            return self._accuracy_inverse_call(input_llr, sigma)
        result = self.inner(input_llr, sigma=sigma)
        if self.mode == "mask_b2_b5":
            shaped = tf.reshape(result, [-1, 28 * 28, 8])
            mask = np.ones(8, dtype=np.float32)
            mask[list(MASKED_PLANES)] = 0.0
            result = tf.reshape(
                shaped * tf.constant(mask, dtype=shaped.dtype)[None, None, :],
                tf.shape(result),
            )
        return result


def build_experiment_decoder(ldpc, schedule, checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    last_error = None
    for attempt in range(2):
        try:
            decoder, old_proxy, _, _ = build_decoder(
                ldpc,
                checkpoint,
                schedule,
                device,
            )
            break
        except (OSError, RuntimeError, SystemError) as error:
            last_error = error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt:
                raise
    else:
        raise last_error
    proxy = ReadoutScheduleProxy(old_proxy.inner)
    decoder._denoiser = proxy
    decoder._ep_config_logged = True
    decoder.track_u_hat = True
    return decoder, proxy


def first_crc_success(history, final_logits, crc_decoder, schedule, batch):
    captured = np.zeros(batch, dtype=bool)
    iterations = np.full(batch, float(sum(schedule)), dtype=np.float64)
    cumulative = np.cumsum(schedule)
    tensors = list(history)
    if not tensors:
        tensors = [final_logits]
        cumulative = np.asarray([sum(schedule)])
    for chunk_index, info_logits in enumerate(tensors):
        _, valid = crc_decoder(info_logits)
        valid_np = np.asarray(valid.numpy()).reshape(-1).astype(bool)
        newly = valid_np & ~captured
        iterations[newly] = float(cumulative[min(chunk_index, len(cumulative) - 1)])
        captured |= valid_np
    return captured, iterations


def configure(
    decoder,
    proxy,
    candidate,
    schedule,
    endpoint,
    readout,
):
    scheduler, path = geom_scheduler(schedule, endpoint)
    configure_candidate(
        decoder,
        proxy,
        scheduler,
        path,
        candidate,
    )
    proxy.set_readout(
        readout["mode"],
        readout.get("sigma_post", 3.0),
    )
    decoder.track_u_hat = True
    decoder.altproj_early_stop = False
    return path


def candidate_name(candidate, schedule_name, endpoint):
    if candidate["scheme"] == "altproj":
        knob = f"d{candidate['delta']:g}_r{candidate['rho']:g}"
    elif candidate["scheme"] == "ep":
        knob = f"a{candidate['alpha_ep']:g}"
    else:
        knob = f"a{candidate['alpha']:g}"
    return (
        f"{candidate['scheme']}_{schedule_name}_{knob}"
        f"_send{float(endpoint):g}"
    )


def candidate_grid(scheme, endpoint, rho_values=(0.9,)):
    if scheme == "altproj":
        return [
            {
                "name": f"altproj_d{delta:g}_r{rho:g}",
                "scheme": "altproj",
                "delta": float(delta),
                "rho": float(rho),
            }
            for delta in (0.02, 0.05, 0.1, 0.2)
            for rho in rho_values
        ]
    if scheme == "ep":
        return [
            {
                "name": f"ep_a{alpha:g}",
                "scheme": "ep",
                "alpha_ep": float(alpha),
            }
            for alpha in (0.01, 0.02, 0.05, 0.1)
        ]
    if scheme == "legacy":
        return [
            {
                "name": f"legacy_a{alpha:g}",
                "scheme": "legacy",
                "alpha": float(alpha),
            }
            for alpha in (0.1, 0.2, 0.3)
        ]
    raise ValueError(scheme)


def readout_from_a(path):
    if path is None:
        return {
            "name": "global_sigma_post_3",
            "mode": "global",
            "sigma_post": 3.0,
        }
    payload = read_json(path)
    return dict(payload["selection"]["readout"])


def run_masks_for_configs(
    *,
    configs,
    schedule,
    readout,
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
    decoder, proxy = build_experiment_decoder(
        ldpc,
        schedule,
        checkpoint,
    )
    masks = {config["name"]: [] for config in configs}
    iterations = {config["name"]: [] for config in configs}
    for round_index in range(blocks // batch):
        _, _, _, channel_llr = make_channel_batch(
            round_index,
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
        for config in configs:
            configure(
                decoder,
                proxy,
                config["candidate"],
                schedule,
                config["sigma_end"],
                readout,
            )
            logits = decoder(channel_llr)
            success, used = first_crc_success(
                decoder.last_u_hat_hist,
                logits,
                crc_decoder,
                schedule,
                batch,
            )
            masks[config["name"]].extend(success.tolist())
            iterations[config["name"]].extend(used.tolist())
        print(
            f"Es/N0={esn0_db:+.2f} budget={sum(schedule)} "
            f"round {round_index + 1}/{blocks // batch}",
            flush=True,
        )
    return masks, iterations


def make_result_row(success, iterations, config, reference=None):
    row = mask_summary(success, reference)
    values = np.asarray(iterations, dtype=np.float64)
    row["actual_mean_bp_iters"] = float(np.mean(values))
    row["actual_iteration_distribution"] = {
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.1)),
        "q90": float(np.quantile(values, 0.9)),
    }
    row["config"] = config
    return row


def run_sigma_post(args):
    schedule = SCHEDULES[100]["5x20"]
    (
        crc_encoder,
        crc_decoder,
        ldpc,
        mapper,
        demapper,
        awgn,
    ) = make_system()
    images, bit_bank = load_fashion_mnist()
    decoder, proxy = build_experiment_decoder(
        ldpc,
        schedule,
        args.checkpoint,
    )
    decoder.altproj_crc_check = _crc_check_callable(crc_decoder)
    candidate = {
        "name": "altproj_d0.02_r0.9",
        "scheme": "altproj",
        "delta": 0.02,
        "rho": 0.9,
    }
    initial = [
        {
            "name": f"global_sigma_post_{value:g}",
            "mode": "global",
            "sigma_post": value,
        }
        for value in (1.5, 3.0, 6.0)
    ] + [
        {
            "name": "accuracy_inverse",
            "mode": "accuracy_inverse",
            "sigma_post": 3.0,
        },
        {
            "name": "mask_b2_b5",
            "mode": "mask_b2_b5",
            "sigma_post": 3.0,
        },
    ]

    def evaluate(readouts):
        masks = {item["name"]: [] for item in readouts}
        iterations = {item["name"]: [] for item in readouts}
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
            for item in readouts:
                scheduler, path = geom_scheduler(schedule, 0.05)
                configure_candidate(
                    decoder,
                    proxy,
                    scheduler,
                    path,
                    candidate,
                )
                proxy.set_readout(item["mode"], item["sigma_post"])
                decoder.altproj_early_stop = True
                decoder.altproj_crc_check = _crc_check_callable(crc_decoder)
                decoder.track_u_hat = True
                logits = decoder(channel_llr)
                _, valid = crc_decoder(logits)
                success = np.asarray(valid.numpy()).reshape(-1).astype(bool)
                masks[item["name"]].extend(success.tolist())
                _, used = first_crc_success(
                    decoder.last_u_hat_hist,
                    logits,
                    crc_decoder,
                    schedule,
                    args.batch,
                )
                iterations[item["name"]].extend(used.tolist())
            print(
                f"sigma_post round {round_index + 1}/"
                f"{args.blocks // args.batch}",
                flush=True,
            )
        return masks, iterations

    if args.blocks % args.batch:
        raise ValueError("blocks must be divisible by batch")
    masks, iterations = evaluate(initial)
    global_names = [item["name"] for item in initial[:3]]
    global_best = min(
        global_names,
        key=lambda name: int(np.sum(~np.asarray(masks[name], dtype=bool))),
    )
    extra = None
    if global_best.endswith("_1.5"):
        extra = {
            "name": "global_sigma_post_0.75",
            "mode": "global",
            "sigma_post": 0.75,
        }
    elif global_best.endswith("_6"):
        extra = {
            "name": "global_sigma_post_12",
            "mode": "global",
            "sigma_post": 12.0,
        }
    if extra is not None:
        extra_masks, extra_iterations = evaluate([extra])
        masks.update(extra_masks)
        iterations.update(extra_iterations)
        initial.append(extra)

    reference = np.asarray(masks["global_sigma_post_3"], dtype=bool)
    rows = {}
    for item in initial:
        rows[item["name"]] = make_result_row(
            masks[item["name"]],
            iterations[item["name"]],
            item,
            None if item["name"] == "global_sigma_post_3" else reference,
        )

    best_name = min(rows, key=lambda name: rows[name]["failures"])
    paired = rows[best_name].get("paired_vs_reference")
    significant = (
        best_name != "global_sigma_post_3"
        and rows[best_name]["failures"]
        < rows["global_sigma_post_3"]["failures"]
        and paired is not None
        and paired["exact_two_sided_p"] < 0.05
    )
    selected_name = best_name if significant else "global_sigma_post_3"
    result = {
        "kind": "altproj_sigma_post_closure",
        "configuration": {
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "scheme": "altproj",
            "delta": 0.02,
            "rho": 0.9,
            "bp_schedule": schedule,
            "sigma_start": 0.3,
            "sigma_end": 0.05,
            "crc_early_stop": True,
            "accuracy_msb_to_lsb": ACCURACY_MSB_TO_LSB.tolist(),
            "accuracy_inverse_plane_sigma_post": proxy.plane_stds,
            "masked_plane_indices_msb_first": MASKED_PLANES,
            "paired_payload_and_noise": True,
        },
        "rows": rows,
        "selection": {
            "raw_best": best_name,
            "significant_vs_sigma_post_3": significant,
            "criterion": (
                "lower paired failures and exact two-sided sign-test p<0.05"
            ),
            "selected": selected_name,
            "readout": rows[selected_name]["config"],
        },
    }
    dump_json(args.output, result)


def coarse_configs(schedule_name):
    configs = []
    for scheme in ("altproj", "ep", "legacy"):
        for candidate in candidate_grid(scheme, 0.05, rho_values=(0.9,)):
            name = candidate_name(candidate, schedule_name, 0.05)
            configs.append(
                {
                    "name": name,
                    "candidate": candidate,
                    "sigma_end": 0.05,
                }
            )
    return configs


def run_coarse(args):
    budgets = [int(value) for value in args.budgets.split(",")]
    readout = readout_from_a(args.a_result)
    result = {
        "kind": "budget_scheme_coarse_screen",
        "configuration": {
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "readout": readout,
            "sigma_start": 0.3,
            "sigma_end": 0.05,
            "altproj_rho": 0.9,
            "legacy_beta": LEGACY_BETA_BP_EXTRINSIC_BOOST,
            "ep_beta_ep": EP_BETA_CODE_SITE,
            "crc_early_stop": (
                "universal first CRC-pass over existing per-chunk history"
            ),
            "paired_within_each_schedule": True,
        },
        "budgets": {},
    }
    for budget in budgets:
        budget_rows = {}
        for schedule_name, schedule in SCHEDULES[budget].items():
            configs = coarse_configs(schedule_name)
            masks, iterations = run_masks_for_configs(
                configs=configs,
                schedule=schedule,
                readout=readout,
                esn0_db=args.esn0_db,
                blocks=args.blocks,
                batch=args.batch,
                seed=args.seed + budget * 100,
                checkpoint=args.checkpoint,
            )
            rows = {
                config["name"]: make_result_row(
                    masks[config["name"]],
                    iterations[config["name"]],
                    config,
                )
                for config in configs
            }
            budget_rows[schedule_name] = {
                "schedule": schedule,
                "rows": rows,
            }
            result["budgets"][str(budget)] = budget_rows
            dump_json(args.output, result)
    dump_json(args.output, result)


def select_schedule_by_scheme(coarse, budget, scheme):
    candidates = []
    for schedule_name, schedule_row in coarse["budgets"][str(budget)].items():
        scheme_rows = [
            row
            for row in schedule_row["rows"].values()
            if row["config"]["candidate"]["scheme"] == scheme
        ]
        best = min(
            scheme_rows,
            key=lambda row: (
                row["failures"],
                row["actual_mean_bp_iters"],
            ),
        )
        candidates.append((best, schedule_name, schedule_row["schedule"]))
    return min(
        candidates,
        key=lambda item: (
            item[0]["failures"],
            item[0]["actual_mean_bp_iters"],
            len(item[2]),
        ),
    )


def refined_configs(scheme, schedule_name):
    configs = []
    rho_values = (0.9, 0.95) if scheme == "altproj" else (0.9,)
    for endpoint in (0.05, 0.02):
        for candidate in candidate_grid(
            scheme,
            endpoint,
            rho_values=rho_values,
        ):
            name = candidate_name(candidate, schedule_name, endpoint)
            configs.append(
                {
                    "name": name,
                    "candidate": candidate,
                    "sigma_end": endpoint,
                }
            )
    return configs


def run_refine(args):
    coarse = read_json(args.coarse)
    coarse_b20 = (
        read_json(args.coarse_b20)
        if args.coarse_b20 is not None
        else None
    )
    budgets = [int(value) for value in args.budgets.split(",")]
    readout = readout_from_a(args.a_result)
    initial_result = {
        "kind": "budget_scheme_refined_screen",
        "configuration": {
            "esn0_db": args.esn0_db,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "readout": readout,
            "sigma_start": 0.3,
            "sigma_endpoints": [0.05, 0.02],
            "legacy_beta": LEGACY_BETA_BP_EXTRINSIC_BOOST,
            "ep_beta_ep": EP_BETA_CODE_SITE,
            "crc_early_stop": (
                "universal first CRC-pass over existing per-chunk history"
            ),
            "reduction": (
                "coarse screen chooses one schedule per budget and scheme; "
                "full strength/endpoint grid is rerun on that schedule"
            ),
            "budget20_schedule_source": (
                args.coarse_b20
                if coarse_b20 is not None
                else args.coarse
            ),
        },
        "budgets": {},
        "best_by_budget": {},
    }
    result = (
        read_json(args.output)
        if args.resume and Path(args.output).exists()
        else initial_result
    )
    for budget in budgets:
        scheme_results = result["budgets"].get(str(budget), {})
        all_rows = {}
        for scheme in ("altproj", "ep", "legacy"):
            if scheme in scheme_results:
                all_rows.update(scheme_results[scheme]["rows"])
                continue
            schedule_source = (
                coarse_b20
                if budget == 20 and coarse_b20 is not None
                else coarse
            )
            coarse_best, schedule_name, schedule = select_schedule_by_scheme(
                schedule_source,
                budget,
                scheme,
            )
            configs = refined_configs(scheme, schedule_name)
            masks, iterations = run_masks_for_configs(
                configs=configs,
                schedule=schedule,
                readout=readout,
                esn0_db=args.esn0_db,
                blocks=args.blocks,
                batch=args.batch,
                seed=args.seed + budget * 100,
                checkpoint=args.checkpoint,
            )
            rows = {
                config["name"]: make_result_row(
                    masks[config["name"]],
                    iterations[config["name"]],
                    config,
                )
                for config in configs
            }
            scheme_best_name = min(
                rows,
                key=lambda name: (
                    rows[name]["failures"],
                    rows[name]["actual_mean_bp_iters"],
                    0 if rows[name]["config"]["sigma_end"] == 0.05 else 1,
                ),
            )
            scheme_results[scheme] = {
                "coarse_selected_schedule": schedule_name,
                "schedule": schedule,
                "coarse_best": coarse_best,
                "rows": rows,
                "best": rows[scheme_best_name],
            }
            all_rows.update(rows)
            result["budgets"][str(budget)] = scheme_results
            dump_json(args.output, result)
            del masks, iterations
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        best_name = min(
            all_rows,
            key=lambda name: (
                all_rows[name]["failures"],
                all_rows[name]["actual_mean_bp_iters"],
                {"altproj": 0, "ep": 1, "legacy": 2}[
                    all_rows[name]["config"]["candidate"]["scheme"]
                ],
            ),
        )
        result["best_by_budget"][str(budget)] = all_rows[best_name]
        dump_json(args.output, result)


def load_best_config(path, budget):
    result = read_json(path)
    return result["best_by_budget"][str(int(budget))]


def run_waterfall(args):
    if len(args.esn0) != len(args.blocks):
        raise ValueError("esn0 and blocks lengths differ")
    budget = int(args.budget)
    best = load_best_config(args.refined, budget)
    config = best["config"]
    schedule_name = config["name"].split("_")[1]
    schedule = SCHEDULES[budget][schedule_name]
    readout = readout_from_a(args.a_result)
    initial_result = {
        "kind": "budget_best_source_waterfall",
        "configuration": {
            "budget": budget,
            "schedule_name": schedule_name,
            "schedule": schedule,
            "candidate": config["candidate"],
            "sigma_start": 0.3,
            "sigma_end": config["sigma_end"],
            "readout": readout,
            "legacy_beta": LEGACY_BETA_BP_EXTRINSIC_BOOST,
            "ep_beta_ep": EP_BETA_CODE_SITE,
            "crc_early_stop": (
                "universal first CRC-pass over existing per-chunk history"
            ),
            "batch": args.batch,
            "seed": args.seed,
        },
        "points": {},
        "knees": {"bler_0.1": None, "bler_0.01": None},
    }
    result = (
        read_json(args.output)
        if args.resume and Path(args.output).exists()
        else initial_result
    )
    for point_index, (esn0_db, blocks) in enumerate(
        zip(args.esn0, args.blocks)
    ):
        point_key = f"{esn0_db:.3f}"
        if point_key in result["points"]:
            continue
        runner_config = {
            "name": config["name"],
            "candidate": config["candidate"],
            "sigma_end": config["sigma_end"],
        }
        masks, iterations = run_masks_for_configs(
            configs=[runner_config],
            schedule=schedule,
            readout=readout,
            esn0_db=esn0_db,
            blocks=blocks,
            batch=args.batch,
            seed=args.seed + point_index * 10000,
            checkpoint=args.checkpoint,
        )
        row = make_result_row(
            masks[config["name"]],
            iterations[config["name"]],
            runner_config,
        )
        row.pop("success_mask")
        row["esn0_db"] = float(esn0_db)
        result["points"][point_key] = row
        rows = list(result["points"].values())
        result["knees"]["bler_0.1"] = linear_knee(rows, target=0.1)
        result["knees"]["bler_0.01"] = linear_knee(rows, target=0.01)
        dump_json(args.output, result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    sub = parser.add_subparsers(dest="mode", required=True)

    sigma = sub.add_parser("sigma-post")
    sigma.add_argument("--esn0-db", type=float, default=-2.85)
    sigma.add_argument("--blocks", type=int, default=1024)
    sigma.add_argument("--batch", type=int, default=64)
    sigma.add_argument("--seed", type=int, default=20260820)
    sigma.add_argument("--output", required=True)

    coarse = sub.add_parser("coarse")
    coarse.add_argument("--budgets", default="100,50,30,20")
    coarse.add_argument("--esn0-db", type=float, default=-2.85)
    coarse.add_argument("--blocks", type=int, default=512)
    coarse.add_argument("--batch", type=int, default=64)
    coarse.add_argument("--seed", type=int, default=20260821)
    coarse.add_argument("--a-result")
    coarse.add_argument("--output", required=True)

    refine = sub.add_parser("refine")
    refine.add_argument("--coarse", required=True)
    refine.add_argument(
        "--coarse-b20",
        help=(
            "optional non-saturated budget-20 coarse result used only for "
            "schedule selection"
        ),
    )
    refine.add_argument("--budgets", default="100,50,30,20")
    refine.add_argument("--esn0-db", type=float, default=-2.85)
    refine.add_argument("--blocks", type=int, default=512)
    refine.add_argument("--batch", type=int, default=64)
    refine.add_argument("--seed", type=int, default=20260822)
    refine.add_argument("--a-result")
    refine.add_argument("--resume", action="store_true")
    refine.add_argument("--output", required=True)

    waterfall = sub.add_parser("waterfall")
    waterfall.add_argument("--refined", required=True)
    waterfall.add_argument("--budget", type=int, required=True)
    waterfall.add_argument("--esn0", type=float, nargs="+", required=True)
    waterfall.add_argument("--blocks", type=int, nargs="+", required=True)
    waterfall.add_argument("--batch", type=int, default=64)
    waterfall.add_argument("--seed", type=int, default=20260823)
    waterfall.add_argument("--a-result")
    waterfall.add_argument("--resume", action="store_true")
    waterfall.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "sigma-post":
        run_sigma_post(args)
    elif args.mode == "coarse":
        run_coarse(args)
    elif args.mode == "refine":
        run_refine(args)
    elif args.mode == "waterfall":
        run_waterfall(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
