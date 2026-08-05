#!/usr/bin/env python3
"""Reduced altproj integration study after the intermediate multiview gate.

Production decoder.py is imported unchanged. A proxy around the already-loaded
denoiser adds either an explicit side pull or fuses the aligned side image into
the denoiser input.
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
from scipy.ndimage import affine_transform

tf.config.set_visible_devices([], "GPU")

import torch

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
from denoiser_sigma_three_scheme import DEFAULT_CHECKPOINT, build_decoder
from low_budget_source_study import geom_scheduler
from multiview_gate import STRENGTHS, make_views
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder


SCHEDULE = [5] * 20
SOURCE_CALLS = len(SCHEDULE) - 1
PIXELS = IMG_H * IMG_W
BIT_WEIGHTS = np.asarray([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.float64)


def estimate_alignment_grid(side, target, strength):
    """Register side to a first-BP hard image using receiver-visible data only."""
    angles = np.linspace(-strength.max_angle, strength.max_angle, 7)
    shifts = np.linspace(-strength.max_shift, strength.max_shift, 5)
    center = np.asarray([(IMG_H - 1) / 2.0, (IMG_W - 1) / 2.0])
    aligned = np.empty_like(side, dtype=np.float32)
    estimates = []
    for index, (image, reference) in enumerate(zip(side, target)):
        best = None
        for angle in angles:
            theta = math.radians(float(angle))
            c, s = math.cos(theta), math.sin(theta)
            rotation = np.asarray([[c, s], [-s, c]], dtype=np.float64)
            for tx in shifts:
                for ty in shifts:
                    translation = np.asarray([ty, tx], dtype=np.float64)
                    offset = center + translation - rotation @ center
                    raw = affine_transform(
                        image, rotation, offset=offset,
                        output_shape=(IMG_H, IMG_W), order=1,
                        mode="constant", cval=0.0, prefilter=False,
                    )
                    denom = float(np.sum(raw * raw)) + 1.0e-12
                    scale = float(np.clip(np.sum(raw * reference) / denom,
                                          0.5, 2.0))
                    candidate = np.clip(scale * raw, 0.0, 1.0)
                    loss = float(np.mean((candidate - reference) ** 2))
                    if best is None or loss < best[0]:
                        best = (loss, candidate, float(tx), float(ty),
                                float(angle), scale)
        _, candidate, tx, ty, angle, scale = best
        aligned[index] = candidate
        estimates.append({
            "tx": tx, "ty": ty, "angle_deg": angle,
            "brightness": float(1.0 / scale), "fit_mse": float(best[0]),
        })
    return aligned, estimates


class MultiViewSigmaProxy(tf.keras.layers.Layer):
    """Experiment-only side correction preserving decoder.py unchanged."""

    def __init__(self, inner):
        super().__init__(name="multiview_sigma_proxy")
        self.inner = inner
        self.override = True
        self.path: tuple[float, ...] = ()
        self.call_index = 0
        self.mode = "none"
        self.delta = 0.02
        self.delta_side = 0.0
        self.side_gate = "constant"
        self.fusion_weight = 0.0
        self.side_pt = None
        self.side_logits_tf = None

    @property
    def sigma_post(self):
        return self.inner.sigma_post

    def reset(self, override, path):
        self.override = bool(override)
        self.path = tuple(float(value) for value in path)
        self.call_index = 0

    def configure(self, config, path, aligned_side):
        self.mode = config["mode"]
        self.delta = float(config["delta"])
        self.delta_side = float(config.get("delta_side", 0.0))
        self.side_gate = config.get("side_gate", "constant")
        self.fusion_weight = float(config.get("fusion_weight", 0.0))
        self.reset(True, path)
        device = self.inner._device
        self.side_pt = torch.from_numpy(aligned_side[:, None]).float().to(device)
        prior = self.inner.prior_model
        with torch.no_grad():
            side_logits = prior.soft_field_to_posterior_logits(self.side_pt)
        self.side_logits_tf = tf.constant(
            side_logits.cpu().numpy(), dtype=tf.float32
        )

    def _sigma_tensor(self, sigma, batch):
        tensor = torch.as_tensor(
            sigma, dtype=torch.float32, device=self.inner._device
        ).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(batch)
        return tensor

    def call(self, input_llr, sigma=None):
        if self.override:
            if self.call_index >= len(self.path):
                raise RuntimeError("altproj consumed more source calls than sigma path")
            sigma = self.path[self.call_index]
        sigma_value = float(np.asarray(sigma).reshape(-1)[0])
        self.call_index += 1

        if self.mode == "input_fusion":
            prior = self.inner.prior_model
            llr_pt = torch.from_numpy(input_llr.numpy()).float().to(
                self.inner._device
            )
            with torch.no_grad():
                cavity = prior.llr_to_soft_field(llr_pt)
                mixed = (
                    (1.0 - self.fusion_weight) * cavity
                    + self.fusion_weight * self.side_pt
                )
                projected = prior.net(
                    mixed, self._sigma_tensor(sigma, len(llr_pt))
                ).clamp(0.0, 1.0)
                projected_llr = prior.soft_field_to_posterior_logits(projected)
                extrinsic = projected_llr - llr_pt
            return tf.constant(extrinsic.cpu().numpy(), dtype=input_llr.dtype)

        denoiser_pull = self.inner(input_llr, sigma=sigma)
        if self.mode == "none" or self.delta_side == 0.0:
            return denoiser_pull
        active = self.side_gate == "constant" or (
            self.side_gate == "sigma_ge_0.1" and sigma_value >= 0.1
        )
        if not active:
            return denoiser_pull
        side_pull = tf.cast(self.side_logits_tf, input_llr.dtype) - input_llr
        # decoder multiplies the proxy output by delta, so this realizes
        # delta*D + delta_side*S without touching decoder.py.
        return denoiser_pull + tf.cast(
            self.delta_side / self.delta, input_llr.dtype
        ) * side_pull


def config_name(config):
    if config["mode"] == "none":
        return f"none_d{config['delta']:g}_e{config['endpoint']:g}"
    if config["mode"] == "side_pull":
        return (
            f"pull_d{config['delta']:g}_ds{config['delta_side']:g}_"
            f"{config['side_gate']}_e{config['endpoint']:g}"
        )
    return (
        f"input_d{config['delta']:g}_w{config['fusion_weight']:g}_"
        f"e{config['endpoint']:g}"
    )


def screen_configs():
    configs = [
        {"mode": "none", "delta": 0.02, "rho": 0.9, "endpoint": 0.05},
    ]
    for delta in (0.02, 0.05):
        for delta_side in (0.002, 0.005, 0.01, 0.02):
            for gate in ("constant", "sigma_ge_0.1"):
                configs.append(
                    {
                        "mode": "side_pull",
                        "delta": delta,
                        "delta_side": delta_side,
                        "side_gate": gate,
                        "rho": 0.9,
                        "endpoint": 0.05,
                    }
                )
        for weight in (0.25, 0.5, 0.75):
            configs.append(
                {
                    "mode": "input_fusion",
                    "delta": delta,
                    "fusion_weight": weight,
                    "rho": 0.9,
                    "endpoint": 0.05,
                }
            )
    for config in configs:
        config["name"] = config_name(config)
    return configs


def confirm_configs():
    # Filled from the 32-block screen. Delta=0.05 and direct input fusion broke
    # successful blocks, so confirmation concentrates on the non-destructive
    # delta=0.02 early-side family and checks all requested sigma endpoints.
    configs = [
        {"mode": "none", "delta": 0.02, "rho": 0.9, "endpoint": 0.05},
        {"mode": "side_pull", "delta": 0.02, "delta_side": 0.005,
         "side_gate": "sigma_ge_0.1", "rho": 0.9, "endpoint": 0.05},
        {"mode": "side_pull", "delta": 0.02, "delta_side": 0.01,
         "side_gate": "sigma_ge_0.1", "rho": 0.9, "endpoint": 0.05},
        {"mode": "side_pull", "delta": 0.02, "delta_side": 0.02,
         "side_gate": "sigma_ge_0.1", "rho": 0.9, "endpoint": 0.05},
        {"mode": "side_pull", "delta": 0.02, "delta_side": 0.002,
         "side_gate": "constant", "rho": 0.9, "endpoint": 0.05},
        {"mode": "side_pull", "delta": 0.02, "delta_side": 0.01,
         "side_gate": "sigma_ge_0.1", "rho": 0.9, "endpoint": 0.02},
        {"mode": "side_pull", "delta": 0.02, "delta_side": 0.01,
         "side_gate": "sigma_ge_0.1", "rho": 0.9, "endpoint": 0.1},
        {"mode": "input_fusion", "delta": 0.02, "fusion_weight": 0.5,
         "rho": 0.9, "endpoint": 0.05},
    ]
    for config in configs:
        config["name"] = config_name(config)
    return configs


def waterfall_configs():
    """Independent-seed confirmation of the two non-destructive screen winners."""
    configs = [
        {"mode": "none", "delta": 0.02, "rho": 0.9, "endpoint": 0.05},
        {"mode": "side_pull", "delta": 0.02, "delta_side": 0.01,
         "side_gate": "sigma_ge_0.1", "rho": 0.9, "endpoint": 0.02},
    ]
    for config in configs:
        config["name"] = config_name(config)
    return configs


def sigma_control_configs():
    configs = [
        {"mode": "none", "delta": 0.02, "rho": 0.9, "endpoint": 0.02},
        {"mode": "none", "delta": 0.02, "rho": 0.9, "endpoint": 0.1},
    ]
    for config in configs:
        config["name"] = config_name(config)
    return configs


def summarize_metrics(accum, seed):
    crc = np.asarray(accum["crc_valid"], dtype=bool)
    mse = np.asarray(accum["hard_mse_255"], dtype=np.float64) / 255.0**2
    failures = int(np.sum(~crc))
    mean_mse = float(np.mean(mse))
    return {
        "blocks": int(len(crc)),
        "failures": failures,
        "crc_bler": float(failures / len(crc)),
        "wilson_95": wilson(failures, len(crc)),
        "mean_mse_01": mean_mse,
        "aggregate_psnr_db": (
            None if mean_mse == 0.0 else float(-10.0 * np.log10(mean_mse))
        ),
        "mse_quantiles": {
            "p50": float(np.quantile(mse, 0.5)),
            "p90": float(np.quantile(mse, 0.9)),
            "p99": float(np.quantile(mse, 0.99)),
        },
        "success_mask": crc.tolist(),
        "seed": seed,
    }


def metrics_from_logits(logits, truth_bits, truth_images, crc_decoder):
    payload = np.asarray(logits.numpy())[:, :K_PAYLOAD]
    hard_bits = payload.reshape(-1, PIXELS, BPP) > 0.0
    hard_image = np.sum(hard_bits * BIT_WEIGHTS[None, None, :], axis=2)
    _, valid = hard_crc_decode(crc_decoder, logits)
    return {
        "crc_valid": np.asarray(valid.numpy()).reshape(-1).astype(bool).tolist(),
        "hard_mse_255": np.mean(
            (hard_image - truth_images.reshape(-1, PIXELS) * 255.0) ** 2,
            axis=1,
        ).tolist(),
        "bit_errors": np.sum(
            hard_bits.reshape(-1, K_PAYLOAD)
            != np.asarray(truth_bits).reshape(-1, K_PAYLOAD).astype(bool),
            axis=1,
        ).astype(int).tolist(),
    }


def paired(candidate, reference):
    cand = np.asarray(candidate["success_mask"], dtype=bool)
    ref = np.asarray(reference["success_mask"], dtype=bool)
    return {
        "broken": int(np.sum(ref & ~cand)),
        "rescued": int(np.sum(~ref & cand)),
        "net_failure_change": int(np.sum(ref & ~cand) - np.sum(~ref & cand)),
    }


def run(args, configs):
    images, bit_bank = load_fashion_mnist()
    crc_encoder, crc_decoder, ldpc, mapper, demapper, awgn = make_system()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, old_proxy, _, _ = build_decoder(
        ldpc, args.checkpoint, SCHEDULE, device
    )
    proxy = MultiViewSigmaProxy(old_proxy.inner)
    decoder._denoiser = proxy
    decoder._ep_config_logged = True
    decoder.track_u_hat = args.registration == "estimated"
    pure_decoder = None
    if not args.skip_bp:
        pure_decoder = LDPC5GDecoder(
            ldpc,
            cn_update="boxplus-phi",
            vn_update="sum",
            cn_schedule="flooding",
            hard_out=False,
            return_infobits=True,
            num_iter=100,
            llr_max=30.0,
        )
    strength = next(value for value in STRENGTHS if value.name == args.strength)

    result = {
        "kind": f"multiview_altproj_{args.mode}",
        "configuration": {
            "branch": "codex/multiview",
            "channel": "BPSK/AWGN/perfect_CSI",
            "payload_bits": K_PAYLOAD,
            "n": N_CODEWORD,
            "rate": K_PAYLOAD / N_CODEWORD,
            "schedule": SCHEDULE,
            "budget": 100,
            "blocks_per_snr": args.blocks,
            "batch": args.batch,
            "snrs_db": args.esn0,
            "seed": args.seed,
            "strength": strength.__dict__,
            "registration": (
                "oracle inverse affine and brightness"
                if args.registration == "oracle"
                else "receiver grid estimate from first 5-iteration BP hard image"
            ),
            "side_channel_cost": 0,
            "side_available_losslessly": True,
            "production_decoder_modified": False,
        },
        "configs": configs,
        "points": {},
    }

    for snr_index, snr in enumerate(args.esn0):
        point_seed = args.seed + snr_index * 10000
        names = [config["name"] for config in configs]
        if pure_decoder is not None:
            names.append("bp100")
        accum = {
            name: {"crc_valid": [], "hard_mse_255": [], "bit_errors": []}
            for name in names
        }
        registration_diag = {
            "oracle_mse_01": [], "estimated_mse_01": [],
            "raw_side_mse_01": [], "abs_tx_error": [],
            "abs_ty_error": [], "abs_angle_error_deg": [],
            "abs_brightness_error": [],
        }
        for round_index in range(args.blocks // args.batch):
            indices, payload, truth_images, channel_llr = make_channel_batch(
                round_index,
                args.batch,
                point_seed,
                snr,
                images,
                bit_bank,
                crc_encoder,
                ldpc,
                mapper,
                demapper,
                awgn,
            )
            side, oracle_aligned, true_params = make_views(
                truth_images,
                strength,
                point_seed + 500000 + round_index * args.batch,
            )
            aligned = oracle_aligned
            for config_index, config in enumerate(configs):
                scheduler, path = geom_scheduler(SCHEDULE, config["endpoint"])
                decoder.altproj = True
                decoder.ep_mode = False
                decoder.altproj_delta = float(config["delta"])
                decoder.altproj_rho = float(config["rho"])
                decoder.altproj_warm_start = True
                decoder.altproj_early_stop = False
                decoder.sigma_scheduler = scheduler
                proxy.configure(config, path, aligned)
                logits = decoder(channel_llr)
                if proxy.call_index != SOURCE_CALLS:
                    raise RuntimeError(
                        f"source calls {proxy.call_index}, expected {SOURCE_CALLS}"
                    )
                values = metrics_from_logits(
                    logits, payload.numpy(), truth_images, crc_decoder
                )
                for key in accum[config["name"]]:
                    accum[config["name"]][key].extend(values[key])
                if config_index == 0 and args.registration == "estimated":
                    first = np.asarray(decoder.last_u_hat_hist[0].numpy())[
                        :, :K_PAYLOAD
                    ]
                    first_bits = first.reshape(-1, PIXELS, BPP) > 0.0
                    first_image = np.sum(
                        first_bits * BIT_WEIGHTS[None, None, :], axis=2
                    ).reshape(-1, IMG_H, IMG_W) / 255.0
                    aligned, estimates = estimate_alignment_grid(
                        side, first_image, strength
                    )
                    registration_diag["oracle_mse_01"].extend(
                        np.mean((oracle_aligned - truth_images) ** 2,
                                axis=(1, 2)).tolist()
                    )
                    registration_diag["estimated_mse_01"].extend(
                        np.mean((aligned - truth_images) ** 2,
                                axis=(1, 2)).tolist()
                    )
                    registration_diag["raw_side_mse_01"].extend(
                        np.mean((side - truth_images) ** 2,
                                axis=(1, 2)).tolist()
                    )
                    for truth, estimate in zip(true_params, estimates):
                        registration_diag["abs_tx_error"].append(
                            abs(truth["tx"] - estimate["tx"])
                        )
                        registration_diag["abs_ty_error"].append(
                            abs(truth["ty"] - estimate["ty"])
                        )
                        registration_diag["abs_angle_error_deg"].append(
                            abs(truth["angle_deg"] - estimate["angle_deg"])
                        )
                        registration_diag["abs_brightness_error"].append(
                            abs(truth["brightness"] - estimate["brightness"])
                        )
            if pure_decoder is not None:
                bp_logits = pure_decoder(channel_llr)
                values = metrics_from_logits(
                    bp_logits, payload.numpy(), truth_images, crc_decoder
                )
                for key in accum["bp100"]:
                    accum["bp100"][key].extend(values[key])
            print(
                f"{args.mode} Es/N0={snr:+.2f} round "
                f"{round_index + 1}/{args.blocks // args.batch}",
                flush=True,
            )

        summaries = {
            name: summarize_metrics(values, point_seed + i * 101)
            for i, (name, values) in enumerate(accum.items())
        }
        reference = summaries[configs[0]["name"]]
        for config in configs[1:]:
            summaries[config["name"]]["paired_vs_no_side"] = paired(
                summaries[config["name"]], reference
            )
        result["points"][f"{snr:.3f}"] = {
            "esn0_db": float(snr),
            "systems": summaries,
        }
        if args.registration == "estimated":
            result["points"][f"{snr:.3f}"]["registration_diagnostics"] = {
                key: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p90": float(np.quantile(values, 0.9)),
                }
                for key, values in registration_diag.items()
            }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("screen", "confirm", "waterfall", "control"),
        default="screen"
    )
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--esn0", type=float, nargs="+", default=[-2.85])
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--strength", default="strong")
    parser.add_argument("--registration", choices=("oracle", "estimated"),
                        default="oracle")
    parser.add_argument(
        "--rho", type=float, default=None,
        help="optional common rho override for canonical-condition checks",
    )
    parser.add_argument("--skip-bp", action="store_true")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default="results/multiview_altproj_screen32.json")
    args = parser.parse_args()
    if args.blocks % args.batch:
        raise ValueError("blocks must be divisible by batch")
    if args.mode == "screen":
        configs = screen_configs()
    elif args.mode == "confirm":
        configs = confirm_configs()
    elif args.mode == "waterfall":
        configs = waterfall_configs()
    else:
        configs = sigma_control_configs()
    if args.rho is not None:
        for config in configs:
            config["rho"] = float(args.rho)
    result = run(args, configs)
    for key, point in result["points"].items():
        print("\nSNR", key)
        ranked = sorted(
            point["systems"].items(), key=lambda item: (item[1]["failures"], item[1]["mean_mse_01"])
        )
        for name, row in ranked[:8]:
            print(name, row["failures"], f"mse={row['mean_mse_01']:.6g}")


if __name__ == "__main__":
    main()
