#!/usr/bin/env python3
"""Gate a receiver-only correlated-view prior before decoder integration.

The script never enters the LDPC loop. It creates a target Fashion-MNIST view
X and a transformed side view Y, oracle-aligns Y with the known affine and
brightness parameters, and compares pixel estimators after the existing
sigma_post=3 bit readout.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import affine_transform
from torchvision.datasets import FashionMNIST

from source_prior import SourcePriorDenoiser


H = W = 28
BPP = 8
PIXELS = H * W
PAYLOAD_BITS = PIXELS * BPP
SIGMAS = (0.2, 0.1, 0.05, 0.02, 0.005)
CONVEX_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
SIMPLEX_WEIGHTS = tuple(
    (wd / 4.0, ws / 4.0, (4 - wd - ws) / 4.0)
    for wd in range(5)
    for ws in range(5 - wd)
)


@dataclass(frozen=True)
class Strength:
    name: str
    max_shift: float
    max_angle: float
    brightness_low: float
    brightness_high: float


# The first three are the requested starting points. The last two are retained
# as automatic non-triviality fallbacks if oracle-aligned side accuracy is >90%.
STRENGTHS = (
    Strength("strong", 1.0, 3.0, 0.95, 1.05),
    Strength("medium", 2.0, 7.0, 0.90, 1.10),
    Strength("weak", 4.0, 15.0, 0.80, 1.20),
    Strength("weaker_fallback", 6.0, 25.0, 0.65, 1.35),
    Strength("weakest_fallback", 8.0, 35.0, 0.50, 1.50),
)


def bits_from_images(images_u8: np.ndarray) -> np.ndarray:
    return np.unpackbits(images_u8.reshape(-1, PIXELS, 1), axis=2)


def affine_pair(
    image: np.ndarray,
    angle_deg: float,
    tx: float,
    ty: float,
    brightness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed Y and Y oracle-aligned back to X coordinates."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    # Coordinates are (row=y, col=x). q = R p + offset maps X points to Y.
    rotation_yx = np.array([[c, s], [-s, c]], dtype=np.float64)
    center = np.array([(H - 1) / 2.0, (W - 1) / 2.0], dtype=np.float64)
    translation_yx = np.array([ty, tx], dtype=np.float64)
    forward_offset = center + translation_yx - rotation_yx @ center

    # Render Y: output q samples input p = R^-1(q-offset).
    inverse_rotation = rotation_yx.T
    y = affine_transform(
        image,
        inverse_rotation,
        offset=-inverse_rotation @ forward_offset,
        output_shape=(H, W),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    y = np.clip(brightness * y, 0.0, 1.0)

    # Oracle registration: output p samples Y at q = R p + offset.
    aligned = affine_transform(
        y,
        rotation_yx,
        offset=forward_offset,
        output_shape=(H, W),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    aligned = np.clip(aligned / brightness, 0.0, 1.0)
    return y.astype(np.float32), aligned.astype(np.float32)


def make_views(
    images: np.ndarray, strength: Strength, seed: int
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    side = np.empty_like(images, dtype=np.float32)
    aligned = np.empty_like(images, dtype=np.float32)
    params: list[dict] = []
    for index, image in enumerate(images):
        tx = float(rng.uniform(-strength.max_shift, strength.max_shift))
        ty = float(rng.uniform(-strength.max_shift, strength.max_shift))
        angle = float(rng.uniform(-strength.max_angle, strength.max_angle))
        brightness = float(
            rng.uniform(strength.brightness_low, strength.brightness_high)
        )
        side[index], aligned[index] = affine_pair(
            image, angle, tx, ty, brightness
        )
        params.append(
            {
                "tx": tx,
                "ty": ty,
                "angle_deg": angle,
                "brightness": brightness,
            }
        )
    return side, aligned, params


def counter() -> dict:
    return {"correct": np.zeros(BPP, dtype=np.int64), "total": 0}


def update_counter(target: dict, hard: torch.Tensor, truth: torch.Tensor) -> None:
    equal = hard.reshape(-1, PIXELS, BPP) == truth.reshape(-1, PIXELS, BPP)
    target["correct"] += equal.sum(dim=(0, 1)).cpu().numpy().astype(np.int64)
    target["total"] += int(equal.shape[0] * equal.shape[1])


def summarize(target: dict) -> dict:
    plane = target["correct"].astype(np.float64) / target["total"]
    return {
        "overall": float(np.mean(plane)),
        "bit_plane_msb_to_lsb": plane.tolist(),
        "central_b2_to_b5_mean": float(np.mean(plane[2:6])),
        "bits_compared": int(target["total"] * BPP),
    }


def image_hard_bits(prior: SourcePriorDenoiser, image: torch.Tensor) -> torch.Tensor:
    logits = prior.soft_field_to_posterior_logits(image)
    return logits > 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--checkpoint",
        default="/home/LJH/onlyextrinsic_ada_sigma/checkpoints/denoiser.pt",
    )
    parser.add_argument("--output", default="results/multiview_gate_256.json")
    args = parser.parse_args()
    if args.blocks % args.batch:
        raise ValueError("blocks must be divisible by batch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FashionMNIST(root="/tmp/fmnist", train=False, download=False)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=args.blocks, replace=False)
    images_u8 = dataset.data.numpy()[indices].astype(np.uint8)
    images = images_u8.astype(np.float32) / 255.0
    truth_bits_np = bits_from_images(images_u8).astype(bool)

    prior = SourcePriorDenoiser(sigma_post=3.0).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    prior.load_state_dict(state)
    prior.eval()

    view_bank: dict[str, dict] = {}
    for strength_index, strength in enumerate(STRENGTHS):
        side, aligned, params = make_views(
            images, strength, args.seed + 1000 * (strength_index + 1)
        )
        view_bank[strength.name] = {
            "spec": strength.__dict__,
            "side": side,
            "aligned": aligned,
            "params": params,
        }

    result = {
        "kind": "multiview_oracle_registration_gate",
        "configuration": {
            "branch": "codex/multiview",
            "dataset": "FashionMNIST test",
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "indices": indices.tolist(),
            "payload_bits": PAYLOAD_BITS,
            "sigma_actual": list(SIGMAS),
            "sigma_post": 3.0,
            "fixed_sigma_reference": 0.3,
            "registration": "oracle inverse affine and brightness",
            "side_channel_cost": 0,
            "side_assumption": "Y is already available losslessly at receiver",
            "convex_side_weights": list(CONVEX_WEIGHTS),
            "three_way_weights_order": "(denoiser, side, cavity)",
            "three_way_weights": [list(value) for value in SIMPLEX_WEIGHTS],
        },
        "strengths": {},
    }

    for strength_index, strength in enumerate(STRENGTHS):
        bank = view_bank[strength.name]
        rows = []
        for sigma_index, sigma in enumerate(SIGMAS):
            estimator_counts = {
                "cavity": counter(),
                "denoiser_matched": counter(),
                "denoiser_fixed_0.3": counter(),
                "side_oracle": counter(),
            }
            convex_counts = {str(weight): counter() for weight in CONVEX_WEIGHTS}
            simplex_counts = {
                f"{wd},{ws},{wc}": counter()
                for wd, ws, wc in SIMPLEX_WEIGHTS
            }
            for batch_index, start in enumerate(range(0, args.blocks, args.batch)):
                stop = start + args.batch
                truth_image = torch.from_numpy(images[start:stop, None]).to(device)
                truth_bits = torch.from_numpy(truth_bits_np[start:stop]).to(device)
                side_image = torch.from_numpy(bank["aligned"][start:stop, None]).to(device)
                noise_gen = torch.Generator(device=device)
                noise_gen.manual_seed(
                    args.seed + sigma_index * 10000 + batch_index * 101
                )
                cavity = torch.clamp(
                    truth_image
                    + sigma
                    * torch.randn(
                        truth_image.shape,
                        generator=noise_gen,
                        device=device,
                        dtype=truth_image.dtype,
                    ),
                    0.0,
                    1.0,
                )
                sigma_vector = torch.full(
                    (len(truth_image),), sigma, device=device
                )
                fixed_vector = torch.full(
                    (len(truth_image),), 0.3, device=device
                )
                with torch.no_grad():
                    den_matched = prior.net(cavity, sigma_vector).clamp(0.0, 1.0)
                    den_fixed = prior.net(cavity, fixed_vector).clamp(0.0, 1.0)
                    for name, estimate in (
                        ("cavity", cavity),
                        ("denoiser_matched", den_matched),
                        ("denoiser_fixed_0.3", den_fixed),
                        ("side_oracle", side_image),
                    ):
                        update_counter(
                            estimator_counts[name],
                            image_hard_bits(prior, estimate),
                            truth_bits,
                        )
                    for weight in CONVEX_WEIGHTS:
                        estimate = (1.0 - weight) * den_matched + weight * side_image
                        update_counter(
                            convex_counts[str(weight)],
                            image_hard_bits(prior, estimate),
                            truth_bits,
                        )
                    for wd, ws, wc in SIMPLEX_WEIGHTS:
                        estimate = wd * den_matched + ws * side_image + wc * cavity
                        key = f"{wd},{ws},{wc}"
                        update_counter(
                            simplex_counts[key],
                            image_hard_bits(prior, estimate),
                            truth_bits,
                        )
            estimators = {
                name: summarize(values) for name, values in estimator_counts.items()
            }
            convex = {
                key: summarize(values) for key, values in convex_counts.items()
            }
            simplex = {
                key: summarize(values) for key, values in simplex_counts.items()
            }
            # Gate on genuine fusion, not the pure endpoints contained in the
            # diagnostic grids. Both denoiser and side view must have weight.
            fusion_convex_keys = [key for key in convex if 0.0 < float(key) < 1.0]
            fusion_simplex_keys = [
                key
                for key in simplex
                if float(key.split(",")[0]) > 0.0
                and float(key.split(",")[1]) > 0.0
            ]
            best_convex_key = max(
                fusion_convex_keys, key=lambda key: convex[key]["overall"]
            )
            best_simplex_key = max(
                fusion_simplex_keys, key=lambda key: simplex[key]["overall"]
            )
            candidates = {
                "convex": (best_convex_key, convex[best_convex_key]),
                "three_way": (best_simplex_key, simplex[best_simplex_key]),
            }
            best_kind, (best_key, best_metrics) = max(
                candidates.items(), key=lambda item: item[1][1]["overall"]
            )
            baseline = estimators["denoiser_matched"]
            rows.append(
                {
                    "sigma_actual": sigma,
                    "estimators": estimators,
                    "best_convex": {
                        "side_weight": float(best_convex_key),
                        **convex[best_convex_key],
                    },
                    "best_three_way": {
                        "weights_denoiser_side_cavity": [
                            float(value) for value in best_simplex_key.split(",")
                        ],
                        **simplex[best_simplex_key],
                    },
                    "best_combination": {
                        "kind": best_kind,
                        "key": best_key,
                        **best_metrics,
                        "overall_gain_vs_denoiser": (
                            best_metrics["overall"] - baseline["overall"]
                        ),
                        "central_gain_vs_denoiser": (
                            best_metrics["central_b2_to_b5_mean"]
                            - baseline["central_b2_to_b5_mean"]
                        ),
                    },
                }
            )
            print(
                f"gate {strength.name} sigma={sigma:g} "
                f"den={baseline['overall']:.4f} side={estimators['side_oracle']['overall']:.4f} "
                f"best={best_metrics['overall']:.4f}",
                flush=True,
            )

        side_accuracy = rows[0]["estimators"]["side_oracle"]["overall"]
        strong_rows = [
            row
            for row in rows
            if row["best_combination"]["overall"] >= 0.85
            and row["best_combination"]["central_gain_vs_denoiser"] >= 0.15
        ]
        max_combined = max(row["best_combination"]["overall"] for row in rows)
        if side_accuracy >= 0.90:
            verdict = "reject_trivial_side_over_90pct"
        elif len(strong_rows) >= 2:
            verdict = "strong_signal_proceed_full"
        elif max_combined >= 0.75:
            verdict = "intermediate_signal_proceed_reduced"
        else:
            verdict = "weak_signal_stop"
        result["strengths"][strength.name] = {
            "spec": bank["spec"],
            "transform_parameter_summary": {
                key: {
                    "min": float(min(item[key] for item in bank["params"])),
                    "mean": float(np.mean([item[key] for item in bank["params"]])),
                    "max": float(max(item[key] for item in bank["params"])),
                }
                for key in ("tx", "ty", "angle_deg", "brightness")
            },
            "rows": rows,
            "gate": {
                "side_accuracy": side_accuracy,
                "strong_noise_levels": [row["sigma_actual"] for row in strong_rows],
                "max_combined_accuracy": max_combined,
                "verdict": verdict,
            },
        }

    nontrivial = [
        (name, entry)
        for name, entry in result["strengths"].items()
        if entry["gate"]["side_accuracy"] < 0.90
        and entry["gate"]["verdict"] != "weak_signal_stop"
    ]
    if nontrivial:
        selected_name, selected = max(
            nontrivial,
            key=lambda item: (
                len(item[1]["gate"]["strong_noise_levels"]),
                item[1]["gate"]["max_combined_accuracy"],
            ),
        )
        proceed = selected["gate"]["verdict"]
    else:
        selected_name, proceed = None, "stop"
    result["global_gate"] = {
        "selected_strength": selected_name,
        "decision": proceed,
        "rule": (
            "side<90%; strong if combined>=85% and central b2..b5 gain>=15pp "
            "at >=2 sigma levels; otherwise reduced if max combined>=75%"
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["global_gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
