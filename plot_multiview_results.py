#!/usr/bin/env python3
"""Create the final multiview gate/waterfall figures and derived summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load(name):
    return json.loads((RESULTS / name).read_text())


def interp_knee(points, target):
    ordered = sorted(points)
    for (x0, y0), (x1, y1) in zip(ordered[:-1], ordered[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return None


def main():
    gate = load("multiview_gate_256.json")
    wf = load("multiview_altproj_canonical_waterfall512.json")
    control = load("multiview_no_side_sigma_control512.json")
    reg_est = load("multiview_altproj_estimated_registration64.json")
    reg_oracle = load("multiview_altproj_oracle_registration64.json")

    strong = gate["strengths"]["strong"]
    rows = strong["rows"]
    sigmas = np.asarray([row["sigma_actual"] for row in rows])
    cavity = np.asarray([row["estimators"]["cavity"]["overall"] for row in rows])
    den = np.asarray([row["estimators"]["denoiser_matched"]["overall"] for row in rows])
    side = np.asarray([row["estimators"]["side_oracle"]["overall"] for row in rows])
    combo = np.asarray([row["best_combination"]["overall"] for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
    ax = axes[0]
    for values, marker, label in (
        (cavity, "o", "cavity"),
        (den, "s", "denoiser"),
        (side, "^", "oracle-aligned side"),
        (combo, "D", "best genuine fusion"),
    ):
        ax.plot(sigmas, 100 * values, marker=marker, lw=1.8, label=label)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"artificial cavity noise $\sigma_{actual}$")
    ax.set_ylabel("bit agreement (%)")
    ax.set_title("Gate: help only while cavity is noisy")
    ax.legend(fontsize=8)

    ax = axes[1]
    row = rows[0]
    labels = ["MSB", "b6", "b5", "b4", "b3", "b2", "b1", "LSB"]
    x = np.arange(8)
    for values, marker, label in (
        (row["estimators"]["denoiser_matched"]["bit_plane_msb_to_lsb"], "s", "denoiser"),
        (row["estimators"]["side_oracle"]["bit_plane_msb_to_lsb"], "^", "side"),
        (row["best_combination"]["bit_plane_msb_to_lsb"], "D", "fusion"),
    ):
        ax.plot(x, 100 * np.asarray(values), marker=marker, lw=1.8, label=label)
    ax.set_xticks(x, labels)
    ax.set_ylim(55, 100)
    ax.set_ylabel("bit agreement (%)")
    ax.set_title(r"Bit planes at $\sigma_{actual}=0.2$")
    ax.legend(fontsize=8)

    ax = axes[2]
    strength_names = list(gate["strengths"])
    side_acc = [gate["strengths"][name]["gate"]["side_accuracy"] for name in strength_names]
    max_combo = [gate["strengths"][name]["gate"]["max_combined_accuracy"] for name in strength_names]
    labels_short = ["strong", "medium", "weak", "weaker", "weakest"]
    xx = np.arange(len(strength_names))
    ax.plot(xx, 100 * np.asarray(side_acc), "o-", label="side alone")
    ax.plot(xx, 100 * np.asarray(max_combo), "D-", label="max fusion")
    ax.axhline(90, color="black", ls=":", lw=1, label="triviality guard (90%)")
    ax.set_xticks(xx, labels_short, rotation=18)
    ax.set_ylabel("bit agreement (%)")
    ax.set_title("Correlation-strength gate")
    ax.legend(fontsize=8)
    fig.suptitle("Correlated-view gating (256 Fashion-MNIST blocks)", fontsize=14)
    fig.savefig(RESULTS / "multiview_gate.png", dpi=190)
    plt.close(fig)

    names = ["none_d0.02_e0.05", "pull_d0.02_ds0.01_sigma_ge_0.1_e0.02"]
    labels = ["ours, no side (end=.05)", "ours + side (end=.02, early gate)"]
    colors = ["#4C78A8", "#E45756"]
    series = {name: [] for name in names}
    for key, point in wf["points"].items():
        x = float(key)
        for name in names:
            row = point["systems"][name]
            series[name].append((x, row["crc_bler"], row["wilson_95"], row["mean_mse_01"]))

    no_points = [(x, y) for x, y, _, _ in series[names[0]]]
    side_points = [(x, y) for x, y, _, _ in series[names[1]]]
    same_endpoint_points = []
    for key, point in control["points"].items():
        row = point["systems"]["none_d0.02_e0.02"]
        same_endpoint_points.append((float(key), row["crc_bler"]))
    knees = {}
    for target in (0.1, 0.01):
        k0 = interp_knee(no_points, target)
        ksame = interp_knee(same_endpoint_points, target)
        k1 = interp_knee(side_points, target)
        knees[str(target)] = {
            "no_side_db": k0,
            "no_side_same_endpoint_db": ksame,
            "with_side_db": k1,
            "side_gain_db": None if k0 is None or k1 is None else k0 - k1,
            "side_gain_same_endpoint_db": (
                None if ksame is None or k1 is None else ksame - k1
            ),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)
    ax = axes[0]
    for name, label, color in zip(names, labels, colors):
        values = sorted(series[name])
        x = np.asarray([v[0] for v in values])
        y = np.asarray([v[1] for v in values])
        lo = np.asarray([v[2][0] for v in values])
        hi = np.asarray([v[2][1] for v in values])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=3,
                    lw=2, color=color, label=label)
    same_x = np.asarray([p[0] for p in sorted(same_endpoint_points)])
    same_y = np.asarray([p[1] for p in sorted(same_endpoint_points)])
    ax.plot(same_x, same_y, "o--", color="#888888", lw=1.5,
            label="no side, matched end=.02 control")
    ax.axhline(0.1, color="black", ls=":", lw=1)
    ax.axhline(0.01, color="black", ls=":", lw=1)
    baseline_knees = {
        "PixelCNN-MAX": -3.826,
        "WebP-MAX": -2.801,
        "raw + BP-100": -2.331,
    }
    for label, knee in baseline_knees.items():
        ax.axvline(knee, ls="--", lw=1.2, alpha=0.72, label=f"{label} knee@0.1")
    ax.set_yscale("log")
    ax.set_xlim(-4.0, -2.2)
    ax.set_ylim(0.002, 1.0)
    ax.set_xlabel("Es/N0 (dB)")
    ax.set_ylabel("CRC BLER (hard CRC)")
    ax.set_title("Budget-100 waterfall, 512 blocks/point")
    ax.legend(fontsize=7.5, ncol=2)

    ax = axes[1]
    for name, label, color in zip(names, labels, colors):
        values = sorted(series[name])
        x = np.asarray([v[0] for v in values])
        mse = np.asarray([v[3] for v in values])
        ax.plot(x, mse, "o-", lw=2, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Es/N0 (dB)")
    ax.set_ylabel("mean hard-image MSE / 255²")
    ax.set_title("Best-effort distortion")
    ax.legend(fontsize=8)
    fig.suptitle("Canonical multiview altproj: rho=.95, delta=.02, delta_side=.01", fontsize=14)
    fig.savefig(RESULTS / "multiview_waterfall.png", dpi=190)
    plt.close(fig)

    shift = knees["0.1"]["side_gain_same_endpoint_db"]
    established = {
        "raw_bp100_knee_0p1_db": -2.331,
        "ours_no_side_knee_0p1_db": -2.940,
        "webp_max_knee_0p1_db": -2.801,
        "pixelcnn_max_knee_0p1_db": -3.826,
    }
    anchored_side = established["ours_no_side_knee_0p1_db"] - shift
    p0 = wf["points"]["-3.050"]["systems"][names[0]]["crc_bler"]
    ps = wf["points"]["-3.050"]["systems"][names[1]]["crc_bler"]
    summary = {
        "canonical_512_knees_linear_interpolation": knees,
        "established_hard_crc_knees": established,
        "anchored_multiview_knee_0p1_db": anchored_side,
        "remaining_gap_to_pixelcnn_db": anchored_side - established["pixelcnn_max_knee_0p1_db"],
        "advantage_over_webp_db": established["webp_max_knee_0p1_db"] - anchored_side,
        "same_endpoint_paired_side_accounting": {},
        "estimated_registration_same_seed_minus3p05": {
            "no_side_failures": reg_est["points"]["-3.050"]["systems"][names[0]]["failures"],
            "oracle_side_failures": reg_oracle["points"]["-3.050"]["systems"][names[1]]["failures"],
            "estimated_side_failures": reg_est["points"]["-3.050"]["systems"][names[1]]["failures"],
            "diagnostics": reg_est["points"]["-3.050"]["registration_diagnostics"],
        },
        "two_stage_accounting_not_direct_measurement": {
            "snr_db": -3.05,
            "assumption": "first-view no-side BLER equals target no-side BLER and view failures are independent",
            "pair_bler_no_side": 1.0 - (1.0 - p0) ** 2,
            "pair_bler_sequential_side": 1.0 - (1.0 - p0) * (1.0 - ps),
            "per_view_bler_with_no_side_fallback": 0.5 * (p0 + (1.0 - p0) * ps + p0 * p0),
        },
    }
    for key, point in wf["points"].items():
        side_mask = np.asarray(point["systems"][names[1]]["success_mask"], dtype=bool)
        ref_mask = np.asarray(
            control["points"][key]["systems"]["none_d0.02_e0.02"]["success_mask"],
            dtype=bool,
        )
        summary["same_endpoint_paired_side_accounting"][key] = {
            "broken": int(np.sum(ref_mask & ~side_mask)),
            "rescued": int(np.sum(~ref_mask & side_mask)),
        }
    (RESULTS / "multiview_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
