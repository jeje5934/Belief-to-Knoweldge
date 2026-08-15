#!/usr/bin/env python3
"""Merge paired raw/codec multiview measurements and make paper-facing plots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RAW_NAME = {
    "A_raw_bp100": "bp100",
    "B_altproj_no_side": "none_d0.02_e0.05",
    "C_altproj_side_in_loop": "pull_d0.02_ds0.01_sigma_ge_0.1_e0.02",
}


def load_point(path):
    payload = json.loads(Path(path).read_text())
    return next(iter(payload["points"].values()))


def log_knee(points, target):
    points = sorted((float(x), float(y)) for x, y in points if y > 0.0)
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if (y1 - target) * (y2 - target) <= 0.0 and y1 != y2:
            fraction = ((math.log10(target) - math.log10(y1)) /
                        (math.log10(y2) - math.log10(y1)))
            return x1 + fraction * (x2 - x1)
    return None


def paired_account(reference, candidate):
    ref = np.asarray(reference, dtype=bool)
    new = np.asarray(candidate, dtype=bool)
    return {
        "rescued": int(np.sum(~ref & new)),
        "broken": int(np.sum(ref & ~new)),
        "net_failure_reduction": int(np.sum(~ref) - np.sum(~new)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", nargs="+", required=True)
    parser.add_argument("--codec", nargs="+", required=True)
    parser.add_argument("--entropy", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-plot", required=True)
    args = parser.parse_args()

    raw = {float(p["esn0_db"]): p for p in map(load_point, args.raw)}
    codec = {float(p["esn0_db"]): p for p in map(load_point, args.codec)}
    snrs = sorted(set(raw) & set(codec))
    if len(snrs) != len(raw) or len(snrs) != len(codec):
        raise RuntimeError("raw and codec SNR grids differ")

    rows = []
    accounting = {}
    series = {name: {"bler": [], "mse": [], "ci": []} for name in (
        *RAW_NAME, "D_webp_independent", "E_webp_post_side",
        "D_pixelcnn_independent", "E_pixelcnn_post_side",
    )}
    for snr in snrs:
        rp, cp = raw[snr], codec[snr]
        raw_rows = {label: rp["systems"][key] for label, key in RAW_NAME.items()}
        for label, row in raw_rows.items():
            series[label]["bler"].append(row["crc_bler"])
            series[label]["mse"].append(row["mean_mse_01"])
            series[label]["ci"].append(row["wilson_95"])
            rows.append({
                "esn0_db": snr, "system": label,
                "failures": row["failures"], "blocks": row["blocks"],
                "bler": row["crc_bler"], "wilson_95": row["wilson_95"],
                "mean_mse_01": row["mean_mse_01"],
                "psnr_db": row["aggregate_psnr_db"],
                **row["mse_quantiles"],
            })

        for codec_name in ("webp", "pixelcnn"):
            row = cp["codecs"][codec_name]
            labels = (
                (f"D_{codec_name}_independent", row["D_lower_envelope"]),
                (f"E_{codec_name}_post_side", row["E_primary"]),
            )
            for label, mse_row in labels:
                series[label]["bler"].append(row["crc_bler"])
                series[label]["mse"].append(mse_row["mean_mse_01"])
                series[label]["ci"].append(row["wilson_95"])
                rows.append({
                    "esn0_db": snr, "system": label,
                    "failures": row["crc_failures"], "blocks": row["blocks"],
                    "bler": row["crc_bler"], "wilson_95": row["wilson_95"],
                    "mean_mse_01": mse_row["mean_mse_01"],
                    "psnr_db": mse_row["aggregate_psnr_db"],
                    "p50": mse_row["p50"], "p90": mse_row["p90"],
                    "p99": mse_row["p99"],
                })

        a = raw_rows["A_raw_bp100"]["success_mask"]
        b = raw_rows["B_altproj_no_side"]["success_mask"]
        c = raw_rows["C_altproj_side_in_loop"]["success_mask"]
        webp = cp["codecs"]["webp"]["crc_success_mask"]
        accounting[f"{snr:.2f}"] = {
            "A_to_C": paired_account(a, c),
            "B_to_C": paired_account(b, c),
            "WebP_D_to_E_bit_exact": {"rescued": 0, "broken": 0,
                "note": "postprocessing cannot repair the compressed bitstream CRC"},
            "WebP_vs_C": {
                "C_only_success": int(np.sum(np.asarray(c) & ~np.asarray(webp))),
                "WebP_only_success": int(np.sum(~np.asarray(c) & np.asarray(webp))),
            },
        }

    knees = {}
    for label, values in series.items():
        knees[label] = {
            "bler_0.1_db": log_knee(zip(snrs, values["bler"]), 0.1),
            "bler_0.01_db": log_knee(zip(snrs, values["bler"]), 0.01),
        }

    entropy = json.loads(Path(args.entropy).read_text())
    strong = entropy["strengths"]["strong"]
    db_per_bpp = 1.494 / (8.0 - entropy["pixelcnn_operational_hx_proxy_bpp"])
    available_db_proxy = strong["i_x_y_pixelwise_bpp"] * db_per_bpp
    b01 = knees["B_altproj_no_side"]["bler_0.1_db"]
    c01 = knees["C_altproj_side_in_loop"]["bler_0.1_db"]
    b001 = knees["B_altproj_no_side"]["bler_0.01_db"]
    c001 = knees["C_altproj_side_in_loop"]["bler_0.01_db"]
    realization = {
        "calibration_db_per_bpp": db_per_bpp,
        "calibration": "1.494 dB / (8 - PixelCNN mean 3.036 bpp)",
        "strong_i_x_y_pixelwise_bpp": strong["i_x_y_pixelwise_bpp"],
        "available_gain_proxy_db": available_db_proxy,
        "measured_side_gain_at_bler_0.1_db": b01 - c01,
        "realization_fraction_at_bler_0.1": (b01 - c01) / available_db_proxy,
        "measured_side_gain_at_bler_0.01_db": b001 - c001,
        "realization_fraction_at_bler_0.01": (b001 - c001) / available_db_proxy,
        "warning": "heuristic cross-calibration, not an information-theoretic efficiency",
    }

    output = {
        "kind": "multiview_loop_vs_post_summary",
        "snrs_db": snrs,
        "rows": rows,
        "paired_accounting": accounting,
        "knees": knees,
        "correlation_realization": realization,
        "primary_post_policy": "side_direct (selected on independent 64-block screen)",
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2) + "\n")

    colors = {
        "A_raw_bp100": "#8c8c8c", "B_altproj_no_side": "#1f77b4",
        "C_altproj_side_in_loop": "#d62728", "D_webp_independent": "#2ca02c",
        "E_webp_post_side": "#9467bd", "D_pixelcnn_independent": "#ff7f0e",
        "E_pixelcnn_post_side": "#8c564b",
    }
    labels = {
        "A_raw_bp100": "A raw + BP-100", "B_altproj_no_side": "B raw + altproj",
        "C_altproj_side_in_loop": "C side inside loop", "D_webp_independent": "D WebP + BP",
        "E_webp_post_side": "E WebP + post side", "D_pixelcnn_independent": "D PixelCNN + BP",
        "E_pixelcnn_post_side": "E PixelCNN + post side",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1), constrained_layout=True)
    bler_order = ["A_raw_bp100", "B_altproj_no_side", "C_altproj_side_in_loop",
                  "D_webp_independent", "D_pixelcnn_independent"]
    for name in bler_order:
        y = np.asarray(series[name]["bler"], dtype=float)
        ci = np.asarray(series[name]["ci"], dtype=float)
        plot_y = np.maximum(y, 4e-4)
        axes[0].plot(snrs, plot_y, marker="o", linewidth=2, color=colors[name], label=labels[name])
        axes[0].fill_between(snrs, np.maximum(ci[:, 0], 3e-4), np.maximum(ci[:, 1], 3e-4),
                             color=colors[name], alpha=0.12)
    axes[0].set_yscale("log")
    axes[0].set_ylim(3e-4, 1.15)
    axes[0].set_xlabel("Es/N0 (dB)")
    axes[0].set_ylabel("CRC BLER")
    axes[0].set_title("Bit-exact reliability\n(D and E overlap in BLER)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].text(0.02, 0.02, "0-failure points shown at 4e-4; Wilson bands retained",
                 transform=axes[0].transAxes, fontsize=7, color="#555555")

    mse_order = ["A_raw_bp100", "B_altproj_no_side", "C_altproj_side_in_loop",
                 "D_webp_independent", "E_webp_post_side",
                 "D_pixelcnn_independent", "E_pixelcnn_post_side"]
    for name in mse_order:
        y = np.maximum(np.asarray(series[name]["mse"], dtype=float), 1e-7)
        style = "--" if name.startswith("E_") else "-"
        axes[1].plot(snrs, y, marker="o", linewidth=2, linestyle=style,
                     color=colors[name], label=labels[name])
    axes[1].set_yscale("log")
    axes[1].set_ylim(5e-8, 0.12)
    axes[1].set_xlabel("Es/N0 (dB)")
    axes[1].set_ylabel("Mean MSE (0–1)")
    axes[1].set_title("Distortion including CRC failures\n(E uses baseline-favouring concealment)")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].text(0.02, 0.02, "Exact-zero MSE shown at 1e-7 for log-scale visibility",
                 transform=axes[1].transAxes, fontsize=7, color="#555555")
    fig.suptitle("Correlation inside the LDPC loop vs post-decode fusion (estimated registration)")
    Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=180)
    plt.close(fig)
    print(json.dumps({"knees": knees, "realization": realization}, indent=2))


if __name__ == "__main__":
    main()
