#!/usr/bin/env python3
"""Aggregate and plot the single-terminal site-variance EP study."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_FILES = (
    "site_variance_both_waterfall_final.json",
    "site_variance_both_m255_a512.json",
    "site_variance_both_m250_256.json",
    "site_variance_both_m245_256.json",
)
SERIES = {
    "const": "constant moments (baseline)",
    "both_c1_cp1": "honest full variance (c=1, c'=1)",
    "both_c1_cp0.5": "scaled source variance (c=1, c'=0.5)",
}


def load_points(result_dir: Path):
    points = {}
    for filename in RESULT_FILES:
        payload = json.loads((result_dir / filename).read_text())
        for key, point in payload["points"].items():
            points[float(key)] = point
    return points


def interpolate_knee(points, name, target):
    observations = sorted(
        (snr, point["rows"][name]["crc_bler"])
        for snr, point in points.items()
        if name in point["rows"] and point["rows"][name]["crc_bler"] > 0.0
    )
    for (x0, y0), (x1, y1) in zip(observations, observations[1:]):
        if (y0 - target) * (y1 - target) <= 0.0:
            log0, log1, logt = map(math.log10, (y0, y1, target))
            knee = x0 + (logt - log0) * (x1 - x0) / (log1 - log0)
            return {
                "estimate_db": knee,
                "bracket": [[x0, y0], [x1, y1]],
                "method": "linear interpolation in log10(BLER)",
            }
    return None


def main():
    result_dir = Path("results")
    points = load_points(result_dir)
    summary = {
        "kind": "site_variance_ep_waterfall_summary",
        "input_files": list(RESULT_FILES),
        "series": SERIES,
        "points": {},
        "knees": {},
    }
    for snr in sorted(points):
        summary["points"][f"{snr:.3f}"] = {
            name: {
                key: row[key]
                for key in (
                    "blocks",
                    "failures",
                    "crc_bler",
                    "wilson_95",
                    "payload_ber",
                    "paired_vs_constant",
                    "mean_network_forwards_per_decode",
                )
                if key in row
            }
            for name, row in points[snr]["rows"].items()
            if name in SERIES
        }
    for name in SERIES:
        summary["knees"][name] = {
            "bler_1e-1": interpolate_knee(points, name, 1.0e-1),
            "bler_1e-2": interpolate_knee(points, name, 1.0e-2),
        }
    result_dir.joinpath("site_variance_ep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    colors = ("#2b6cb0", "#c53030", "#d69e2e")
    markers = ("o", "s", "^")
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for (name, label), color, marker in zip(SERIES.items(), colors, markers):
        xs, ys, lows, highs = [], [], [], []
        for snr in sorted(points):
            row = points[snr]["rows"].get(name)
            if row is None:
                continue
            xs.append(snr)
            estimate = row["crc_bler"]
            plot_value = estimate if estimate > 0.0 else 0.5 / row["blocks"]
            ys.append(plot_value)
            lows.append(max(row["wilson_95"][0], 2.0e-4))
            highs.append(row["wilson_95"][1])
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        lows = np.asarray(lows)
        highs = np.asarray(highs)
        ax.plot(xs, ys, marker=marker, linewidth=2.0, markersize=6, color=color, label=label)
        ax.fill_between(xs, lows, highs, color=color, alpha=0.13, linewidth=0)
    ax.axhline(1.0e-1, color="0.45", linestyle="--", linewidth=0.9)
    ax.axhline(1.0e-2, color="0.45", linestyle=":", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_ylim(2.0e-4, 1.05)
    ax.set_xlim(-2.88, -2.42)
    ax.set_xlabel("Es/N0 (dB)")
    ax.set_ylabel("CRC BLER (hard decision)")
    ax.set_title("Damped EP with pixelwise channel/source variance")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="lower left", fontsize=9)
    fig.savefig("site_variance_ep_waterfall.png", dpi=180)


if __name__ == "__main__":
    main()
