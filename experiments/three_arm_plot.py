#!/usr/bin/env python3
"""Merge arm-process JSON files and render the common-resource waterfall."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.waterfall_common import interpolate_log_bler_knee  # noqa: E402


LABELS = {
    "ldpc_only": "A. 5G LDPC only (BP-100)",
    "ldpc_score_legacy": "B. 5G LDPC + score ([5]x20)",
    "rsc_score_spc": "C. RSC/BCJR + score + SPC",
    "rsc_no_spc": "RSC/BCJR only",
    "rsc_spc": "RSC/BCJR + SPC",
}
COLORS = {
    "ldpc_only": "#1f77b4",
    "ldpc_score_legacy": "#2ca02c",
    "rsc_score_spc": "#d62728",
    "rsc_no_spc": "#7f7f7f",
    "rsc_spc": "#9467bd",
}
MARKERS = {
    "ldpc_only": "o",
    "ldpc_score_legacy": "s",
    "rsc_score_spc": "^",
    "rsc_no_spc": "x",
    "rsc_spc": "D",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected: dict[tuple[str, float], dict] = {}
    sources = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path))
        for row in payload["rows"]:
            arm = row["arm"]
            if arm not in LABELS:
                continue
            key = (arm, float(row["esn0_db"]))
            previous = selected.get(key)
            if previous is None or int(row["blocks"]) > int(previous["blocks"]):
                selected[key] = row

    rows = sorted(selected.values(), key=lambda row: (row["arm"], row["esn0_db"]))
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    knees = {}
    for arm, group in by_arm.items():
        crc_rows = [
            {**row, "true_payload_bler": row["crc_detected_bler"]}
            for row in group
        ]
        knees[arm] = {
            "crc_bler_1e-1_esn0_db": interpolate_log_bler_knee(crc_rows, 1e-1),
            "crc_bler_1e-2_esn0_db": interpolate_log_bler_knee(crc_rows, 1e-2),
            "true_bler_1e-1_esn0_db": interpolate_log_bler_knee(group, 1e-1),
            "true_bler_1e-2_esn0_db": interpolate_log_bler_knee(group, 1e-2),
        }

    output = {
        "experiment": "three_arm_common_resource_waterfall",
        "source_json": sources,
        "selection_rule": "highest block count for duplicate arm/EsN0 rows",
        "primary_bler": "CRC-detected BLER; CRC pass defines success",
        "knee_interpolation": "linear Es/N0 versus log10(BLER)",
        "knees": knees,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    columns = [
        "arm", "esn0_db", "blocks", "true_payload_block_errors",
        "true_payload_bler", "true_payload_bler_wilson95", "bit_errors", "ber",
        "crc_detected_block_errors", "crc_detected_bler",
        "crc_detected_bler_wilson95", "undetected_errors", "elapsed_seconds",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["true_payload_bler_wilson95"] = json.dumps(
                row["true_payload_bler_wilson95"])
            serializable["crc_detected_bler_wilson95"] = json.dumps(
                row["crc_detected_bler_wilson95"])
            writer.writerow(serializable)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    for arm in ["ldpc_only", "ldpc_score_legacy", "rsc_score_spc", "rsc_no_spc", "rsc_spc"]:
        group = by_arm.get(arm)
        if not group:
            continue
        group = sorted(group, key=lambda row: row["esn0_db"])
        x = np.asarray([row["esn0_db"] for row in group], dtype=float)
        point = np.asarray([
            row["crc_detected_bler"]
            if row["crc_detected_block_errors"] > 0 else 0.5 / row["blocks"]
            for row in group
        ])
        lower = np.asarray([row["crc_detected_bler_wilson95"][0] for row in group])
        upper = np.asarray([row["crc_detected_bler_wilson95"][1] for row in group])
        plotted_lower = np.maximum(lower, 1e-5)
        error = np.vstack([point - plotted_lower, upper - point])
        error = np.maximum(error, 0.0)
        axis.errorbar(
            x,
            point,
            yerr=error,
            label=LABELS[arm],
            color=COLORS[arm],
            marker=MARKERS[arm],
            linewidth=1.8,
            markersize=5.5,
            capsize=2.5,
        )
    axis.set_yscale("log")
    axis.set_ylim(1e-4, 1.05)
    axis.set_xlabel("Es/N0 (dB)")
    axis.set_ylabel("CRC-detected BLER (CRC fail = block error)")
    axis.set_title("Common resource waterfall: 6272 payload bits, N=12600, AWGN")
    axis.grid(True, which="both", alpha=0.28)
    axis.legend(loc="best", fontsize=8.5)
    figure.tight_layout()
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_plot, dpi=220)
    plt.close(figure)
    print(f"wrote {args.output_json}, {args.output_csv}, {args.output_plot}")


if __name__ == "__main__":
    main()
