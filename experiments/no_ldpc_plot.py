#!/usr/bin/env python3
"""Plot no-LDPC smoke BLER with Wilson intervals and an explicit warning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def wilson_interval(errors: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = errors / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/no_ldpc_smoke_bler.png"))
    parser.add_argument(
        "--title", default="No-LDPC paired BLER (SMOKE ONLY — no SNR-knee claim)")
    return parser.parse_args()


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data.get("results", data.get("rows"))
    if not rows:
        raise ValueError("input JSON contains no result rows")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scheme"], []).append(row)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for scheme, scheme_rows in sorted(grouped.items()):
        scheme_rows = sorted(scheme_rows, key=lambda row: row["esn0_db"])
        x = np.asarray([row["esn0_db"] for row in scheme_rows], dtype=float)
        y = np.asarray([row["true_payload_bler"] for row in scheme_rows], dtype=float)
        lower = []
        upper = []
        for row, estimate in zip(scheme_rows, y):
            lo, hi = wilson_interval(
                int(row["true_payload_block_errors"]), int(row["blocks"]))
            lower.append(estimate - lo)
            upper.append(hi - estimate)
        axis.errorbar(
            x,
            y,
            yerr=np.asarray([lower, upper]),
            marker="o",
            capsize=4,
            linewidth=1.5,
            label=(
                "bcjr_only_same_spc_frame"
                if scheme == "bcjr_only" else scheme),
        )

    axis.set_xlabel("Es/N0 (dB)")
    axis.set_ylabel("True payload BLER")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.set_title(args.title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
