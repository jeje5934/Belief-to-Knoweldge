"""Consolidate and plot the equal-budget source-vs-codec waterfalls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BUDGETS = (100, 50, 30, 20)
CODECS = ("webp", "gzip", "png")
COLORS = {
    "ours": "#D1495B",
    "webp": "#00798C",
    "gzip": "#30638E",
    "png": "#6A4C93",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def merge_point(target, point):
    key = f"{float(point['esn0_db']):.3f}"
    previous = target.get(key)
    if previous is None or int(point["blocks"]) > int(previous["blocks"]):
        target[key] = dict(point)


def ours_points(primary, low):
    result = {}
    for payload in (primary, low):
        for point in payload["points"].values():
            merge_point(result, point)
    return result


def codec_points(primary, low, codec, budget):
    result = {}
    for payload in (primary, low):
        data = payload["codecs"][codec]
        for point in data["points"].values():
            row = {
                "esn0_db": point["esn0_db"],
                **point["budgets"][str(budget)],
            }
            merge_point(result, row)
    return result


def interpolate(rows, target):
    rows = sorted(rows, key=lambda row: row["esn0_db"])
    for left, right in zip(rows[:-1], rows[1:]):
        y0, y1 = left["crc_bler"], right["crc_bler"]
        if (y0 - target) * (y1 - target) > 0 or y0 == y1:
            continue
        fraction = (target - y0) / (y1 - y0)
        knee = left["esn0_db"] + fraction * (
            right["esn0_db"] - left["esn0_db"]
        )
        actual = left["actual_mean_bp_iters"] + fraction * (
            right["actual_mean_bp_iters"]
            - left["actual_mean_bp_iters"]
        )
        return {
            "esn0_db": float(knee),
            "actual_mean_bp_iters": float(actual),
            "bracket": [
                {
                    "esn0_db": left["esn0_db"],
                    "blocks": left["blocks"],
                    "failures": left["failures"],
                    "crc_bler": left["crc_bler"],
                    "wilson_95": left["wilson_95"],
                },
                {
                    "esn0_db": right["esn0_db"],
                    "blocks": right["blocks"],
                    "failures": right["failures"],
                    "crc_bler": right["crc_bler"],
                    "wilson_95": right["wilson_95"],
                },
            ],
        }
    return None


def sorted_rows(points):
    return sorted(points.values(), key=lambda row: row["esn0_db"])


def draw_curve(axis, rows, label, color, marker):
    rows = sorted_rows(rows)
    x = np.asarray([row["esn0_db"] for row in rows])
    y = np.asarray(
        [
            max(row["crc_bler"], 0.5 / row["blocks"])
            for row in rows
        ]
    )
    lo = np.asarray(
        [max(row["wilson_95"][0], 1.0e-5) for row in rows]
    )
    hi = np.asarray(
        [
            max(row["wilson_95"][1], 0.5 / row["blocks"])
            for row in rows
        ]
    )
    axis.plot(
        x,
        y,
        color=color,
        marker=marker,
        linewidth=1.8,
        markersize=4.0,
        label=label,
    )
    axis.fill_between(x, lo, hi, color=color, alpha=0.08)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--plot-output", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)

    ours_primary = {
        budget: load(root / f"low_budget_ours_b{budget}_waterfall.json")
        for budget in BUDGETS
    }
    ours_low = {
        budget: load(root / f"low_budget_ours_b{budget}_lowbler_3200.json")
        for budget in BUDGETS
    }
    codec_primary = {
        codec: load(root / f"low_budget_codec_{codec}_1024.json")
        for codec in CODECS
    }
    codec_low = {
        codec: load(root / f"low_budget_codec_{codec}_lowbler_3200.json")
        for codec in CODECS
    }

    consolidated = {
        "kind": "low_budget_source_vs_commercial_codec_duel",
        "method": {
            "knee_interpolation": "linear in BLER",
            "duplicate_snr_rule": (
                "when 1024- and 3200-block points share an SNR, retain 3200"
            ),
            "targets": [0.1, 0.01],
            "pixelcnn_reference": {
                "budget": 100,
                "knee_bler_0.1_db": -3.79,
                "note": "retained reference; lower budgets not remeasured",
            },
        },
        "ours": {},
        "codecs": {codec: {} for codec in CODECS},
        "comparison": {},
    }

    for budget in BUDGETS:
        points = ours_points(ours_primary[budget], ours_low[budget])
        consolidated["ours"][str(budget)] = {
            "configuration": ours_primary[budget]["configuration"],
            "points": points,
            "knees": {
                "bler_0.1": interpolate(sorted_rows(points), 0.1),
                "bler_0.01": interpolate(sorted_rows(points), 0.01),
            },
        }
        for codec in CODECS:
            codec_rows = codec_points(
                codec_primary[codec],
                codec_low[codec],
                codec,
                budget,
            )
            metadata = codec_primary[codec]["codecs"][codec]
            consolidated["codecs"][codec][str(budget)] = {
                "container_bits": metadata["container_bits"],
                "ldpc_k": metadata["ldpc_k"],
                "ldpc_rate": metadata["ldpc_rate"],
                "points": codec_rows,
                "knees": {
                    "bler_0.1": interpolate(
                        sorted_rows(codec_rows),
                        0.1,
                    ),
                    "bler_0.01": interpolate(
                        sorted_rows(codec_rows),
                        0.01,
                    ),
                },
            }

        comparison = {}
        for target_name in ("bler_0.1", "bler_0.01"):
            ours_knee = consolidated["ours"][str(budget)]["knees"][
                target_name
            ]["esn0_db"]
            comparison[target_name] = {
                "ours": ours_knee,
                "commercial": {},
            }
            for codec in CODECS:
                codec_knee = consolidated["codecs"][codec][str(budget)][
                    "knees"
                ][target_name]["esn0_db"]
                comparison[target_name]["commercial"][codec] = {
                    "knee": codec_knee,
                    "ours_minus_codec_db": float(ours_knee - codec_knee),
                    "ours_better": bool(ours_knee < codec_knee),
                }
        consolidated["comparison"][str(budget)] = comparison

    dump(args.json_output, consolidated)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    markers = {"ours": "o", "webp": "s", "gzip": "^", "png": "D"}
    for axis, budget in zip(axes.flat, BUDGETS):
        draw_curve(
            axis,
            consolidated["ours"][str(budget)]["points"],
            "Ours AltProj",
            COLORS["ours"],
            markers["ours"],
        )
        for codec in CODECS:
            draw_curve(
                axis,
                consolidated["codecs"][codec][str(budget)]["points"],
                f"{codec.upper()}-MAX BP",
                COLORS[codec],
                markers[codec],
            )
        axis.axhline(0.1, color="#444444", linestyle="--", linewidth=0.8)
        axis.axhline(0.01, color="#777777", linestyle=":", linewidth=0.8)
        axis.set_yscale("log")
        axis.set_ylim(8.0e-4, 1.15)
        axis.grid(True, which="both", alpha=0.22)
        axis.set_title(f"Equal nominal BP budget = {budget}")
        axis.set_xlabel("Es/N0 (dB)")
        axis.set_ylabel("CRC BLER")
        axis.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Budget-retuned AltProj vs commercial lossless codecs\n"
        "payload=6272, N=12600, AWGN + perfect CSI",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(args.plot_output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_output, dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
