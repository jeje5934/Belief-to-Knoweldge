"""Aggregate abstract latency-matched BLER frontiers and render figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


L_DEN_VALUES = (20.0, 50.0, 100.0)
SYSTEMS = ("ours", "pixelcnn", "webp", "raw")
LABELS = {
    "ours": "Ours (best source scheme)",
    "pixelcnn": "PixelCNN-MAX + BP",
    "webp": "WebP-MAX + BP",
    "raw": "Raw + BP only",
}
COLORS = {
    "ours": "#D1495B",
    "pixelcnn": "#2A9D8F",
    "webp": "#00798C",
    "raw": "#555555",
}
MARKERS = {"ours": "o", "pixelcnn": "s", "webp": "^", "raw": "D"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def selected_source_points(source, snr_key):
    points = []
    for group_name, group in source["points"][snr_key]["groups"].items():
        name = group["selected"]
        row = group["rows"][name]
        points.append(
            {
                "configuration": name,
                "group": group_name,
                "nominal_bp_budget": group["nominal_bp_budget"],
                "bp": row["actual_mean_bp_iters"],
                "source_calls": row["actual_mean_source_calls"],
                "bler": row["crc_bler"],
                "wilson_95": row["wilson_95"],
                "blocks": row["blocks"],
            }
        )
    return points


def baseline_points(baseline, system, snr_key):
    rows = baseline["systems"][system]["points"][snr_key]["budgets"]
    return [
        {
            "configuration": f"BP-{budget}",
            "nominal_bp_budget": int(budget),
            "bp": row["actual_mean_bp_iters"],
            "source_calls": 0.0,
            "bler": row["crc_bler"],
            "wilson_95": row["wilson_95"],
            "blocks": row["blocks"],
        }
        for budget, row in rows.items()
    ]


def add_latency(points, system, l_den, include_tx, tx_latency):
    output = []
    for point in points:
        row = dict(point)
        row["rx_latency"] = float(
            row["bp"] + row["source_calls"] * l_den
        )
        row["tx_latency"] = float(tx_latency[system] if include_tx else 0.0)
        row["latency"] = row["rx_latency"] + row["tx_latency"]
        output.append(row)
    return output


def frontier(points):
    """Best measured BLER attainable under each observed mean-latency cap."""
    ordered = sorted(points, key=lambda row: (row["latency"], row["bler"]))
    result = []
    best = None
    for row in ordered:
        if best is None or row["bler"] < best["bler"]:
            best = row
            result.append(row)
    return result


def state_at(points, latency):
    eligible = [row for row in points if row["latency"] <= latency + 1.0e-9]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (row["bler"], row["latency"]))


def winner_segments(system_points):
    thresholds = sorted(
        {row["latency"] for points in system_points.values() for row in points}
    )
    segments = []
    for index, start in enumerate(thresholds):
        states = {
            system: state_at(points, start)
            for system, points in system_points.items()
        }
        available = {k: v for k, v in states.items() if v is not None}
        best_bler = min(row["bler"] for row in available.values())
        winners = sorted(
            system
            for system, row in available.items()
            if abs(row["bler"] - best_bler) < 1.0e-12
        )
        ours = states.get("ours")
        segment = {
            "start": float(start),
            "end": (
                float(thresholds[index + 1])
                if index + 1 < len(thresholds)
                else None
            ),
            "winners": winners,
            "best_bler": float(best_bler),
            "ours_bler": None if ours is None else float(ours["bler"]),
            "ours_practical_bler_0.1": bool(
                ours is not None and ours["bler"] <= 0.1
            ),
            "ours_practical_bler_0.01": bool(
                ours is not None and ours["bler"] <= 0.01
            ),
        }
        if segments and all(
            segments[-1][key] == segment[key]
            for key in (
                "winners",
                "best_bler",
                "ours_bler",
                "ours_practical_bler_0.1",
                "ours_practical_bler_0.01",
            )
        ):
            segments[-1]["end"] = segment["end"]
        else:
            segments.append(segment)
    return segments


def plot_panel(axis, system_points, title):
    for system in SYSTEMS:
        points = system_points[system]
        curve = frontier(points)
        x = np.asarray([row["latency"] for row in curve])
        y = np.asarray(
            [max(row["bler"], 0.5 / row["blocks"]) for row in curve]
        )
        axis.step(
            x,
            y,
            where="post",
            color=COLORS[system],
            linewidth=2.0,
            label=LABELS[system],
        )
        axis.scatter(
            x,
            y,
            color=COLORS[system],
            marker=MARKERS[system],
            s=26,
            zorder=3,
        )
    axis.axhline(0.1, color="#555555", linestyle="--", linewidth=0.9)
    axis.axhline(0.01, color="#777777", linestyle=":", linewidth=0.9)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(4.0e-4, 1.15)
    axis.grid(True, which="both", alpha=0.22)
    axis.set_title(title)
    axis.set_xlabel("abstract mean latency (BP-iteration units)")
    axis.set_ylabel("CRC BLER")


def render_snr_figure(path, snr, panels):
    fig, axes = plt.subplots(3, 2, figsize=(14.4, 13.2), sharey=True)
    for row_index, l_den in enumerate(L_DEN_VALUES):
        for col_index, mode in enumerate(("rx_only", "tx_included")):
            plot_panel(
                axes[row_index, col_index],
                panels[str(int(l_den))][mode]["system_points"],
                (
                    f"L_den={int(l_den)}; "
                    + ("receiver only" if mode == "rx_only" else "TX + receiver")
                ),
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"Latency-matched BLER frontiers at Es/N0={snr:+.1f} dB\n"
        "payload=6272, N=12600, AWGN + perfect CSI",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_key_figure(path, all_panels, snrs):
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.6), sharey=True)
    for row_index, snr in enumerate(snrs):
        snr_panels = all_panels[f"{snr:.3f}"]["50"]
        for col_index, mode in enumerate(("rx_only", "tx_included")):
            plot_panel(
                axes[row_index, col_index],
                snr_panels[mode]["system_points"],
                (
                    f"Es/N0={snr:+.1f} dB, L_den=50, "
                    + ("receiver only" if mode == "rx_only" else "TX + receiver")
                ),
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Professor-discussion view: latency-matched reliability",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-prefix", required=True)
    args = parser.parse_args()
    source = load(args.source)
    baseline = load(args.baseline)
    profile = load(args.profile)
    snrs = sorted(
        float(key) for key in set(source["points"]) & {
            key
            for system in baseline["systems"].values()
            for key in system["points"]
        }
    )
    pixelcnn_tx = float(profile["pixelcnn"]["encoder_depth_units"])
    tx_latency = {
        "ours": 0.0,
        "raw": 0.0,
        # Fixed small-library allowance.  It is an explicit convention, not a
        # wall-clock claim; setting it to zero moves WebP left by only 20 units.
        "webp": 20.0,
        "pixelcnn": pixelcnn_tx,
    }
    result = {
        "kind": "abstract_latency_matched_bler",
        "latency_model": {
            "bp_iteration": 1.0,
            "denoiser_values": L_DEN_VALUES,
            "denoiser_architectural_midpoint": profile["songunet"][
                "critical_path_stage_estimate"
            ],
            "serial_composition": "mean BP iters + mean source calls * L_den",
            "crc_early_stop": "actual first-pass mean per block",
            "tx_latency": tx_latency,
            "pixelcnn_symbol_only_lower_bound": profile["pixelcnn"][
                "optimistic_symbol_only_lower_bound"
            ],
            "webp_tx_note": (
                "20-unit fixed small-library allowance; not hardware measured"
            ),
            "parallelism_assumption": (
                "unlimited parallel resources inside one BP iteration, one "
                "UNet layer, and across codewords/spatial positions"
            ),
            "latency_statistic": "mean per-block critical-path latency",
            "curve_semantics": (
                "step frontier: best measured BLER attainable under a mean "
                "latency cap; no interpolation between configurations"
            ),
        },
        "architecture_profile": profile,
        "snrs": {},
    }
    all_panels = {}
    prefix = Path(args.plot_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    for snr in snrs:
        snr_key = f"{snr:.3f}"
        raw_points = {
            "ours": selected_source_points(source, snr_key),
            "pixelcnn": baseline_points(baseline, "pixelcnn", snr_key),
            "webp": baseline_points(baseline, "webp", snr_key),
            "raw": baseline_points(baseline, "raw", snr_key),
        }
        snr_result = {}
        plot_panels = {}
        for l_den in L_DEN_VALUES:
            lden_key = str(int(l_den))
            snr_result[lden_key] = {}
            plot_panels[lden_key] = {}
            for mode, include_tx in (("rx_only", False), ("tx_included", True)):
                system_points = {
                    system: add_latency(
                        points, system, l_den, include_tx, tx_latency
                    )
                    for system, points in raw_points.items()
                }
                segments = winner_segments(system_points)
                serializable_points = {
                    system: points for system, points in system_points.items()
                }
                panel = {
                    "system_points": serializable_points,
                    "frontiers": {
                        system: frontier(points)
                        for system, points in system_points.items()
                    },
                    "winner_segments": segments,
                    "ours_unique_segments": [
                        row for row in segments if row["winners"] == ["ours"]
                    ],
                    "ours_unique_practical_0.1": [
                        row
                        for row in segments
                        if row["winners"] == ["ours"]
                        and row["ours_practical_bler_0.1"]
                    ],
                    "ours_unique_practical_0.01": [
                        row
                        for row in segments
                        if row["winners"] == ["ours"]
                        and row["ours_practical_bler_0.01"]
                    ],
                }
                snr_result[lden_key][mode] = panel
                plot_panels[lden_key][mode] = panel
        result["snrs"][snr_key] = snr_result
        all_panels[snr_key] = plot_panels
        render_snr_figure(
            prefix.with_name(prefix.name + f"_m{abs(snr):.1f}.png"),
            snr,
            plot_panels,
        )

    render_key_figure(
        prefix.with_name(prefix.name + "_key.png"), all_panels, snrs
    )
    dump(args.output, result)


if __name__ == "__main__":
    main()
