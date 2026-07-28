"""Fixed-latency SNR-BLER maps from measured source and codec points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LATENCY_BUDGETS = (40, 300, 1100)
L_DEN_VALUES = (20, 50, 100)
BASELINE_BUDGETS = (20, 30, 50, 100, 200)
SYSTEMS = ("ours", "pixelcnn", "webp", "raw")
LABELS = {
    "ours": "Ours (latency-fit LUT)",
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

OURS_CONFIGS = {
    "bp10": {"bp": 10, "calls": 0, "rank": 0},
    "10x2": {"bp": 20, "calls": 1, "rank": 1},
    "5x4": {"bp": 20, "calls": 3, "rank": 2},
    "5x6": {"bp": 30, "calls": 5, "rank": 3, "budget": 30},
    "5x10": {"bp": 50, "calls": 9, "rank": 4, "budget": 50},
    "5x20": {"bp": 100, "calls": 19, "rank": 5, "budget": 100},
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


def normalize_row(row, esn0_db):
    return {
        "esn0_db": float(esn0_db),
        "blocks": int(row["blocks"]),
        "failures": int(row["failures"]),
        "crc_bler": float(row["crc_bler"]),
        "wilson_95": [float(value) for value in row["wilson_95"]],
    }


def merge_row(target, row):
    key = f"{row['esn0_db']:.3f}"
    previous = target.get(key)
    if previous is None or row["blocks"] > previous["blocks"]:
        target[key] = row


def merge_source_file(curves, payload):
    for snr_key, point in payload["points"].items():
        for group_name, group in point["groups"].items():
            selected = group["selected"]
            row = normalize_row(group["rows"][selected], float(snr_key))
            merge_row(curves[group_name], row)


def source_curves(source_files, duel):
    curves = {name: {} for name in OURS_CONFIGS}
    for payload in source_files:
        merge_source_file(curves, payload)
    for name, metadata in OURS_CONFIGS.items():
        budget = metadata.get("budget")
        if budget is None:
            continue
        for row in duel["ours"][str(budget)]["points"].values():
            merge_row(curves[name], normalize_row(row, row["esn0_db"]))
    return curves


def merge_baseline_payload(curves, payload):
    for system, system_payload in payload["systems"].items():
        for snr_key, point in system_payload["points"].items():
            for budget, row in point["budgets"].items():
                normalized = normalize_row(row, float(snr_key))
                merge_row(curves[system][int(budget)], normalized)


def baseline_curves(baseline_files, duel):
    curves = {
        system: {budget: {} for budget in BASELINE_BUDGETS}
        for system in ("pixelcnn", "webp", "raw")
    }
    for payload in baseline_files:
        merge_baseline_payload(curves, payload)
    for budget in (20, 30, 50, 100):
        for row in duel["codecs"]["webp"][str(budget)]["points"].values():
            merge_row(
                curves["webp"][budget],
                normalize_row(row, row["esn0_db"]),
            )
    return curves


def choose_ours(latency_budget, l_den):
    eligible = []
    for name, metadata in OURS_CONFIGS.items():
        nominal = metadata["bp"] + metadata["calls"] * l_den
        if nominal <= latency_budget:
            eligible.append((metadata["rank"], name, nominal))
    if not eligible:
        return None
    _, name, nominal = max(eligible)
    return {"configuration": name, "nominal_latency": float(nominal)}


def choose_baseline(latency_budget, system, include_tx, tx_latency):
    allowance = latency_budget - (tx_latency[system] if include_tx else 0)
    eligible = [budget for budget in BASELINE_BUDGETS if budget <= allowance]
    if not eligible:
        return None
    budget = max(eligible)
    return {
        "configuration": f"BP-{budget}",
        "bp_budget": budget,
        "nominal_latency": float(
            budget + (tx_latency[system] if include_tx else 0)
        ),
    }


def sorted_rows(rows):
    return sorted(rows.values(), key=lambda row: row["esn0_db"])


def interpolate_knee(rows, target):
    rows = sorted(rows, key=lambda row: row["esn0_db"])
    for left, right in zip(rows[:-1], rows[1:]):
        y0, y1 = left["crc_bler"], right["crc_bler"]
        if (y0 - target) * (y1 - target) > 0 or y0 == y1:
            continue
        fraction = (target - y0) / (y1 - y0)
        return float(
            left["esn0_db"]
            + fraction * (right["esn0_db"] - left["esn0_db"])
        )
    return None


def exact_point(rows, snr):
    key = f"{snr:.3f}"
    return rows.get(key)


def draw_curve(axis, rows, system):
    data = sorted_rows(rows)
    if not data:
        return
    x = np.asarray([row["esn0_db"] for row in data])
    y = np.asarray(
        [max(row["crc_bler"], 0.5 / row["blocks"]) for row in data]
    )
    lo = np.asarray(
        [max(row["wilson_95"][0], 2.0e-4) for row in data]
    )
    hi = np.asarray(
        [max(row["wilson_95"][1], 0.5 / row["blocks"]) for row in data]
    )
    axis.plot(
        x,
        y,
        color=COLORS[system],
        marker=MARKERS[system],
        linewidth=1.9,
        markersize=4.5,
        label=LABELS[system],
    )
    axis.fill_between(x, lo, hi, color=COLORS[system], alpha=0.09)


def render(path, l_den, panels):
    fig, axes = plt.subplots(3, 2, figsize=(14.4, 13.2), sharex=True, sharey=True)
    for row_index, latency_budget in enumerate(LATENCY_BUDGETS):
        for col_index, mode in enumerate(("rx_only", "tx_included")):
            axis = axes[row_index, col_index]
            panel = panels[str(latency_budget)][mode]
            for system in SYSTEMS:
                curve = panel["curves"].get(system)
                if curve is not None:
                    draw_curve(axis, curve["points"], system)
            unavailable = [
                LABELS[system]
                for system in SYSTEMS
                if panel["curves"].get(system) is None
            ]
            if unavailable:
                axis.text(
                    0.02,
                    0.04,
                    "unavailable within L: " + ", ".join(unavailable),
                    transform=axis.transAxes,
                    fontsize=8.5,
                    color="#555555",
                )
            axis.axhline(0.1, color="#555555", linestyle="--", linewidth=0.8)
            axis.axhline(0.01, color="#777777", linestyle=":", linewidth=0.8)
            axis.set_yscale("log")
            axis.set_ylim(4.0e-4, 1.15)
            axis.set_xlim(-3.22, -2.28)
            axis.grid(True, which="both", alpha=0.22)
            axis.set_title(
                f"L={latency_budget}; "
                + ("receiver only" if mode == "rx_only" else "TX + receiver")
            )
            axis.set_xlabel("Es/N0 (dB)")
            axis.set_ylabel("CRC BLER")
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
        f"Fixed-latency SNR-BLER map (L_den={l_den})\n"
        "payload=6272, N=12600, AWGN + perfect CSI",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=195, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--duel", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-prefix", required=True)
    args = parser.parse_args()
    duel = load(args.duel)
    profile = load(args.profile)
    source = source_curves([load(path) for path in args.source], duel)
    baselines = baseline_curves(
        [load(path) for path in args.baselines], duel
    )
    tx_latency = {
        "ours": 0.0,
        "raw": 0.0,
        "webp": 20.0,
        "pixelcnn": float(profile["pixelcnn"]["encoder_depth_units"]),
    }
    result = {
        "kind": "fixed_latency_snr_bler_map",
        "model": {
            "latency_budgets": LATENCY_BUDGETS,
            "l_den_values": L_DEN_VALUES,
            "default_l_den": 50,
            "tx_latency": tx_latency,
            "selection_rule": (
                "largest pre-tuned LUT/BP configuration whose worst-case "
                "nominal serial depth fits L; selection is independent of BLER"
            ),
            "zero_failure_plot_floor": "0.5 / blocks",
            "curve_lines": (
                "visual guides between measured SNR points; knees use linear "
                "BLER interpolation consistent with earlier reports"
            ),
        },
        "panels": {},
    }
    prefix = Path(args.plot_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    for l_den in L_DEN_VALUES:
        lden_panels = {}
        for latency_budget in LATENCY_BUDGETS:
            budget_panels = {}
            for mode, include_tx in (("rx_only", False), ("tx_included", True)):
                curves = {}
                ours_selection = choose_ours(latency_budget, l_den)
                if ours_selection is None:
                    curves["ours"] = None
                else:
                    rows = source[ours_selection["configuration"]]
                    curves["ours"] = {
                        "selection": ours_selection,
                        "points": rows,
                        "knees": {
                            "bler_0.1": interpolate_knee(sorted_rows(rows), 0.1),
                            "bler_0.01": interpolate_knee(sorted_rows(rows), 0.01),
                        },
                    }
                for system in ("pixelcnn", "webp", "raw"):
                    selection = choose_baseline(
                        latency_budget, system, include_tx, tx_latency
                    )
                    if selection is None:
                        curves[system] = None
                        continue
                    rows = baselines[system][selection["bp_budget"]]
                    curves[system] = {
                        "selection": selection,
                        "points": rows,
                        "knees": {
                            "bler_0.1": interpolate_knee(sorted_rows(rows), 0.1),
                            "bler_0.01": interpolate_knee(sorted_rows(rows), 0.01),
                        },
                    }
                comparisons = {}
                for snr in (-2.7, -2.9):
                    rows_at_snr = {
                        system: (
                            None
                            if curve is None
                            else exact_point(curve["points"], snr)
                        )
                        for system, curve in curves.items()
                    }
                    available = {
                        system: row
                        for system, row in rows_at_snr.items()
                        if row is not None
                    }
                    if available:
                        best = min(row["crc_bler"] for row in available.values())
                        winners = sorted(
                            system
                            for system, row in available.items()
                            if abs(row["crc_bler"] - best) < 1.0e-12
                        )
                    else:
                        winners = []
                    comparisons[f"{snr:.3f}"] = {
                        "rows": rows_at_snr,
                        "winners": winners,
                    }
                budget_panels[mode] = {
                    "curves": curves,
                    "comparisons": comparisons,
                }
            lden_panels[str(latency_budget)] = budget_panels
        result["panels"][str(l_den)] = lden_panels
        render(
            prefix.with_name(prefix.name + f"_lden{l_den}.png"),
            l_den,
            lden_panels,
        )
    dump(args.output, result)


if __name__ == "__main__":
    main()
