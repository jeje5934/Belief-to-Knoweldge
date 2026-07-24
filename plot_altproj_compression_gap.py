"""Plot the final altproj/compression comparison and ablations."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GRID = [-2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0]


def load(path):
    return json.load(open(path, encoding="utf-8"))


def match_point(points, esn0):
    for point in points.values():
        value = point.get("esno_db", point.get("esn0_db"))
        if value is not None and abs(float(value) - esn0) < 1e-6:
            return point
    return None


def old_ours(payload, label, esn0):
    return payload.get(f"{label}@{esn0}")


def arrays(getter):
    x, y, lo, hi = [], [], [], []
    for esn0 in GRID:
        point = getter(esn0)
        if point is None:
            continue
        interval = (
            point.get("wilson_95")
            or point.get("bler_ci")
            or [point["ci_lo"], point["ci_hi"]]
        )
        blocks = int(point.get("blocks", point.get("total", point.get("N", 1024))))
        bler = (
            point["crc_bler"]
            if "crc_bler" in point
            else point["bler"]
        )
        x.append(esn0)
        y.append(max(float(bler), 0.5 / blocks))
        lo.append(max(float(interval[0]), 1e-5))
        hi.append(max(float(interval[1]), 0.5 / blocks))
    return np.asarray(x), np.asarray(y), np.asarray(lo), np.asarray(hi)


def draw(ax, getter, label, color, marker, linestyle="-"):
    x, y, lo, hi = arrays(getter)
    ax.plot(
        x, y, marker=marker, linestyle=linestyle, color=color,
        linewidth=1.9, markersize=4.5, label=label,
    )
    ax.fill_between(x, lo, hi, color=color, alpha=0.09)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    old_our = load(f"{args.baseline_dir}/our_waterfall.json")
    channel = load(f"{args.baseline_dir}/channel_bler.json")
    gzip = load(f"{args.baseline_dir}/gzip_bler.json")
    png = load(f"{args.baseline_dir}/png_bler.json")
    canonical = load(f"{args.new_dir}/altproj_compression_waterfall.json")
    best100 = load(f"{args.new_dir}/altproj_budget100_es_waterfall.json")
    best50 = load(f"{args.new_dir}/altproj_budget50_es_waterfall.json")
    best30 = load(f"{args.new_dir}/altproj_budget30_es_waterfall.json")
    spc = load(f"{args.new_dir}/altproj_spc_compression_waterfall.json")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    left, right = axes
    draw(
        left,
        lambda es: match_point(channel["configs"]["MAX"]["points"], es),
        "PixelCNN-MAX r=.389", "#2A9D8F", "o",
    )
    draw(
        left,
        lambda es: old_ours(old_our, "legacy_[5]x20", es),
        "legacy r=.500", "#457B9D", "v",
    )
    draw(
        left,
        lambda es: old_ours(old_our, "EP_[5]x20", es),
        "EP r=.500", "#8E6C8A", "^",
    )
    draw(
        left,
        lambda es: match_point(png["configs"]["webpMAX"]["points"], es),
        "WebP-MAX r=.463", "#F4A261", "D",
    )
    draw(
        left,
        lambda es: match_point(gzip["configs"]["gzipMAX"]["points"], es),
        "gzip-MAX r=.485", "#E76F51", "P",
    )
    draw(
        left,
        lambda es: match_point(best100["points"], es),
        "altproj best B100 r=.500", "#264653", "s",
        linestyle="--",
    )
    left.set_title("Final same-resource comparison")

    draw(
        right,
        lambda es: match_point(canonical["points"], es),
        "altproj canonical B100, ES off", "#264653", "o",
    )
    draw(
        right,
        lambda es: match_point(best100["points"], es),
        "altproj B100, ES on", "#2A9D8F", "s",
    )
    draw(
        right,
        lambda es: match_point(best50["points"], es),
        "altproj B50, ES on", "#F4A261", "^",
    )
    draw(
        right,
        lambda es: match_point(best30["points"], es),
        "altproj B30, ES on", "#E76F51", "v",
    )
    draw(
        right,
        lambda es: match_point(spc["points"], es),
        "altproj+SPC B100 r=.562", "#8E6C8A", "D",
        linestyle="--",
    )
    right.set_title("SPC and compute-budget ablations")

    for ax in axes:
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_ylim(8e-5, 1.4)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("$E_s/N_0$ (dB), N=12600")
        ax.axhline(0.1, color="black", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=8, loc="lower left")
    left.set_ylabel("CRC BLER")
    fig.suptitle("Altproj vs learned-compression baseline")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
