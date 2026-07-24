"""Plot the fair sigma-corrected fading comparison from its JSON artifact."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.load(open(args.input, encoding="utf-8"))
    schemes = ("legacy", "ep", "altproj")
    labels = {
        "legacy": "legacy (α=.15, β=0)",
        "ep": "EP (α_ep=.01, β_ep=1)",
        "altproj": "altproj (δ=.02, ρ=.9)",
    }
    colors = {"legacy": "#277DA1", "ep": "#F8961E", "altproj": "#43AA8B"}
    markers = {"legacy": "o", "ep": "s", "altproj": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharey=True)
    for ax, sigma_e2 in zip(axes, (0.0, 0.1, 0.2)):
        points = [
            point for point in payload["points"].values()
            if point["sigma_e2"] == sigma_e2
        ]
        points.sort(key=lambda item: item["ebno_db"])
        x = np.array([point["ebno_db"] for point in points])
        for scheme in schemes:
            y, lo, hi = [], [], []
            for point in points:
                row = next(
                    value for name, value in point["results"].items()
                    if name.startswith(scheme + "_")
                )
                p = float(row["crc_bler"])
                interval = row["wilson_95"]
                # Put observed zero at half an event for log plotting.  Its
                # Wilson interval remains visible and the caption states this.
                y.append(max(p, 0.5 / row["blocks"]))
                lo.append(max(float(interval[0]), 1e-4))
                hi.append(max(float(interval[1]), 0.5 / row["blocks"]))
            y = np.asarray(y)
            lo = np.asarray(lo)
            hi = np.asarray(hi)
            ax.plot(
                x, y, marker=markers[scheme], color=colors[scheme],
                linewidth=2, markersize=5, label=labels[scheme],
            )
            ax.fill_between(x, lo, hi, color=colors[scheme], alpha=0.12)
        ax.set_yscale("log")
        ax.set_ylim(7e-4, 1.2)
        ax.set_xlim(-0.15, 6.15)
        ax.grid(True, which="both", alpha=0.25)
        title = "perfect CSI" if sigma_e2 == 0 else "imperfect CSI"
        ax.set_title(rf"$\sigma_e^2={sigma_e2:g}$ ({title})")
        ax.set_xlabel("$E_b/N_0$ (dB)")
    axes[0].set_ylabel("CRC BLER")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Sigma-corrected 3-way fading comparison "
        "(QPSK fast Rayleigh, method-B LLR, 512 blocks/point)"
    )
    fig.text(
        0.5, 0.01,
        "Shading: Wilson 95% CI. Observed zero is displayed at 0.5/512.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
