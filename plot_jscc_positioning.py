#!/usr/bin/env python3
"""Plot the measured operating point against published lossy-JSCC regions.

This is deliberately a positioning figure, not a head-to-head performance
comparison: datasets, SNRs, and training objectives differ across papers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OURS_RATIO = 6300 / 784
OURS_ESN0 = np.array([-3.20, -3.00, -2.90, -2.85, -2.80, -2.75, -2.70, -2.60])
OURS_PSNR = np.array([18.84, 22.01, 24.48, 25.67, 27.67, 28.78, 33.11, 36.03])

# Approximate visual digitization of DeepJSCC Fig. 4 (CIFAR-10, k/n=1/6).
# These are intentionally marked as approximate in both the figure and JSON.
DEEPJSCC_SNR = np.array([1.0, 4.0, 7.0, 13.0, 19.0])
DEEPJSCC_PSNR = np.array([24.6, 26.5, 28.5, 30.8, 34.0])

payloads = {
    "Fashion-MNIST 28x28x1": 28 * 28 * 1 * 8,
    "CIFAR-10 32x32x3": 32 * 32 * 3 * 8,
    "D2-JSCC training crop 256x256x3": 256 * 256 * 3 * 8,
    "Kodak 768x512x3": 768 * 512 * 3 * 8,
    "CLIC max 2048x1890x3": 2048 * 1890 * 3 * 8,
}

data = {
    "figure_kind": "cross-domain positioning; not a head-to-head comparison",
    "ratio_definition": "complex channel uses / source scalar (H*W*C)",
    "ours": {
        "dataset": "Fashion-MNIST 28x28 grayscale, 8 bit",
        "ratio": OURS_RATIO,
        "real_bpsk_uses": 12600,
        "equivalent_complex_uses": 6300,
        "esn0_db": OURS_ESN0.tolist(),
        "complex_symbol_snr_db_under_iq_packing": (OURS_ESN0 + 10 * np.log10(2)).tolist(),
        "psnr_db": OURS_PSNR.tolist(),
        "source": "results/mse_fairness_analysis_512.json",
    },
    "literature": [
        {
            "system": "D2-JSCC + polar",
            "dataset": "Kodak",
            "ratio": 0.0625,
            "snr_db": 2.0,
            "psnr_db": 27.2,
            "quality": "paper text, Fig. 7",
            "url": "https://arxiv.org/abs/2403.07338",
        },
        {
            "system": "D2-JSCC + polar",
            "dataset": "CLIC",
            "ratio": 0.0625,
            "snr_db": 2.0,
            "psnr_db": 30.8,
            "quality": "paper text, Fig. 7",
            "url": "https://arxiv.org/abs/2403.07338",
        },
        {
            "system": "NTSCC",
            "dataset": "Kodak image example",
            "ratio": 0.038,
            "snr_db": 10.0,
            "psnr_db": 34.22,
            "quality": "single-image example, Fig. 13; not a dataset mean",
            "url": "https://arxiv.org/abs/2112.10961",
        },
        {
            "system": "DeepJSCC",
            "dataset": "CIFAR-10",
            "ratio": 1 / 6,
            "snr_db": DEEPJSCC_SNR.tolist(),
            "psnr_db": DEEPJSCC_PSNR.tolist(),
            "quality": "approximate visual digitization of Fig. 4",
            "url": "https://arxiv.org/abs/1809.01733",
        },
    ],
    "raw_payload_bits": payloads,
}

(OUT_DIR / "jscc_positioning_data.json").write_text(
    json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)

fig, (ax, axr) = plt.subplots(
    1, 2, figsize=(13.8, 6.7), gridspec_kw={"width_ratios": [1.75, 1.0]}
)

# Published lossy-JSCC operating region.
ax.axvspan(0.02, 1 / 6, color="#d5e8d4", alpha=0.45, zorder=0)
ax.text(0.023, 38.4, "published lossy-JSCC\noperating region", color="#315b3a")

ax.plot(
    np.full_like(DEEPJSCC_PSNR, 1 / 6),
    DEEPJSCC_PSNR,
    "o--",
    color="#6a51a3",
    label="DeepJSCC, CIFAR-10, r=1/6 (Fig. 4, approx.)",
)
for x, y, s in zip(np.full_like(DEEPJSCC_PSNR, 1 / 6), DEEPJSCC_PSNR, DEEPJSCC_SNR):
    ax.annotate(f"{s:g} dB", (x, y), xytext=(7, 0), textcoords="offset points", fontsize=8)

ax.scatter([0.0625], [27.2], marker="s", s=62, color="#238b45", zorder=4)
ax.scatter([0.0625], [30.8], marker="D", s=56, color="#238b45", zorder=4)
ax.annotate("D²-JSCC Kodak\n2 dB, 27.2 dB", (0.0625, 27.2), xytext=(-80, -32),
            textcoords="offset points", fontsize=8, arrowprops={"arrowstyle": "-", "lw": 0.7})
ax.annotate("D²-JSCC CLIC\n2 dB, 30.8 dB", (0.0625, 30.8), xytext=(-80, 18),
            textcoords="offset points", fontsize=8, arrowprops={"arrowstyle": "-", "lw": 0.7})

ax.scatter([0.038], [34.22], marker="*", s=100, facecolors="none", edgecolors="#d95f0e", zorder=4)
ax.annotate("NTSCC single Kodak example\n10 dB, r=.038, 34.22 dB", (0.038, 34.22),
            xytext=(12, 14), textcoords="offset points", fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.7})

ax.plot(
    np.full_like(OURS_PSNR, OURS_RATIO),
    OURS_PSNR,
    "o-",
    lw=2.2,
    color="#1f78b4",
    label="ours, Fashion-MNIST, r=8.04 (512 blocks/SNR)",
)
for y, esn0 in zip(OURS_PSNR, OURS_ESN0):
    ax.annotate(f"Es/N0 {esn0:g}", (OURS_RATIO, y), xytext=(7, 0),
                textcoords="offset points", fontsize=8)

ax.annotate(
    "48× more uses than r=1/6\n129× more than r=1/16",
    xy=(OURS_RATIO, 20.2),
    xytext=(0.38, 20.2),
    arrowprops={"arrowstyle": "<->", "lw": 1.2, "color": "#555555"},
    va="center",
    fontsize=9,
)

ax.set_xscale("log")
ax.set_xlim(0.017, 15)
ax.set_ylim(17, 40)
ax.set_xlabel("Bandwidth ratio r = complex channel uses / source scalar (H×W×C)")
ax.set_ylabel("PSNR (dB)")
ax.set_title("Resource–quality positioning (cross-domain; not head-to-head)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower left", frameon=True)

labels = ["D² convergence", "NTSCC / D² main", "DeepJSCC-Q max", "DeepJSCC max", "ours"]
ratios = [0.022, 0.0625, 0.15, 1 / 6, OURS_RATIO]
colors = ["#74c476", "#41ab5d", "#fd8d3c", "#6a51a3", "#1f78b4"]
y = np.arange(len(labels))
axr.barh(y, ratios, color=colors, alpha=0.86)
axr.set_yticks(y, labels)
axr.invert_yaxis()
axr.set_xscale("log")
axr.set_xlim(0.015, 15)
axr.set_xlabel("complex channel uses / source scalar")
axr.set_title("Published resource scales")
axr.grid(True, axis="x", which="both", alpha=0.25)
for yi, ratio in zip(y, ratios):
    axr.text(ratio * 1.08, yi, f"{ratio:.4g}", va="center", fontsize=9)

fig.suptitle("Lossy JSCC and receiver-side source-aided LDPC occupy different resource regimes", fontsize=14)
fig.text(
    0.5,
    0.012,
    "Literature points use different datasets/SNR/training. NTSCC star is one image, DeepJSCC values are visually digitized; "
    "they must not be read as a performance ranking.",
    ha="center",
    fontsize=8,
    color="#555555",
)
fig.tight_layout(rect=[0, 0.045, 1, 0.95])
fig.savefig(OUT_DIR / "jscc_ratio_psnr_positioning.png", bbox_inches="tight")
print(OUT_DIR / "jscc_ratio_psnr_positioning.png")
print(OUT_DIR / "jscc_positioning_data.json")
