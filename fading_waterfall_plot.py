"""
Waterfall figure + table from the 3-way grid CSV (fading_3way output).
One panel per σ_e²; curves BP-only / legacy / EP (+ minus if present).
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results"


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            if k not in ("channel", "perfect_csi", "llr_method"):
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
    return rows


def main(csv_path="results/fading_3way_grid.csv"):
    rows = load(csv_path)
    se2s = sorted(set(r["sigma_e2"] for r in rows))
    series = [("bler_bp", "BP-only", "tab:gray", "x-"),
              ("bler_legacy", "legacy", "tab:blue", "o-"),
              ("bler_ep", "EP", "tab:red", "s-")]
    if any("bler_minus" in r for r in rows):
        series.append(("bler_minus", "minus", "tab:green", "D-"))

    n = len(se2s)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2), squeeze=False)
    print("=== 3-way BLER grid ===")
    for j, se2 in enumerate(se2s):
        ax = axes[0][j]
        sub = sorted([r for r in rows if r["sigma_e2"] == se2], key=lambda r: r["ebno_db"])
        ebnos = [r["ebno_db"] for r in sub]
        print(f"\n sigma_e2={se2}  (Eb/N0: BP / legacy / EP)")
        for key, lab, col, st in series:
            if key not in sub[0]:
                continue
            ys = [max(r[key], 8e-5) for r in sub]
            ax.plot(ebnos, ys, st, color=col, ms=4, label=lab)
        for r in sub:
            print(f"   {r['ebno_db']:>4}: {r['bler_bp']:.4f} / {r['bler_legacy']:.4f}"
                  f" / {r['bler_ep']:.4f}")
        ax.set_yscale("log"); ax.set_ylim(7e-5, 1.3)
        ax.set_title(f"σ_e²={se2}"); ax.set_xlabel("Eb/N0 (dB)")
        if j == 0:
            ax.set_ylabel("BLER")
        ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("QPSK fast Rayleigh fading + imperfect CSI — 3-way (method B)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RES, "fading_waterfall.png")
    fig.savefig(out, dpi=120); print("\nsaved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/fading_3way_grid.csv")
