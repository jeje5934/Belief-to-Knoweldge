"""
Four-curve waterfall + head-to-head table on the shared Es/N0 axis.

Curves: ours legacy [5]x20, ours EP [5]x20 (from our_waterfall.json), baseline
MAX and BG1LEAN (from channel_bler.json).  Baseline overflow floors drawn as
dashed lines (the SNR-independent structural BLER = overflow/N).

Per task 3(a): the table annotates each system's info-bit count k and its Eb/N0
at the shared Es/N0, so a deeper baseline knee is read as the *rate cost* (fewer
info bits carried for the same energy), not a free lunch.
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.dirname(__file__) + "/results"
GRID = [-2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0]
OURS_K = 6296          # 6272 payload + 24 CRC
OURS_RATE = OURS_K / 12600


def eb(esno, rate):
    return esno - 10 * math.log10(rate)


def base_at(cfg, esno):
    """find a baseline point matching esno_db within 1e-2."""
    for pt in cfg["points"].values():
        if abs(pt["esno_db"] - esno) < 1e-2:
            return pt
    return None


def main():
    ours = json.load(open(os.path.join(RESULTS, "our_waterfall.json")))
    ch = json.load(open(os.path.join(RESULTS, "channel_bler.json")))
    MAX, B1 = ch["configs"]["MAX"], ch["configs"]["BG1LEAN"]

    def our(mode, esno):
        return ours.get(f"{mode}@{esno}")

    # ---- table ----
    L = []
    L.append("Four-way waterfall on shared Es/N0 (same total energy, N=12600, 3200 cw, Wilson CI)")
    L.append(f"info bits k:  ours=6296 (rate {OURS_RATE:.4f})  |  base MAX k={MAX['k_ldpc']} "
             f"(rate {MAX['rate']:.4f})  base BG1LEAN k={B1['k_ldpc']} (rate {B1['rate']:.4f})")
    L.append(f"baseline overflow floors: MAX={MAX['overflow']}/3200={MAX['overflow']/3200:.4f}  "
             f"BG1LEAN={B1['overflow']}/3200={B1['overflow']/3200:.4f}")
    L.append("")
    L.append(f"{'Es/N0':>6} | {'ours legacy':>20} {'ours EP':>20} | "
             f"{'base MAX':>20} {'base BG1LEAN':>20} | ebN0: ours/MAX/BG1")
    def f(pt, key="bler"):
        if not pt: return "-"
        lo, hi = pt.get("bler_ci") or [pt["ci_lo"], pt["ci_hi"]]
        return f"{pt[key]:.4f}[{lo:.4f},{hi:.4f}]"
    for es in GRID:
        lg, ep = our("legacy_[5]x20", es), our("EP_[5]x20", es)
        mx, b1 = base_at(MAX, es), base_at(B1, es)
        ebs = f"{eb(es,OURS_RATE):.2f}/{eb(es,MAX['rate']):.2f}/{eb(es,B1['rate']):.2f}"
        L.append(f"{es:>6.2f} | {f(lg):>20} {f(ep):>20} | {f(mx):>20} {f(b1):>20} | {ebs}")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(RESULTS, "channel_table.txt"), "w").write(txt + "\n")

    # ---- curve ----
    fig, ax = plt.subplots(figsize=(8.5, 6))
    def series(getter, key):
        xs, ys, lo, hi = [], [], [], []
        for es in GRID:
            pt = getter(es)
            if not pt: continue
            b = pt[key]; c = pt.get("bler_ci") or [pt["ci_lo"], pt["ci_hi"]]
            xs.append(es); ys.append(max(b, 8e-5))
            lo.append(max(c[0], 4e-5)); hi.append(max(c[1], 8e-5))
        return xs, ys, lo, hi
    for label, getter, color, mk in [
        ("ours legacy [5]x20 (r0.500)", lambda e: our("legacy_[5]x20", e), "tab:blue", "v"),
        ("ours EP [5]x20 (r0.500)",     lambda e: our("EP_[5]x20", e),     "tab:cyan", "^"),
        ("base MAX (r0.389)",           lambda e: base_at(MAX, e),         "tab:green", "o"),
        ("base BG1LEAN (r0.333)",       lambda e: base_at(B1, e),          "tab:orange", "s"),
    ]:
        xs, ys, lo, hi = series(getter, "bler")
        ax.plot(xs, ys, mk + "-", color=color, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12)
    # overflow floors (structural, SNR-independent)
    ax.axhline(max(MAX["overflow"] / 3200, 8e-5), ls=":", color="tab:green", lw=1,
               label="MAX overflow floor (0)")
    ax.axhline(B1["overflow"] / 3200, ls=":", color="tab:orange", lw=1,
               label="BG1LEAN overflow floor")
    ax.set_yscale("log"); ax.set_xlabel("Es/N0 (dB) — same total energy / same N=12600")
    ax.set_ylabel("BLER"); ax.invert_xaxis()
    ax.set_title("Separation baseline vs denoiser-in-the-loop — full waterfall")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = os.path.join(RESULTS, "channel_waterfall.png")
    fig.savefig(out, dpi=120); print("saved", out)


if __name__ == "__main__":
    main()
