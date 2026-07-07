"""
Unified waterfall (the paper's representative figure) + head-to-head table on the
shared Es/N0 axis.  Six systems:

  ours legacy [5]x20   (denoiser-in-loop, rate 0.500, 6296 bits)
  ours EP [5]x20       (denoiser-in-loop, rate 0.500, 6296 bits)
  base MAX             (PixelCNN 3.03bpp, rate 0.389, 4896 bits)
  base BG1LEAN         (PixelCNN 3.03bpp, rate 0.333, 4200 bits)
  gzip MAX             (gzip 4.8bpp,      rate 0.485, 6112 bits)
  gzip P99             (gzip 4.8bpp,      rate 0.440, 5545 bits)

Baseline overflow floors (SNR-independent, = overflow/N) drawn dashed.  Eb/N0 per
system annotated in the table so a deeper knee reads as the rate cost.
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.dirname(__file__) + "/results"
GRID = [-2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0]
OURS_RATE = 6296 / 12600


def eb(esno, rate):
    return esno - 10 * math.log10(rate)


def at(points, esno):
    for pt in points.values():
        if abs(pt["esno_db"] - esno) < 1e-2:
            return pt
    return None


def ci(pt):
    return pt.get("bler_ci") or [pt["ci_lo"], pt["ci_hi"]]


def main():
    ours = json.load(open(os.path.join(RESULTS, "our_waterfall.json")))
    ch = json.load(open(os.path.join(RESULTS, "channel_bler.json")))
    gz = json.load(open(os.path.join(RESULTS, "gzip_bler.json")))
    nMAX, nB1 = ch["configs"]["MAX"], ch["configs"]["BG1LEAN"]
    gMAX, gP99 = gz["configs"]["gzipMAX"], gz["configs"]["gzipP99"]

    def our(mode, esno):
        return ours.get(f"{mode}@{esno}")

    SYS = [  # label, rate, kbits, getter
        ("ours legacy [5]x20", OURS_RATE, 6296, lambda e: our("legacy_[5]x20", e)),
        ("ours EP [5]x20",     OURS_RATE, 6296, lambda e: our("EP_[5]x20", e)),
        ("base MAX (PixelCNN)", nMAX["rate"], nMAX["k_ldpc"], lambda e: at(nMAX["points"], e)),
        ("base BG1LEAN (PixelCNN)", nB1["rate"], nB1["k_ldpc"], lambda e: at(nB1["points"], e)),
        ("gzip MAX",  gMAX["rate"], gMAX["k_ldpc"], lambda e: at(gMAX["points"], e)),
        ("gzip P99",  gP99["rate"], gP99["k_ldpc"], lambda e: at(gP99["points"], e)),
    ]

    # ---- table ----
    L = ["Unified waterfall — shared Es/N0 (same energy, N=12600, 3200 cw, Wilson CI)",
         "compressor bpp: PixelCNN 3.03 | gzip 4.8 | ours=raw (source in DECODER)",
         "overflow floors: base BG1LEAN=%.4f  gzip P99=%.4f" %
         (nB1["overflow"] / 3200, gP99["overflow"] / gz["N"]), ""]
    hdr = f"{'Es/N0':>6} |" + "".join(f"{s[0].split(' (')[0][:16]:>17}" for s in SYS)
    L.append(hdr)
    L.append(f"{'rate':>6} |" + "".join(f"{s[1]:>17.3f}" for s in SYS))
    for es in GRID:
        row = f"{es:>6.2f} |"
        for _, _, _, g in SYS:
            pt = g(es)
            row += (f"{pt['bler']:>8.4f}{'':>9}" if pt else f"{'-':>17}")
        L.append(row)
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(RESULTS, "channel_table.txt"), "w").write(txt + "\n")

    # ---- knees (Es/N0 at BLER~0.1, linear-interp in log-BLER) ----
    def knee(getter):
        xs, ys = [], []
        for es in GRID:
            pt = getter(es)
            if pt: xs.append(es); ys.append(pt["bler"])
        xs, ys = np.array(xs), np.array(ys)
        for i in range(len(xs) - 1):
            if ys[i] < 0.1 <= ys[i + 1]:
                # interp on Es/N0 (xs decreasing)
                f = (0.1 - ys[i]) / (ys[i + 1] - ys[i])
                return xs[i] + f * (xs[i + 1] - xs[i])
        return None
    print("\nknees (Es/N0 @ BLER=0.1):")
    for label, rate, kb, g in SYS:
        k = knee(g)
        print(f"  {label:26} rate={rate:.3f} k={kb:5d}  knee={k if k is None else round(k,2)}")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9, 6.2))
    styles = ["v-", "^-", "o-", "s-", "D-", "P-"]
    colors = ["tab:blue", "tab:cyan", "tab:green", "tab:olive", "tab:red", "tab:orange"]
    for (label, rate, kb, g), st, col in zip(SYS, styles, colors):
        xs, ys, lo, hi = [], [], [], []
        for es in GRID:
            pt = g(es)
            if not pt: continue
            c = ci(pt)
            xs.append(es); ys.append(max(pt["bler"], 8e-5))
            lo.append(max(c[0], 4e-5)); hi.append(max(c[1], 8e-5))
        ax.plot(xs, ys, st, color=col, ms=4, label=f"{label}  r={rate:.3f}, {kb}b")
        ax.fill_between(xs, lo, hi, color=col, alpha=0.10)
    ax.axhline(nB1["overflow"] / 3200, ls=":", color="tab:olive", lw=1)
    ax.axhline(gP99["overflow"] / gz["N"], ls=":", color="tab:orange", lw=1)
    ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("Es/N0 (dB) — same total energy / same N=12600")
    ax.set_ylabel("BLER"); ax.set_ylim(7e-5, 1.5)
    ax.set_title("Separation (learned vs commercial compressor) vs joint denoiser")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    out = os.path.join(RESULTS, "channel_waterfall.png")
    fig.savefig(out, dpi=130); print("saved", out)


if __name__ == "__main__":
    main()
