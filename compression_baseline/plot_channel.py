"""
Representative figure + 3-tier spectrum table on the shared Es/N0 axis.

The figure shows the fair 0-overflow (MAX-container) waterfall for every
compressor plus the joint denoiser system, so the curves form a clean spectrum
ordered by compression strength:

  [learned source knowledge]  base MAX (PixelCNN, 3.03 bpp)
  [source in the DECODER]     ours legacy / EP (raw, rate 0.5)
  [general image structure]   WebP-MAX (4.53 bpp), PNG-MAX (5.17 bpp)
  [no knowledge]              gzip-MAX (4.77 bpp)

The p99 / BG1LEAN / gzip-P99 variants (which trade an overflow floor for more
parity) are summarised in the table footnote, not drawn, to keep the figure clean.
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.dirname(__file__) + "/results"
GRID = [-2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0]


def load(name):
    return json.load(open(os.path.join(RESULTS, name)))


def at(points, esno):
    for pt in points.values():
        if abs(pt["esno_db"] - esno) < 1e-2:
            return pt
    return None


def knee(getter, thr=0.1):
    xs, ys = [], []
    for es in GRID:
        pt = getter(es)
        if pt: xs.append(es); ys.append(pt["bler"])
    for i in range(len(xs) - 1):
        if ys[i] < thr <= ys[i + 1]:
            f = (thr - ys[i]) / (ys[i + 1] - ys[i])
            return xs[i] + f * (xs[i + 1] - xs[i])
    return None


def main():
    ours = load("our_waterfall.json")
    ch = load("channel_bler.json"); gz = load("gzip_bler.json"); pg = load("png_bler.json")
    nMAX = ch["configs"]["MAX"]
    gMAX = gz["configs"]["gzipMAX"]
    pngMAX = pg["configs"]["pngMAX"]; webpMAX = pg["configs"]["webpMAX"]
    ourf = lambda m: (lambda e: ours.get(f"{m}@{e}"))

    # label, tier, compressor bpp, rate, kbits, getter, color, style
    SYS = [
        ("base MAX",   "learned",  "PixelCNN 3.03", nMAX["rate"], nMAX["k_ldpc"],
         lambda e: at(nMAX["points"], e), "tab:green", "o-"),
        ("ours legacy", "joint",   "raw (decoder)", 0.4997, 6296,
         ourf("legacy_[5]x20"), "tab:blue", "v-"),
        ("WebP MAX",   "image",    "WebP 4.53", webpMAX["rate"], webpMAX["k_ldpc"],
         lambda e: at(webpMAX["points"], e), "tab:orange", "D-"),
        ("ours EP",    "joint",    "raw (decoder)", 0.4997, 6296,
         ourf("EP_[5]x20"), "tab:cyan", "^-"),
        ("gzip MAX",   "none",     "gzip 4.77", gMAX["rate"], gMAX["k_ldpc"],
         lambda e: at(gMAX["points"], e), "tab:red", "P-"),
        ("PNG MAX",    "image",    "PNG 5.17", pngMAX["rate"], pngMAX["k_ldpc"],
         lambda e: at(pngMAX["points"], e), "tab:purple", "X-"),
    ]

    # ---- 3-tier spectrum table ----
    TIER = {"learned": "learned source knowledge (PixelCNN)",
            "joint": "source in DECODER (raw denoiser)",
            "image": "general image structure (PNG/WebP)",
            "none": "no knowledge (gzip)"}
    leg_knee = knee(ourf("legacy_[5]x20"))
    L = ["3-tier compression spectrum — MAX container (0 overflow), shared Es/N0, 3200 cw",
         "knee = Es/N0 @ BLER 0.1 (more negative = deeper = better); "
         "adv = legacy_knee - knee (+ = beats legacy)",
         f"{'tier':<38} {'system':<12}{'rate':>6}{'kbits':>7}{'knee':>7}{'adv/legacy':>11}",
         "-" * 82]
    rows = []
    for label, tier, bpp, rate, kb, g, *_ in SYS:
        k = knee(g)
        adv = "" if label == "ours legacy" else (f"{leg_knee-k:+.2f} dB" if k else "n/a")
        rows.append((k if k is not None else -9, TIER[tier], label, rate, kb, k, adv))
    for _, tier, label, rate, kb, k, adv in sorted(rows):        # deepest knee first
        ks = f"{k:.2f}" if k is not None else "<-4"
        L.append(f"{tier:<38} {label:<12}{rate:>6.3f}{kb:>7}{ks:>7}{adv:>11}")
    L.append("")
    L.append("footnote — overflow-floor variants (traded parity for a %overflow BLER floor):")
    L.append(f"  base BG1LEAN r0.333: 0 LDPC-fail to -4.0, floor {ch['configs']['BG1LEAN']['overflow']/3200:.4f}")
    L.append(f"  gzip-P99 r0.440: LDPC knee ~-3.04, floor {gz['configs']['gzipP99']['overflow']/gz['N']:.4f}")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(RESULTS, "channel_table.txt"), "w").write(txt + "\n")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for label, tier, bpp, rate, kb, g, color, style in SYS:
        xs, ys, lo, hi = [], [], [], []
        for es in GRID:
            pt = g(es)
            if not pt: continue
            c = pt.get("bler_ci") or [pt["ci_lo"], pt["ci_hi"]]
            xs.append(es); ys.append(max(pt["bler"], 8e-5))
            lo.append(max(c[0], 4e-5)); hi.append(max(c[1], 8e-5))
        ax.plot(xs, ys, style, color=color, ms=4,
                label=f"{label} — {bpp}, r={rate:.3f}")
        ax.fill_between(xs, lo, hi, color=color, alpha=0.10)
    ax.set_yscale("log"); ax.invert_xaxis(); ax.set_ylim(7e-5, 1.5)
    ax.set_xlabel("Es/N0 (dB) — same total energy / same N=12600 (MAX container, 0 overflow)")
    ax.set_ylabel("BLER")
    ax.set_title("Coding gain vs compression strength: learned > joint-decoder ≳ image-codec > general")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = os.path.join(RESULTS, "channel_waterfall.png")
    fig.savefig(out, dpi=130); print("\nsaved", out)


if __name__ == "__main__":
    main()
