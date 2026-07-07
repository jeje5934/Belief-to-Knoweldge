"""
Build the head-to-head table + waterfall curve from channel_bler.json.

Reference (our best, denoiser-in-the-loop, rate 0.4997, budget-100) is quoted
verbatim from the pure-EP_practical final table (docs/EP_RESEARCH_SUMMARY §F),
mapped to Es/N0 = Eb/N0 + 10log10(0.4997) = Eb/N0 - 3.011 dB.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")

RAW_RATE = 6296 / 12600
ES_SHIFT = 10 * np.log10(RAW_RATE)      # Eb/N0 -> Es/N0, = -3.011 dB

# our-best reference: raw Eb/N0 -> (Es/N0, legacy BLER+CI, EP BLER+CI, BP100 ceiling)
# 0/3200 shown as point 0 with CI upper 0.0012.
REF = {
    0.5: dict(legacy=(0.0009, 0.0003, 0.0028), ep=(0.0106, 0.0076, 0.0148), bp100=0.565),
    0.6: dict(legacy=(0.0, 0.0, 0.0012),        ep=(0.0006, 0.0002, 0.0023), bp100=0.217),
    0.7: dict(legacy=(0.0, 0.0, 0.0012),        ep=(0.0006, 0.0002, 0.0023), bp100=0.055),
}


def load():
    with open(os.path.join(RESULTS, "channel_bler.json")) as f:
        return json.load(f)


def fmt(pt):
    return (f"{pt['bler']:.4f}[{pt['bler_ci'][0]:.4f},{pt['bler_ci'][1]:.4f}]"
            if pt else "-")


def table(d):
    cfgs = d["configs"]
    rates = {k: cfgs[k]["rate"] for k in cfgs}
    lines = []
    lines.append("Same-Es/N0 (same total energy, same N=12600) head-to-head BLER")
    lines.append(f"LDPC infeasible gap: {d.get('ldpc_infeasible_gap','')}")
    lines.append(f"baseline rates: MAX={rates['MAX']:.4f}(ovf {cfgs['MAX']['overflow']}) "
                 f"BG1LEAN={rates['BG1LEAN']:.4f}(ovf {cfgs['BG1LEAN']['overflow']}) "
                 f"BG2LEAN={rates['BG2LEAN']:.4f}(ovf {cfgs['BG2LEAN']['overflow']})  "
                 f"| ours rate=0.4997\n")
    hdr = (f"{'Es/N0':>7} {'rawEb':>5} | {'ours:legacy':>19} {'ours:EP':>19} "
           f"{'BP100ceil':>9} | {'base MAX':>16} {'base BG1LEAN':>16} {'base BG2LEAN':>16}")
    lines.append(hdr)
    for eb in [0.5, 0.6, 0.7]:
        es = eb + ES_SHIFT
        r = REF[eb]
        lg = f"{r['legacy'][0]:.4f}[{r['legacy'][1]:.4f},{r['legacy'][2]:.4f}]"
        ep = f"{r['ep'][0]:.4f}[{r['ep'][1]:.4f},{r['ep'][2]:.4f}]"
        mx = cfgs["MAX"]["points"].get(f"raw{eb}")
        b1 = cfgs["BG1LEAN"]["points"].get(f"raw{eb}")
        b2 = cfgs["BG2LEAN"]["points"].get(f"raw{eb}")
        lines.append(f"{es:>7.3f} {eb:>5} | {lg:>19} {ep:>19} {r['bp100']:>9.3f} | "
                     f"{fmt(mx):>16} {fmt(b1):>16} {fmt(b2):>16}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(RESULTS, "channel_table.txt"), "w") as f:
        f.write(txt + "\n")


def curve(d):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cfg, color, mk in [("MAX", "tab:green", "o"), ("BG1LEAN", "tab:orange", "s"),
                           ("BG2LEAN", "tab:red", "D")]:
        pts = d["configs"][cfg]["points"]
        xs, ys, los, his = [], [], [], []
        for lab, pt in sorted(pts.items(), key=lambda kv: kv[1]["esno_db"]):
            xs.append(pt["esno_db"]); ys.append(max(pt["bler"], 1e-4))
            los.append(max(pt["bler_ci"][0], 1e-5)); his.append(max(pt["bler_ci"][1], 1e-4))
        rate = d["configs"][cfg]["rate"]
        ax.plot(xs, ys, mk + "-", color=color, label=f"baseline {cfg} (r={rate:.3f})")
        ax.fill_between(xs, los, his, color=color, alpha=0.15)
    # our-best reference points
    for eb in [0.5, 0.6, 0.7]:
        es = eb + ES_SHIFT
        ax.plot(es, max(REF[eb]["legacy"][0], 1e-4), "kv",
                label="ours legacy [5]x20" if eb == 0.5 else None)
        ax.plot(es, REF[eb]["ep"][0], "k^",
                label="ours EP [5]x20" if eb == 0.5 else None)
        ax.plot(es, REF[eb]["bp100"], "kx",
                label="BP-100 ceiling (no source)" if eb == 0.5 else None)
    ax.set_yscale("log")
    ax.set_xlabel("Es/N0 (dB)  — same total energy / same N")
    ax.set_ylabel("BLER")
    ax.set_title("Separation baseline vs denoiser-in-the-loop (same Es/N0)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS, "channel_waterfall.png")
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    d = load()
    table(d)
    curve(d)
