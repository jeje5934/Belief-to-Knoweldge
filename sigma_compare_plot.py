"""
practical_sigma [C] — analysis + figures from sigma_compare.json.

Produces: BLER-by-strategy bar (Wilson CI), per-chunk σ-trajectory overlay,
per-round BER-trajectory overlay, and a quantified σ-trajectory similarity
between the closed-loop adaptive and the open-loop annealing schedules.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results"


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def traj_arr(d):
    return np.array([d[str(k)] if str(k) in d else d[k]
                     for k in sorted(map(int, d.keys()))])


def main(ebno=0.5):
    data = {**load("sigma_compare_screen.json"), **load("sigma_compare.json")}
    rows = {v["label"]: v for k, v in data.items() if abs(v["ebno"] - ebno) < 1e-6}
    order = sorted(rows, key=lambda l: rows[l]["bler"])

    print(f"=== BLER @ {ebno} dB (best→worst) ===")
    for l in order:
        r = rows[l]
        print(f"  {l:20} BLER={r['bler']:.4f} [{r['ci'][0]:.4f},{r['ci'][1]:.4f}] "
              f"({r['nack']}/{r['total']})")

    # --- σ-trajectory similarity: adaptive (closed-loop) vs annealing (open-loop) ---
    if "lut2stage" in rows:
        a = traj_arr(rows["lut2stage"]["sigma_traj"])
        print("\n=== σ-trajectory similarity vs lut2stage (closed-loop) ===")
        for l in rows:
            if l == "lut2stage":
                continue
            b = traj_arr(rows[l]["sigma_traj"])
            n = min(len(a), len(b))
            corr = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 1 else float("nan")
            l2 = float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))
            print(f"  {l:20} corr={corr:+.3f}  RMSΔσ={l2:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    # (1) BLER bar with CI
    xs = np.arange(len(order))
    bler = [rows[l]["bler"] for l in order]
    lo = [rows[l]["bler"] - rows[l]["ci"][0] for l in order]
    hi = [rows[l]["ci"][1] - rows[l]["bler"] for l in order]
    ax[0].bar(xs, bler, yerr=[lo, hi], capsize=3, color="tab:blue", alpha=.7)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(order, rotation=45, ha="right", fontsize=7)
    ax[0].set_ylabel("BLER"); ax[0].set_title(f"BLER @ {ebno}dB (Wilson CI)"); ax[0].grid(alpha=.3, axis="y")
    # (2) σ trajectories
    for l in order:
        t = traj_arr(rows[l]["sigma_traj"])
        ax[1].plot(range(len(t)), t, marker=".", ms=3, label=l)
    ax[1].set_xlabel("BP chunk"); ax[1].set_ylabel("mean σ"); ax[1].set_title("σ trajectory")
    ax[1].legend(fontsize=6); ax[1].grid(alpha=.3)
    # (3) BER trajectories
    for l in order:
        if rows[l].get("ber_traj"):
            t = traj_arr(rows[l]["ber_traj"])
            ax[2].semilogy(range(len(t)), np.maximum(t, 1e-5), marker=".", ms=3, label=l)
    ax[2].set_xlabel("BP chunk"); ax[2].set_ylabel("payload BER"); ax[2].set_title("per-round BER trajectory")
    ax[2].legend(fontsize=6); ax[2].grid(alpha=.3, which="both")
    fig.tight_layout()
    out = os.path.join(RES, "sigma_compare.png"); fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    import sys
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)
