"""
Same-resource rate diagnostic for the "compress-then-transmit" baseline.

The raw transmission system sends k_raw = 6272 payload + 24 CRC = 6296 info bits
through an N = 12600 channel-bit LDPC codeword (rate ~0.4997).  The separation
baseline instead sends the *compressed* payload (B_c bits) + 24 CRC through the
same N = 12600, at rate R_c = (B_c + 24)/12600.  Lower R_c means more parity for
the same channel resource — the classical source/channel-separation gain.

Sionna's LDPC5GEncoder refuses any rate below 1/5 = 0.2 (measured, see report).
This script converts the measured B_c distribution into R_c figures and judges
feasibility against that 0.2 floor, for both variable-length strategies.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
N = 12600
CRC = 24
RAW_K = 6296
LDPC_MIN_RATE = 0.2                      # Sionna 5G LDPC hard floor (measured)
K_MIN = LDPC_MIN_RATE * N               # 2520 info bits -> smallest supported k


def rate(k):
    return k / N


def main():
    Bc = np.load(os.path.join(RESULTS, "bc_lengths.npy"))
    with open(os.path.join(RESULTS, "bc_stats.json")) as f:
        st = json.load(f)

    kc = Bc + CRC                        # info bits fed to LDPC per image
    Rc = kc / N

    def pct(a, q):
        return float(np.percentile(a, q))

    print("==== Same-resource rate diagnostic ====")
    print(f"raw system:  k={RAW_K}  N={N}  rate={rate(RAW_K):.4f}")
    print(f"LDPC floor:  rate>=1/5=0.2  => k>=K_min={K_MIN:.0f} info bits "
          f"(<=> B_c>={K_MIN-CRC:.0f} bits, {(K_MIN-CRC)/784:.3f} bpp)\n")

    print(f"B_c (bits/image) over {len(Bc)} imgs:")
    print(f"  mean={Bc.mean():.1f}  median={np.median(Bc):.1f}  "
          f"p10={pct(Bc,10):.1f}  p90={pct(Bc,90):.1f}  "
          f"p99={pct(Bc,99):.1f}  max={Bc.max()}")
    print(f"  -> R_c: mean={Rc.mean():.4f}  median={np.median(Rc):.4f}  "
          f"p90={pct(Rc,90):.4f}  p99={pct(Rc,99):.4f}  max={Rc.max():.4f}\n")

    frac_below = float((Rc < LDPC_MIN_RATE).mean())
    print(f"fraction of images whose per-image R_c < 0.2 floor: {frac_below:.3f}")

    # --- strategy (i): fixed container sized to a high percentile ---
    print("\n-- Strategy (i): fixed container (pad to a percentile of B_c) --")
    for label, q in [("p90", 90), ("p99", 99), ("max", 100)]:
        cont = int(np.percentile(Bc, q)) if q < 100 else int(Bc.max())
        k_fixed = cont + CRC
        r = rate(k_fixed)
        overflow = float((Bc > cont).mean()) if q < 100 else 0.0
        supported = r >= LDPC_MIN_RATE
        pad_waste = k_fixed - (Bc.mean() + CRC)
        print(f"  container={label:4s} B_c*={cont:5d}  k_fixed={k_fixed:5d}  "
              f"R_c={r:.4f}  {'SUPPORTED' if supported else 'BELOW FLOOR'}  "
              f"overflow_frac={overflow:.3f}  mean_pad={pad_waste:.0f} bits")

    # --- strategy (ii): per-image adaptive rate ---
    print("\n-- Strategy (ii): per-image adaptive rate --")
    print(f"  needs LDPC at each image's R_c; {frac_below*100:.1f}% of images fall "
          f"below the 0.2 floor and cannot be encoded without repetition.")

    # smallest container that clears the floor
    k_floor = int(np.ceil(K_MIN))
    print(f"\n-- Floor-clearing container --")
    print(f"  set k_fixed = K_min = {k_floor} (rate exactly 0.2): container holds "
          f"B_c <= {k_floor-CRC} bits ({(k_floor-CRC)/784:.3f} bpp).")
    fit = float((Bc <= (k_floor - CRC)).mean())
    print(f"  fraction of images that fit under a rate-0.2 container: {fit:.3f} "
          f"(rest overflow and need a larger/again-lower-rate container).")

    out = {
        "raw_rate": rate(RAW_K), "ldpc_min_rate": LDPC_MIN_RATE, "K_min": K_MIN,
        "Rc_mean": float(Rc.mean()), "Rc_median": float(np.median(Rc)),
        "Rc_p90": pct(Rc, 90), "Rc_p99": pct(Rc, 99), "Rc_max": float(Rc.max()),
        "frac_perimage_below_floor": frac_below,
        "container_p99_Bc": float(np.percentile(Bc, 99)),
        "container_max_Bc": int(Bc.max()),
        "container_max_rate": rate(int(Bc.max()) + CRC),
        "frac_fit_rate02_container": fit,
    }
    with open(os.path.join(RESULTS, "rate_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote rate_analysis.json")


if __name__ == "__main__":
    main()
