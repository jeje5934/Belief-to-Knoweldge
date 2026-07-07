"""
gzip separation baseline — waterfall, reusing the neural pipeline's run_config.

Containers: gzip-MAX (k=max B_g+24) and gzip-P99 (k=ceil(p99)+24), LDPC-feasibility
adjusted.  Same channel / CRC24A / BP-100 / grid as the neural baseline.  Because
gzip barely compresses (~4.8 bpp), these rates sit near the raw 0.5 -- so the
freed rate (and hence coding gain) is small.  We extend the grid up to -2.0 dB to
capture gzip's knee, which is expected above the neural baseline's.
"""
import json, os
import numpy as np
import tensorflow as tf
for _g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_g, True)

from compression_baseline.channel_experiment import run_config, CRC_LEN, N_CODEWORD

RESULTS = os.path.dirname(__file__) + "/results"
STREAMS = os.path.join(RESULTS, "gzip_streams.npz")
OUT = os.path.join(RESULTS, "gzip_bler.json")


def feasible_kldpc(k):
    """Nudge k into a supported 5G-LDPC region at n=12600 (gap 3825..4199)."""
    if 3825 <= k <= 4199:
        return 4200
    return k


def main(batch=320):
    d = np.load(STREAMS)
    bits, lengths, imgs = d["bits"], d["lengths"], d["imgs"]
    N = bits.shape[0]
    print(f"loaded {N} gzip streams  B_g mean={lengths.mean():.1f} "
          f"p99={np.percentile(lengths,99):.1f} max={lengths.max()}")

    grid = [-2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0]
    esno_points = [{"label": f"es{e}", "no": float(10.0 ** (-e / 10.0)),
                    "esno_db": float(e), "ebno_raw": None} for e in grid]

    K_max = int(lengths.max())
    K_p99 = int(np.ceil(np.percentile(lengths, 99)))
    configs = []
    for label, K in [("gzipMAX", K_max), ("gzipP99", K_p99)]:
        k_adj = feasible_kldpc(K + CRC_LEN)
        K_use = k_adj - CRC_LEN
        if K_use != K:
            print(f"  {label}: K {K}->{K_use} (LDPC-gap adjust) k={k_adj} "
                  f"rate={k_adj/N_CODEWORD:.4f}")
        configs.append((label, K_use))

    out = {"N": N, "Bg_mean": float(lengths.mean()),
           "Bg_p99": float(np.percentile(lengths, 99)), "Bg_max": int(lengths.max()),
           "raw_rate": 6296 / N_CODEWORD, "configs": {}}
    for label, K in configs:
        out["configs"][label] = run_config(bits, lengths, imgs, K, esno_points,
                                           batch, True, label)
    json.dump(out, open(OUT, "w"), indent=2)
    print("wrote gzip_bler.json")


if __name__ == "__main__":
    main()
