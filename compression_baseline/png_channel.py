"""
PNG / WebP-lossless separation waterfalls (MAX containers), same pipeline/grid as
gzip.  Rates differ from gzip-MAX by >=0.02, so the curves are measured (per the
task's threshold); they are expected to cluster near gzip-MAX since all three
codecs sit at ~4.5-5.2 bpp.
"""
import json, os
import numpy as np
import tensorflow as tf
for _g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_g, True)
from compression_baseline.channel_experiment import run_config, CRC_LEN, N_CODEWORD

RESULTS = os.path.dirname(__file__) + "/results"


def main(batch=320):
    grid = [-2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0]
    esno_points = [{"label": f"es{e}", "no": float(10.0 ** (-e / 10.0)),
                    "esno_db": float(e), "ebno_raw": None} for e in grid]
    out = {"configs": {}}
    for tag, cfg_label in [("png", "pngMAX"), ("webp", "webpMAX")]:
        d = np.load(os.path.join(RESULTS, f"{tag}_streams.npz"))
        bits, lengths, imgs = d["bits"], d["lengths"], d["imgs"]
        K = int(lengths.max())
        print(f"\n{cfg_label}: N={bits.shape[0]} K={K} k={K+CRC_LEN} "
              f"rate={(K+CRC_LEN)/N_CODEWORD:.4f}", flush=True)
        out["configs"][cfg_label] = run_config(bits, lengths, imgs, K,
                                               esno_points, batch, True, cfg_label)
    json.dump(out, open(os.path.join(RESULTS, "png_bler.json"), "w"), indent=2)
    print("wrote png_bler.json")


if __name__ == "__main__":
    main()
