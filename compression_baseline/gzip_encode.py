"""
Commercial-codec (gzip) separation baseline — encoder + B_g distribution.

Same separation pipeline as the neural baseline but the compressor is a general
codec with NO source knowledge: gzip (DEFLATE, level 9), actual bytes incl. the
gzip header/trailer.  This isolates the value of the *learned* source model: any
coding-gain difference vs the PixelCNN baseline is the compression the source
knowledge bought.

Outputs:
  results/gzip_stats.json        B_g distribution on test[:2500] (Phase-1 sample)
  results/gzip_streams.npz       bits/lengths/imgs for test[:3200] (channel set)
"""
import gzip
import json
import os

import numpy as np

from compression_baseline import data as D

RESULTS = os.path.dirname(__file__) + "/results"


def gzip_bits(img_pixels_u8):
    """gzip-compress one 784-byte image; return (bit array uint8, n_bits)."""
    raw = img_pixels_u8.astype(np.uint8).tobytes()
    comp = gzip.compress(raw, compresslevel=9)         # header+trailer included
    bits = np.unpackbits(np.frombuffer(comp, dtype=np.uint8))
    return bits, bits.shape[0], comp


def main():
    test = D.load_test()                                # [N,1,28,28] uint8
    px = test.reshape(test.size(0), D.NPIX).numpy().astype(np.uint8)

    # ---- [1] distribution on the Phase-1 sample (2500) ----
    N_dist = 2500
    Bg = np.array([gzip_bits(px[i])[1] for i in range(N_dist)], dtype=np.int64)
    stats = {
        "n": N_dist, "Bg_mean": float(Bg.mean()), "Bg_median": float(np.median(Bg)),
        "Bg_p10": float(np.percentile(Bg, 10)), "Bg_p90": float(np.percentile(Bg, 90)),
        "Bg_p99": float(np.percentile(Bg, 99)), "Bg_max": int(Bg.max()),
        "Bg_min": int(Bg.min()), "bpp_mean": float(Bg.mean() / 784.0),
        "phase1_ref_mean_bits": 3740.6,
    }
    json.dump(stats, open(os.path.join(RESULTS, "gzip_stats.json"), "w"), indent=2)
    print("== gzip B_g distribution (test[:2500], bits/image) ==")
    for k in ["Bg_mean", "Bg_median", "Bg_p10", "Bg_p90", "Bg_p99", "Bg_max",
              "Bg_min", "bpp_mean"]:
        print(f"  {k:10s} {stats[k]:.1f}")
    print(f"  Phase-1 gzip ref mean = 3740.6 bits  (match: "
          f"{abs(stats['Bg_mean']-3740.6) < 5})")

    # ---- roundtrip bit-exact (formal, 20 images) ----
    ok = 0
    for i in range(20):
        _, _, comp = gzip_bits(px[i])
        back = np.frombuffer(gzip.decompress(comp), dtype=np.uint8)
        ok += int(np.array_equal(back, px[i]))
    print(f"  roundtrip bit-exact: {ok}/20")

    # ---- [channel set] encode 3200, store padded bits ----
    N_ch = 3200
    streams, lengths = [], []
    for i in range(N_ch):
        bits, n, _ = gzip_bits(px[i])
        streams.append(bits); lengths.append(n)
    lengths = np.array(lengths, dtype=np.int32)
    maxlen = int(lengths.max())
    bits_arr = np.zeros((N_ch, maxlen), dtype=np.uint8)
    for i, s in enumerate(streams):
        bits_arr[i, : s.shape[0]] = s
    np.savez_compressed(os.path.join(RESULTS, "gzip_streams.npz"),
                        bits=bits_arr, lengths=lengths,
                        imgs=px[:N_ch].astype(np.uint8))
    print(f"\nsaved gzip_streams.npz  N={N_ch}  B_g mean={lengths.mean():.1f} "
          f"p99={np.percentile(lengths,99):.1f} max={lengths.max()}")


if __name__ == "__main__":
    main()
