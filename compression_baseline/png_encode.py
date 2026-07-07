"""
Image-specific lossless-codec separation baselines: PNG and WebP-lossless.

gzip is a general-purpose codec blind to image structure, so it is not a fair
representative of "commercial lossless".  PNG (zlib + per-row predictive filters)
and WebP-lossless (spatial prediction + entropy) DO know image structure — the
right middle tier between the learned PixelCNN (source knowledge) and gzip (none).

JPEG-LS (pyjpegls/imagecodecs) is unavailable here: pyjpegls requires numpy>=2.0,
which is incompatible with this env's sionna/tensorflow (numpy<2.0) — installing
it breaks the stack, so we use WebP-lossless as the second image codec instead.

Outputs per codec: distribution on test[:2500] and (for waterfall) streams for
test[:3200].  PNG is reported two ways: (a) full file bytes (used for container
design, conservative) and (b) IDAT payload only (fixed chunk/header overhead
removed) — the ~28×28 tile makes the ~57-byte fixed overhead non-negligible.
"""
import io
import json
import os
import struct

import numpy as np
from PIL import Image

from compression_baseline import data as D

RESULTS = os.path.dirname(__file__) + "/results"


def png_bytes(px2d):
    b = io.BytesIO()
    Image.fromarray(px2d, mode="L").save(b, format="PNG", optimize=True)
    return b.getvalue()


def png_idat_payload_bytes(png):
    """Sum of IDAT chunk data lengths (the zlib stream) — excludes signature,
    IHDR, IEND and per-chunk 12-byte overhead."""
    pos = 8  # skip signature
    total = 0
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos:pos + 4])
        ctype = png[pos + 4:pos + 8]
        if ctype == b"IDAT":
            total += length
        pos += 12 + length
        if ctype == b"IEND":
            break
    return total


def webp_bytes(px2d):
    b = io.BytesIO()
    Image.fromarray(px2d, mode="L").save(b, format="WEBP", lossless=True,
                                         quality=100, method=6)
    return b.getvalue()


def roundtrip_ok(px2d, fmt):
    b = io.BytesIO()
    Image.fromarray(px2d, mode="L").save(
        b, format=fmt, **({"optimize": True} if fmt == "PNG"
                          else {"lossless": True, "quality": 100, "method": 6}))
    b.seek(0)
    back = np.array(Image.open(b).convert("L"))
    return np.array_equal(back, px2d)


def dist(bits):
    b = np.asarray(bits, np.int64)
    return {"mean": float(b.mean()), "median": float(np.median(b)),
            "p10": float(np.percentile(b, 10)), "p90": float(np.percentile(b, 90)),
            "p99": float(np.percentile(b, 99)), "max": int(b.max()),
            "min": int(b.min()), "bpp": float(b.mean() / 784.0)}


def main():
    px = D.load_test().reshape(-1, D.NPIX).numpy().astype(np.uint8).reshape(-1, 28, 28)
    N_dist, N_ch = 2500, 3200

    png_file = np.array([len(png_bytes(px[i])) * 8 for i in range(N_dist)])
    png_idat = np.array([png_idat_payload_bytes(png_bytes(px[i])) * 8 for i in range(N_dist)])
    webp_file = np.array([len(webp_bytes(px[i])) * 8 for i in range(N_dist)])

    stats = {"png_file": dist(png_file), "png_idat": dist(png_idat),
             "webp_file": dist(webp_file),
             "ref_bpp": {"PixelCNN": 3.03, "gzip": 4.77}}
    json.dump(stats, open(os.path.join(RESULTS, "png_stats.json"), "w"), indent=2)

    print("== lossless bpp (test[:2500]) ==")
    print(f"  PixelCNN (learned)     3.03 bpp   (ref)")
    print(f"  PNG file (a)          {stats['png_file']['bpp']:.2f} bpp   "
          f"mean {stats['png_file']['mean']:.0f} b  max {stats['png_file']['max']}")
    print(f"  PNG IDAT-only (b)     {stats['png_idat']['bpp']:.2f} bpp   "
          f"mean {stats['png_idat']['mean']:.0f} b  (overhead removed)")
    print(f"  WebP-lossless (a)     {stats['webp_file']['bpp']:.2f} bpp   "
          f"mean {stats['webp_file']['mean']:.0f} b  max {stats['webp_file']['max']}")
    print(f"  gzip (general)         4.77 bpp   (ref)")

    ok_png = sum(roundtrip_ok(px[i], "PNG") for i in range(20))
    ok_webp = sum(roundtrip_ok(px[i], "WEBP") for i in range(20))
    print(f"  roundtrip bit-exact: PNG {ok_png}/20  WebP {ok_webp}/20")

    # ---- streams for channel (full-file bits, conservative container) ----
    for tag, fn in [("png", png_bytes), ("webp", webp_bytes)]:
        streams, lengths = [], []
        for i in range(N_ch):
            comp = fn(px[i])
            bits = np.unpackbits(np.frombuffer(comp, np.uint8))
            streams.append(bits); lengths.append(bits.shape[0])
        lengths = np.array(lengths, np.int32)
        maxlen = int(lengths.max())
        arr = np.zeros((N_ch, maxlen), np.uint8)
        for i, s in enumerate(streams):
            arr[i, : s.shape[0]] = s
        np.savez_compressed(os.path.join(RESULTS, f"{tag}_streams.npz"),
                            bits=arr, lengths=lengths,
                            imgs=px[:N_ch].reshape(N_ch, D.NPIX).astype(np.uint8))
        print(f"  saved {tag}_streams.npz  max B={lengths.max()} "
              f"(k_MAX={lengths.max()+24}, rate={(lengths.max()+24)/12600:.4f})")


if __name__ == "__main__":
    main()
