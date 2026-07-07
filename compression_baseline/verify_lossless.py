"""
Rigorous lossless audit of every compressor used in the study.

For each codec we check bit-exact round trip on a large sample, INCLUDING
decoding from the exact bytes that were stored/transmitted (the *_streams.npz
payloads), not just a fresh encode.  Any single mismatch is reported.

  neural  PixelCNN+AC : canonical cuDNN-off codec, encode->decode on N_NEURAL imgs
  gzip / PNG / WebP   : full test[:N_CODEC] round trip + decode-from-stored-bits
"""
import gzip as gz
import io
import os

import numpy as np
import torch
torch.backends.cudnn.enabled = False        # canonical deterministic codec
from PIL import Image

from compression_baseline import data as D
from compression_baseline.compress import load_model, encode_batch, decode_batch

RESULTS = os.path.dirname(__file__) + "/results"


def audit_neural(n=1024, batch=256):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(dev)
    imgs = D.load_test()[:n]
    bad = 0
    for i in range(0, n, batch):
        chunk = imgs[i:i + batch]
        streams, _ = encode_batch(model, chunk, dev)
        rec = decode_batch(model, streams, dev)
        bad += int((rec != chunk).any(dim=(1, 2, 3)).sum())
    print(f"[neural PixelCNN+AC] {n} imgs  mismatches={bad}  "
          f"-> {'LOSSLESS' if bad == 0 else 'LOSSY!'}")
    return bad


def _webp_bytes(px2d):
    b = io.BytesIO()
    Image.fromarray(px2d, "L").save(b, "WEBP", lossless=True, quality=100, method=6)
    return b.getvalue()


def _png_bytes(px2d):
    b = io.BytesIO()
    Image.fromarray(px2d, "L").save(b, "PNG", optimize=True)
    return b.getvalue()


def audit_codec(name, enc, dec, n):
    px = D.load_test().reshape(-1, D.NPIX).numpy().astype(np.uint8).reshape(-1, 28, 28)[:n]
    bad = 0
    for i in range(n):
        back = dec(enc(px[i]))
        if back.shape != px[i].shape or not np.array_equal(back, px[i]):
            bad += 1
    print(f"[{name}] {n} imgs fresh round trip  mismatches={bad}  "
          f"-> {'LOSSLESS' if bad == 0 else 'LOSSY!'}")
    return bad


def audit_stored(name, npz, dec_from_bits):
    """Decode from the EXACT stored/transmitted bit payloads."""
    path = os.path.join(RESULTS, npz)
    if not os.path.exists(path):
        print(f"[{name} stored] {npz} not found (skip)")
        return 0
    d = np.load(path)
    bits, lengths, imgs = d["bits"], d["lengths"], d["imgs"]
    n = bits.shape[0]
    px = imgs.reshape(n, 28, 28)
    bad = 0
    for i in range(n):
        payload = np.packbits(bits[i, : lengths[i]]).tobytes()
        back = dec_from_bits(payload)
        if back is None or back.shape != px[i].shape or not np.array_equal(back, px[i]):
            bad += 1
    print(f"[{name} stored bits] {n} payloads  mismatches={bad}  "
          f"-> {'LOSSLESS' if bad == 0 else 'LOSSY!'}")
    return bad


def _dec_img(fmt_bytes):
    return np.array(Image.open(io.BytesIO(fmt_bytes)).convert("L"))


def main():
    total = 0
    total += audit_neural(n=1024)
    total += audit_codec("gzip", lambda p: gz.compress(p.tobytes(), 9),
                         lambda c: np.frombuffer(gz.decompress(c), np.uint8).reshape(28, 28),
                         n=2500)
    total += audit_codec("PNG", _png_bytes, _dec_img, n=2500)
    total += audit_codec("WebP-lossless", _webp_bytes, _dec_img, n=2500)
    # stored / transmitted payloads
    total += audit_stored("gzip", "gzip_streams.npz",
                          lambda b: np.frombuffer(gz.decompress(b), np.uint8).reshape(28, 28))
    total += audit_stored("PNG", "png_streams.npz", _dec_img)
    total += audit_stored("WebP", "webp_streams.npz", _dec_img)
    print("\n=== OVERALL:", "ALL LOSSLESS" if total == 0 else f"{total} MISMATCHES", "===")


if __name__ == "__main__":
    main()
