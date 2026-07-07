"""
Lossless neural compression: encode / decode / round-trip verification and the
per-image compressed-bit distribution B_c.

Encoding uses one teacher-forced forward pass (all 784 conditionals at once) and
then arithmetic-codes the raster-scan symbols — this is exact because the model
is causal, so the conditional at pixel i depends only on pixels < i.

Decoding is sequential: 784 model calls, filling the canvas pixel by pixel.  We
batch many images through each of the 784 steps so a whole test batch decodes in
784 forward passes total (slow per the spec, but correct and tractable).

Bit-exactness rests on encoder and decoder feeding the *same* integer CDF into
the coder.  We assert it by round-tripping >=100 images and comparing to the
originals byte for byte.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from compression_baseline import data as D
from compression_baseline.pixelcnn import GatedPixelCNN
from compression_baseline.arithmetic_coder import (
    ArithmeticEncoder, ArithmeticDecoder, quantize_pmf)

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
CKPT = os.path.join(RESULTS, "pixelcnn_fmnist.pt")
H, W = D.IMG_H, D.IMG_W


def load_model(device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    m = GatedPixelCNN(cfg["n_channels"], cfg["n_layers"], cfg["k"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck.get("test_bpp", None)


@torch.no_grad()
def _step_pmf(model, canvas, r, c):
    """Conditional pmf at pixel (r,c) given the current canvas: float64 [B,256].

    Encoder and decoder both call *this* on a canvas whose pixels < (r,c) are the
    true values and whose pixels >= (r,c) are still zero.  Because they run the
    identical forward on identical tensors, the resulting quantized CDF is
    bit-identical on both sides — this is what makes the round trip provably
    lossless regardless of GPU/cuDNN float quirks.  (A single teacher-forced pass
    is faster but its pmf can differ from the sequential pmf by ~1e-7, enough to
    flip a 16-bit quantization boundary and desync the coder — so we do NOT use
    it for coding.)
    """
    logits = model(D.to_model_input(canvas))            # [B,256,H,W]
    return F.softmax(logits[:, :, r, c].float(), dim=1).double().cpu().numpy()


@torch.no_grad()
def encode_batch(model, imgs_u8, device):
    """Sequentially arithmetic-encode a batch (identical forward path as decode).

    Returns (streams, lengths).  Uses the *true* pixels to fill the canvas, so the
    forward inputs are exactly those the decoder reconstructs.
    """
    B = imgs_u8.size(0)
    imgs = imgs_u8.to(device)
    encoders = [ArithmeticEncoder() for _ in range(B)]
    canvas = torch.zeros(B, 1, H, W, dtype=torch.uint8, device=device)
    for r in range(H):
        for c in range(W):
            pmf = _step_pmf(model, canvas, r, c)
            for b in range(B):
                freq = quantize_pmf(pmf[b])
                sym = int(imgs[b, 0, r, c])
                encoders[b].encode_symbol(sym, freq)
                canvas[b, 0, r, c] = sym                 # true pixel
    streams = [e.finish() for e in encoders]
    lengths = [int(s.shape[0]) for s in streams]
    return streams, lengths


@torch.no_grad()
def decode_batch(model, streams, device):
    """Sequentially decode a batch of bitstreams back to uint8 images."""
    B = len(streams)
    decoders = [ArithmeticDecoder(s) for s in streams]
    canvas = torch.zeros(B, 1, H, W, dtype=torch.uint8, device=device)
    for r in range(H):
        for c in range(W):
            pmf = _step_pmf(model, canvas, r, c)
            for b in range(B):
                freq = quantize_pmf(pmf[b])
                sym = decoders[b].decode_symbol(freq)
                canvas[b, 0, r, c] = sym                 # decoded pixel
    return canvas.cpu()


def roundtrip_verify(model, imgs_u8, device, chunk=64):
    """Encode then decode every image; return (all_ok, n, first_bad_index)."""
    n_ok = 0
    for i in range(0, imgs_u8.size(0), chunk):
        batch = imgs_u8[i:i + chunk]
        streams, _ = encode_batch(model, batch, device)
        recon = decode_batch(model, streams, device)
        eq = torch.equal(recon, batch)
        if not eq:
            # locate first mismatching image for diagnostics
            for b in range(batch.size(0)):
                if not torch.equal(recon[b], batch[b]):
                    return False, imgs_u8.size(0), i + b
        n_ok += batch.size(0)
        print(f"  roundtrip {n_ok}/{imgs_u8.size(0)} bit-exact so far", flush=True)
    return True, n_ok, -1


def measure_Bc(model, imgs_u8, device, batch=128):
    """Compressed bits per image over the whole set (encode only, fast)."""
    lengths = []
    for i in range(0, imgs_u8.size(0), batch):
        _, L = encode_batch(model, imgs_u8[i:i + batch], device)
        lengths.extend(L)
        if (i // batch) % 4 == 0:
            print(f"  measured {len(lengths)}/{imgs_u8.size(0)}", flush=True)
    return np.array(lengths, dtype=np.int64)


def gzip_reference(imgs_u8):
    """Context-only general-codec baseline: gzip of the raw 784-byte payloads."""
    import gzip
    sizes = []
    for b in range(imgs_u8.size(0)):
        raw = imgs_u8[b].reshape(-1).cpu().numpy().astype(np.uint8).tobytes()
        sizes.append(len(gzip.compress(raw, 9)) * 8)   # bits
    return np.array(sizes, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_roundtrip", type=int, default=128)
    ap.add_argument("--n_measure", type=int, default=2000)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_per_process_memory_fraction(0.5, 0)
    model, test_bpp = load_model(device)
    print(f"loaded model (train-reported test_bpp={test_bpp})  device={device}")

    test_u8 = D.load_test()

    print(f"\n[roundtrip] verifying {args.n_roundtrip} test images bit-exact ...")
    ok, n, bad = roundtrip_verify(model, test_u8[:args.n_roundtrip], device)
    print(f"[roundtrip] all_bit_exact={ok}  n={n}  first_bad={bad}")

    print(f"\n[B_c] measuring compressed bits over {args.n_measure} test images ...")
    Bc = measure_Bc(model, test_u8[:args.n_measure], device)
    gz = gzip_reference(test_u8[:args.n_measure])

    stats = {
        "n_roundtrip": int(n), "roundtrip_bit_exact": bool(ok),
        "n_measure": int(Bc.shape[0]),
        "Bc_mean": float(Bc.mean()), "Bc_median": float(np.median(Bc)),
        "Bc_p10": float(np.percentile(Bc, 10)),
        "Bc_p90": float(np.percentile(Bc, 90)),
        "Bc_p99": float(np.percentile(Bc, 99)),
        "Bc_max": int(Bc.max()), "Bc_min": int(Bc.min()),
        "Bc_std": float(Bc.std()),
        "bpp_from_Bc": float(Bc.mean() / 784.0),
        "train_reported_test_bpp": test_bpp,
        "gzip_mean_bits": float(gz.mean()),
        "gzip_bpp": float(gz.mean() / 784.0),
        "raw_payload_bits": D.K_PAYLOAD,
    }
    with open(os.path.join(RESULTS, "bc_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    np.save(os.path.join(RESULTS, "bc_lengths.npy"), Bc)

    print("\n==== B_c distribution (bits/image) ====")
    for k in ["Bc_mean", "Bc_median", "Bc_p10", "Bc_p90", "Bc_p99", "Bc_max",
              "Bc_min", "Bc_std", "bpp_from_Bc"]:
        print(f"  {k:14s} {stats[k]:.2f}")
    print(f"  sanity: NLL*784 = {test_bpp*784:.1f} bits vs mean B_c "
          f"{stats['Bc_mean']:.1f} bits (overhead "
          f"{stats['Bc_mean']-test_bpp*784:+.1f})")
    print(f"  gzip (context only): {stats['gzip_mean_bits']:.1f} bits/image "
          f"({stats['gzip_bpp']:.3f} bpp)")
    print(f"  raw payload: {D.K_PAYLOAD} bits/image (8.000 bpp)")


if __name__ == "__main__":
    main()
