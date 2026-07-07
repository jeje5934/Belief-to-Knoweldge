"""
Failure-quality (baseline) stage 2 (torch): AC-decode the CORRUPTED container
bits captured by stage 1 and reconstruct the image.  Because arithmetic coding
is a sequential dependent code, any surviving bit error desyncs the decoder from
that point on -> the reconstruction is catastrophic (all-or-nothing).  We also
AC-decode the TRUE container bits as a control (must reproduce the original).

Canonical codec = cuDNN off (deterministic + batch-invariant).
"""
import os
import numpy as np
import torch
torch.backends.cudnn.enabled = False
from compression_baseline import data as D
from compression_baseline.compress import load_model, decode_batch

RESULTS = "/home/LJH/onlyextrinsic_ada_sigma/compression_baseline/results"
IN = os.path.join(RESULTS, "fail_base_bits.npz")
OUT = os.path.join(RESULTS, "fail_base.npz")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2, axis=(1, 2))
    return np.where(mse == 0, 99.0, 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-9)))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(dev)
    d = np.load(IN)
    rec_bits, true_bits = d["rec_bits"], d["true_bits"]
    orig_px = d["orig_px"].reshape(-1, 28, 28)
    n = rec_bits.shape[0]

    # AC-decode the corrupted containers (decoder reads bits as needed for 784 px)
    corrupt_streams = [rec_bits[i].astype(np.uint8) for i in range(n)]
    true_streams = [true_bits[i].astype(np.uint8) for i in range(n)]
    recon = decode_batch(model, corrupt_streams, dev).squeeze(1).numpy().astype(np.uint8)
    ctrl = decode_batch(model, true_streams, dev).squeeze(1).numpy().astype(np.uint8)

    ps = psnr(recon, orig_px)
    ctrl_exact = int((ctrl == orig_px).all(axis=(1, 2)).sum())
    np.savez(OUT, recon=recon, orig=orig_px, psnr=ps, esno=float(d["esno"]),
             ctrl_exact=ctrl_exact)
    print(f"baseline Es/N0={float(d['esno'])}: AC-decoded {n} corrupted containers  "
          f"PSNR mean={ps.mean():.2f} dB median={np.median(ps):.2f} "
          f"min={ps.min():.2f} max={ps.max():.2f}")
    print(f"  control (true bits) reconstructed bit-exact: {ctrl_exact}/{n}")


if __name__ == "__main__":
    main()
