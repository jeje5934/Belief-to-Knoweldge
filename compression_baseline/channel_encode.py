"""
Phase-2 stage A (torch / GPU, NO TensorFlow in this process).

Encode a fixed set of test images with the PixelCNN + arithmetic coder and dump
the compressed bitstreams + originals to an .npz.  The channel/LDPC stage
(`channel_experiment.py`) is a *separate* TF process — mixing torch-CUDA and
TF-CUDA in one process segfaults on this host (docs/COMPUTE_LESSONS §5).

Output: results/channel_streams.npz
    bits    uint8 [N, MAXLEN]  raster-scan compressed bits (0/1), zero-padded
    lengths int32 [N]          true B_c per image
    imgs    uint8 [N, 784]     original payload bits? no -> original pixels
    idx     int32 [N]          test-set indices used
"""

import argparse
import os

import numpy as np
import torch

from compression_baseline import data as D
from compression_baseline.compress import load_model, encode_batch

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "channel_streams.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3200)
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.5, 0)
    model, test_bpp = load_model(device)
    print(f"model loaded (test_bpp={test_bpp:.4f}) device={device}")

    test_u8 = D.load_test()[:args.n]          # [N,1,28,28]
    N = test_u8.size(0)

    all_streams, lengths = [], []
    for i in range(0, N, args.batch):
        streams, L = encode_batch(model, test_u8[i:i + args.batch], device)
        all_streams.extend(streams)
        lengths.extend(L)
        print(f"  encoded {len(lengths)}/{N}", flush=True)

    lengths = np.array(lengths, dtype=np.int32)
    maxlen = int(lengths.max())
    bits = np.zeros((N, maxlen), dtype=np.uint8)
    for i, s in enumerate(all_streams):
        bits[i, : s.shape[0]] = s

    imgs = test_u8.reshape(N, D.NPIX).numpy().astype(np.uint8)   # 0..255 pixels
    np.savez_compressed(OUT, bits=bits, lengths=lengths, imgs=imgs,
                        idx=np.arange(N, dtype=np.int32))
    print(f"saved {OUT}")
    print(f"  N={N}  B_c: mean={lengths.mean():.1f} p99={np.percentile(lengths,99):.1f} "
          f"max={lengths.max()}  maxlen(bits array)={maxlen}")


if __name__ == "__main__":
    main()
