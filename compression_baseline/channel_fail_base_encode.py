"""
Failure-quality (baseline) stage A (torch, canonical cuDNN-off codec): encode a
few hundred images into MAX-container bits (K = 4872, zero-padded) with the SAME
deterministic codec the decode stage uses, so the control (true bits) round-trips
bit-exact and only channel corruption causes the catastrophic reconstruction.
"""
import os
import numpy as np
import torch
torch.backends.cudnn.enabled = False           # canonical: deterministic + batch-invariant
from compression_baseline import data as D
from compression_baseline.compress import load_model, encode_batch

RESULTS = "/home/LJH/onlyextrinsic_ada_sigma/compression_baseline/results"
OUT = os.path.join(RESULTS, "fail_base_enc.npz")
K_MAX = 4872                                    # established MAX container payload


def main(n=512, batch=128):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(dev)
    imgs = D.load_test()[:n]
    streams, lengths = [], []
    for i in range(0, n, batch):
        s, L = encode_batch(model, imgs[i:i + batch], dev)
        streams.extend(s); lengths.extend(L)
        print(f"  encoded {len(lengths)}/{n}", flush=True)
    lengths = np.array(lengths, dtype=np.int32)
    assert lengths.max() <= K_MAX, f"B_c {lengths.max()} exceeds container {K_MAX}"
    bits = np.zeros((n, K_MAX), dtype=np.uint8)
    for i, s in enumerate(streams):
        bits[i, : s.shape[0]] = s
    px = imgs.reshape(n, D.NPIX).numpy().astype(np.uint8)
    np.savez(OUT, bits=bits, lengths=lengths, imgs=px, K=K_MAX)
    print(f"saved {OUT}  n={n}  B_c max={lengths.max()} (container {K_MAX})")


if __name__ == "__main__":
    main()
