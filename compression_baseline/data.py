"""
Fashion-MNIST loader for the compression baseline.

Fairness note: this uses the *exact* same data source as the transmission
system's denoiser (torchvision FashionMNIST cached at /tmp/fmnist).  The
transmission payload treats each image as 784 pixels x 8 bit = 6272 bits, i.e.
uint8 pixel values in 0..255.  The compressor is trained and evaluated on that
same 8-bit representation so that "compress then transmit" is bit-for-bit
comparable with the raw-payload path.

We deliberately return integer pixels in {0,...,255} (not [0,1] floats) because
lossless compression must round-trip the exact 8-bit payload.
"""

import numpy as np
import torch
import torchvision

FMNIST_ROOT = "/tmp/fmnist"
IMG_H = 28
IMG_W = 28
NPIX = IMG_H * IMG_W          # 784
BITS_PER_PIXEL = 8
K_PAYLOAD = NPIX * BITS_PER_PIXEL   # 6272 bits, matches the transmission system


def _load_split(train: bool) -> torch.Tensor:
    """Return uint8 tensor [N, 1, 28, 28] with pixel values in 0..255."""
    ds = torchvision.datasets.FashionMNIST(root=FMNIST_ROOT, train=train,
                                           download=True)
    # ds.data is a uint8 tensor [N, 28, 28] with the raw 0..255 pixels — this is
    # exactly the 8-bit payload the transmission system serialises.
    imgs = ds.data.to(torch.uint8).unsqueeze(1).contiguous()
    return imgs


def load_train() -> torch.Tensor:
    return _load_split(train=True)


def load_test() -> torch.Tensor:
    return _load_split(train=False)


def to_model_input(uint8_imgs: torch.Tensor) -> torch.Tensor:
    """Normalise uint8 pixels 0..255 -> float in [-1, 1] for the network input.

    The 256-way softmax *target* stays the integer pixel value; only the network
    *input* is normalised for numerical conditioning.
    """
    return uint8_imgs.float().div(255.0).mul(2.0).sub(1.0)


if __name__ == "__main__":
    tr = load_train()
    te = load_test()
    print("train:", tuple(tr.shape), tr.dtype, "min", int(tr.min()), "max", int(tr.max()))
    print("test :", tuple(te.shape), te.dtype)
    print("K_payload bits/image:", K_PAYLOAD)
    # fraction of exactly-zero pixels (drives compressibility of the background)
    z = (te == 0).float().mean().item()
    print(f"test fraction of exact-zero pixels: {z:.3f}")
