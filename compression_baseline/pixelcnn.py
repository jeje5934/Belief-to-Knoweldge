"""
Gated PixelCNN (van den Oord et al., 2016) for 28x28x1 8-bit Fashion-MNIST.

Autoregressive model in raster-scan order.  Output is a 256-way softmax per
pixel: p(x_i = v | x_{<i}) for v in 0..255.  This conditional is what the
arithmetic coder turns into a bit-exact code.

Capacity is set to land in the 5M-9M parameter band (same order as the
transmission system's ~7M-param EDM denoiser) via `n_channels`/`n_layers`.

The two-stack (vertical + horizontal) construction gives a proper causal
receptive field with no blind spot, which the single-mask PixelCNN lacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LEVELS = 256   # 8-bit pixels


class VerticalMaskedConv(nn.Conv2d):
    """(k//2+1) x k conv, zero-padded and shifted so row i sees only rows < i.

    Implemented as a full k x k conv with the lower half of the kernel masked
    to zero; a downward shift (crop) removes the current-row leakage.
    """

    def __init__(self, in_ch, out_ch, k):
        super().__init__(in_ch, out_ch, k, padding=k // 2)
        mask = torch.zeros_like(self.weight)
        mask[:, :, : k // 2 + 1, :] = 1.0   # rows above and including centre
        mask[:, :, k // 2, :] = 0.0          # drop centre row (added back via shift)
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)
        # crop the extra bottom row created by padding so output aligns causally
        return out[:, :, : x.size(2), :]


class HorizontalMaskedConv(nn.Conv2d):
    """1 x (k//2+1) conv: pixel i sees only pixels to its left in the same row.

    `mask_type` 'A' excludes the current pixel (first layer); 'B' includes it.
    """

    def __init__(self, in_ch, out_ch, k, mask_type):
        super().__init__(in_ch, out_ch, (1, k), padding=(0, k // 2))
        mask = torch.zeros_like(self.weight)
        mask[:, :, :, : k // 2] = 1.0
        if mask_type == "B":
            mask[:, :, :, k // 2] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)
        return out[:, :, :, : x.size(3)]


class GatedBlock(nn.Module):
    """One gated PixelCNN layer with coupled vertical + horizontal stacks."""

    def __init__(self, n_ch, k, mask_type):
        super().__init__()
        self.vconv = VerticalMaskedConv(n_ch, 2 * n_ch, k)
        self.v_to_h = nn.Conv2d(2 * n_ch, 2 * n_ch, 1)
        self.hconv = HorizontalMaskedConv(n_ch, 2 * n_ch, k, mask_type)
        self.h_out = nn.Conv2d(n_ch, n_ch, 1)
        self.residual = (mask_type == "B")

    @staticmethod
    def _gate(z):
        a, b = z.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)

    def forward(self, v_in, h_in):
        v = self.vconv(v_in)
        v_out = self._gate(v)

        h = self.hconv(h_in) + self.v_to_h(v)
        h = self._gate(h)
        h = self.h_out(h)
        h_out = h_in + h if self.residual else h
        return v_out, h_out


class GatedPixelCNN(nn.Module):
    def __init__(self, n_channels=120, n_layers=15, k=7, n_levels=N_LEVELS):
        super().__init__()
        self.n_levels = n_levels
        self.in_v = VerticalMaskedConv(1, n_channels, k)
        self.in_h = HorizontalMaskedConv(1, n_channels, k, mask_type="A")

        self.blocks = nn.ModuleList(
            [GatedBlock(n_channels, k, mask_type="B") for _ in range(n_layers)]
        )
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_levels, 1),
        )

    def forward(self, x_norm):
        """x_norm: [B,1,H,W] float in [-1,1].  Returns logits [B,256,H,W]."""
        v = self.in_v(x_norm)
        h = self.in_h(x_norm)
        for blk in self.blocks:
            v, h = blk(v, h)
        return self.out(h)

    def loss_bits_per_pixel(self, x_uint8, x_norm):
        """Mean NLL in bits/pixel for a batch (the training objective)."""
        logits = self.forward(x_norm)                 # [B,256,H,W]
        target = x_uint8.long().squeeze(1)            # [B,H,W]
        nll_nats = F.cross_entropy(logits, target, reduction="mean")
        return nll_nats / torch.log(torch.tensor(2.0, device=nll_nats.device))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for ch, L in [(96, 12), (120, 15), (128, 16), (140, 15)]:
        m = GatedPixelCNN(n_channels=ch, n_layers=L)
        print(f"n_channels={ch:3d} n_layers={L:2d} -> {count_params(m)/1e6:.2f}M params")
