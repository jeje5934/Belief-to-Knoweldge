"""Gated PixelCNN used as a lossless entropy model for Fashion-MNIST."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

N_LEVELS = 256


class VerticalMaskedConv(nn.Conv2d):
    """Square convolution whose output row sees only earlier image rows."""

    def __init__(self, in_ch: int, out_ch: int, k: int) -> None:
        super().__init__(in_ch, out_ch, k, padding=k // 2)
        mask = torch.zeros_like(self.weight)
        mask[:, :, : k // 2 + 1, :] = 1.0
        mask[:, :, k // 2 :, :] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        output = super().forward(x)
        return output[:, :, : x.size(2), :]


class HorizontalMaskedConv(nn.Conv2d):
    """Horizontal convolution excluding (A) or including (B) current pixel."""

    def __init__(self, in_ch: int, out_ch: int, k: int, mask_type: str) -> None:
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'")
        super().__init__(in_ch, out_ch, (1, k), padding=(0, k // 2))
        mask = torch.zeros_like(self.weight)
        mask[:, :, :, : k // 2] = 1.0
        if mask_type == "B":
            mask[:, :, :, k // 2] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        output = super().forward(x)
        return output[:, :, :, : x.size(3)]


class GatedBlock(nn.Module):
    def __init__(self, n_ch: int, k: int, mask_type: str) -> None:
        super().__init__()
        self.vconv = VerticalMaskedConv(n_ch, 2 * n_ch, k)
        self.v_to_h = nn.Conv2d(2 * n_ch, 2 * n_ch, 1)
        self.hconv = HorizontalMaskedConv(n_ch, 2 * n_ch, k, mask_type)
        self.h_out = nn.Conv2d(n_ch, n_ch, 1)
        self.residual = mask_type == "B"

    @staticmethod
    def _gate(z: torch.Tensor) -> torch.Tensor:
        first, second = z.chunk(2, dim=1)
        return torch.tanh(first) * torch.sigmoid(second)

    def forward(
        self, vertical: torch.Tensor, horizontal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vertical_features = self.vconv(vertical)
        vertical_out = self._gate(vertical_features)
        horizontal_features = self.hconv(horizontal) + self.v_to_h(vertical_features)
        horizontal_features = self._gate(horizontal_features)
        horizontal_features = self.h_out(horizontal_features)
        horizontal_out = (
            horizontal + horizontal_features if self.residual else horizontal_features
        )
        return vertical_out, horizontal_out


class GatedPixelCNN(nn.Module):
    def __init__(
        self,
        n_channels: int = 120,
        n_layers: int = 15,
        k: int = 7,
        n_levels: int = N_LEVELS,
    ) -> None:
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

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        vertical = self.in_v(x_norm)
        horizontal = self.in_h(x_norm)
        for block in self.blocks:
            vertical, horizontal = block(vertical, horizontal)
        return self.out(horizontal)

    def loss_bits_per_pixel(
        self, x_uint8: torch.Tensor, x_norm: torch.Tensor
    ) -> torch.Tensor:
        logits = self.forward(x_norm)
        target = x_uint8.long().squeeze(1)
        nll_nats = F.cross_entropy(logits, target, reduction="mean")
        return nll_nats / torch.log(torch.tensor(2.0, device=nll_nats.device))


def count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
