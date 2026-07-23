"""Data helpers shared by the neural-compression baseline."""

from __future__ import annotations

import torch


def to_model_input(uint8_imgs: torch.Tensor) -> torch.Tensor:
    """Normalize uint8 pixels from [0, 255] to [-1, 1]."""

    return uint8_imgs.float().div(255.0).mul(2.0).sub(1.0)
