"""Sequential lossless PixelCNN arithmetic codec."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .arithmetic_coder import ArithmeticDecoder, ArithmeticEncoder, quantize_pmf
from .data import to_model_input
from .pixelcnn import GatedPixelCNN

HEIGHT = 28
WIDTH = 28


def load_model(checkpoint: Path | str, device: str = "cpu") -> tuple[GatedPixelCNN, dict]:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"neural-compression checkpoint not found: {checkpoint}"
        )
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    config = saved["cfg"]
    model = GatedPixelCNN(
        config["n_channels"], config["n_layers"], config["k"]
    ).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


@torch.inference_mode()
def _step_pmf(
    model: GatedPixelCNN, canvas: torch.Tensor, row: int, column: int
) -> np.ndarray:
    # Arithmetic coding requires encoder and decoder probabilities to be
    # bit-identical. GPU convolution reductions can change at float precision
    # when the batch shape changes, which is especially dangerous when only a
    # CRC-passing subset is decoded. Canonicalize every probability evaluation
    # to batch size one so stream syntax is independent of batch partitioning.
    probabilities = []
    for index in range(canvas.shape[0]):
        logits = model(to_model_input(canvas[index : index + 1]))
        probability = (
            F.softmax(logits[:, :, row, column].float(), dim=1)
            .double()
            .cpu()
            .numpy()[0]
        )
        probabilities.append(probability)
    return np.stack(probabilities, axis=0)


@torch.inference_mode()
def encode_batch(
    model: GatedPixelCNN, imgs_u8: torch.Tensor, device: str
) -> tuple[list[np.ndarray], list[int]]:
    """Sequentially encode a batch of 28x28 uint8 images."""

    if imgs_u8.ndim == 3:
        imgs_u8 = imgs_u8.unsqueeze(1)
    if tuple(imgs_u8.shape[1:]) != (1, HEIGHT, WIDTH):
        raise ValueError("imgs_u8 must have shape [B,1,28,28] or [B,28,28]")
    batch = imgs_u8.size(0)
    images = imgs_u8.to(device)
    encoders = [ArithmeticEncoder() for _ in range(batch)]
    canvas = torch.zeros(
        batch, 1, HEIGHT, WIDTH, dtype=torch.uint8, device=device
    )
    for row in range(HEIGHT):
        for column in range(WIDTH):
            pmf = _step_pmf(model, canvas, row, column)
            for index in range(batch):
                frequencies = quantize_pmf(pmf[index])
                symbol = int(images[index, 0, row, column])
                encoders[index].encode_symbol(symbol, frequencies)
                canvas[index, 0, row, column] = symbol
    streams = [encoder.finish() for encoder in encoders]
    return streams, [int(stream.shape[0]) for stream in streams]


@torch.inference_mode()
def decode_batch(
    model: GatedPixelCNN, streams: list[np.ndarray], device: str
) -> torch.Tensor:
    """Decode arithmetic bitstreams to uint8 images."""

    batch = len(streams)
    decoders = [ArithmeticDecoder(stream) for stream in streams]
    canvas = torch.zeros(
        batch, 1, HEIGHT, WIDTH, dtype=torch.uint8, device=device
    )
    for row in range(HEIGHT):
        for column in range(WIDTH):
            pmf = _step_pmf(model, canvas, row, column)
            for index in range(batch):
                frequencies = quantize_pmf(pmf[index])
                symbol = decoders[index].decode_symbol(frequencies)
                canvas[index, 0, row, column] = symbol
    return canvas.cpu()
