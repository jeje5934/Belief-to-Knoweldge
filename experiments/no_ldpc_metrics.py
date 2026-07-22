"""Shared payload/image metrics for no-LDPC experiments."""

from __future__ import annotations

import numpy as np


def payload_bits_to_uint8_images(
    payload_bits,
    source_shape: tuple[int, ...],
    bits_per_symbol: int,
) -> np.ndarray:
    bits = np.asarray(payload_bits, dtype=np.uint8)
    if bits_per_symbol != 8:
        raise ValueError("uint8 image metrics currently require M=8")
    expected = int(np.prod(source_shape)) * bits_per_symbol
    if bits.shape[-1] != expected:
        raise ValueError("payload length does not match image shape")
    packed = np.packbits(bits, axis=-1, bitorder="big")
    return packed.reshape(bits.shape[:-1] + source_shape)


def _uniform_window_ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SSIM with a 7x7 uniform window and standard K1/K2 constants."""
    from numpy.lib.stride_tricks import sliding_window_view

    x = reference.astype(np.float64)
    y = estimate.astype(np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("SSIM expects equal two-dimensional images")
    window = min(7, x.shape[0], x.shape[1])
    if window < 2:
        window = 1
    xw = sliding_window_view(x, (window, window))
    yw = sliding_window_view(y, (window, window))
    axes = (-2, -1)
    mu_x = xw.mean(axis=axes)
    mu_y = yw.mean(axis=axes)
    var_x = (xw * xw).mean(axis=axes) - mu_x * mu_x
    var_y = (yw * yw).mean(axis=axes) - mu_y * mu_y
    covariance = (xw * yw).mean(axis=axes) - mu_x * mu_y
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (
        var_x + var_y + c2)
    return float(np.mean(numerator / denominator))


def _psnr_from_mse(mse: float):
    if mse == 0.0:
        return "inf"
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def image_metrics(
    payload_true,
    payload_hat,
    source_shape: tuple[int, ...],
    bits_per_symbol: int,
    crc_valid,
) -> dict:
    reference = payload_bits_to_uint8_images(
        payload_true, source_shape, bits_per_symbol)
    estimate = payload_bits_to_uint8_images(
        payload_hat, source_shape, bits_per_symbol)
    flat_reference = reference.reshape(-1, *source_shape)
    flat_estimate = estimate.reshape(-1, *source_shape)
    squared_error = (
        flat_reference.astype(np.float64) - flat_estimate.astype(np.float64)) ** 2
    block_mse = squared_error.reshape(squared_error.shape[0], -1).mean(axis=-1)
    mean_mse = float(block_mse.mean())
    ssim = np.asarray([
        _uniform_window_ssim(original, decoded)
        for original, decoded in zip(flat_reference, flat_estimate)
    ])
    crc_failure = ~np.asarray(crc_valid, dtype=bool).reshape(-1)
    conditional_mse = (
        float(block_mse[crc_failure].mean()) if np.any(crc_failure) else None)
    return {
        "psnr_db": _psnr_from_mse(mean_mse),
        "ssim_mean": float(ssim.mean()),
        "expected_mse_0_1": mean_mse / (255.0 ** 2),
        "crc_failure_psnr_db": (
            None if conditional_mse is None else _psnr_from_mse(conditional_mse)),
        "crc_failure_blocks_for_distortion": int(crc_failure.sum()),
    }
