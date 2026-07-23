"""Shared sampling, AWGN, confidence-interval, and metric helpers.

The LDPC and RSC runners execute in separate processes.  Keeping the payload
and noise plan in this dependency-light module makes their comparisons paired
without importing TensorFlow and PyTorch into the same process.
"""

from __future__ import annotations

import math

import numpy as np


PAYLOAD_BITS = 6272
TRANSMITTED_BITS = 12600
PAYLOAD_RATE = PAYLOAD_BITS / TRANSMITTED_BITS


def wilson_interval(errors: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or not 0 <= errors <= total:
        raise ValueError("Wilson interval requires 0 <= errors <= total and total > 0")
    proportion = errors / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def load_fashion_mnist_bits(root: str = "/tmp/fmnist") -> np.ndarray:
    import torchvision

    dataset = torchvision.datasets.FashionMNIST(root, train=False, download=False)
    images = np.stack(
        [np.asarray(dataset[index][0], dtype=np.uint8) for index in range(len(dataset))],
        axis=0,
    )
    bits = np.unpackbits(images.reshape(len(dataset), -1), axis=-1)
    if bits.shape[1] != PAYLOAD_BITS:
        raise AssertionError(f"unexpected payload width {bits.shape[1]}")
    return bits


def _snr_seed(seed: int, esn0_db: float) -> np.random.SeedSequence:
    snr_key = int(round(float(esn0_db) * 1000.0)) + 1_000_000
    if snr_key < 0:
        raise ValueError("Es/N0 is outside the supported deterministic seed range")
    return np.random.SeedSequence([int(seed), snr_key])


def paired_batches(
    bank: np.ndarray,
    *,
    blocks: int,
    batch_size: int,
    esn0_db: float,
    seed: int,
):
    """Yield identical payload indices and standard noise in every arm process."""

    if blocks <= 0 or batch_size <= 0:
        raise ValueError("blocks and batch_size must be positive")
    rng = np.random.default_rng(_snr_seed(seed, esn0_db))
    produced = 0
    while produced < blocks:
        count = min(batch_size, blocks - produced)
        indices = rng.integers(0, bank.shape[0], size=count, dtype=np.int64)
        noise = rng.standard_normal((count, TRANSMITTED_BITS))
        yield indices, bank[indices], noise
        produced += count


def awgn_llr(bits: np.ndarray, standard_noise: np.ndarray, esn0_db: float) -> np.ndarray:
    """BPSK/AWGN logits with bit 0 -> -1, bit 1 -> +1 and logit > 0 -> bit 1."""

    binary = np.asarray(bits, dtype=np.uint8)
    noise = np.asarray(standard_noise, dtype=np.float64)
    if binary.shape != noise.shape or binary.shape[-1] != TRANSMITTED_BITS:
        raise ValueError("bits and noise must have equal [..., 12600] shape")
    n0 = 10.0 ** (-float(esn0_db) / 10.0)
    received = 2.0 * binary.astype(np.float64) - 1.0
    received += np.sqrt(n0 / 2.0) * noise
    return 4.0 * received / n0


def completed_row(
    *,
    arm: str,
    esn0_db: float,
    blocks: int,
    bit_errors: int,
    true_block_errors: int,
    crc_block_errors: int,
    undetected_errors: int,
    crc_fail_payload_correct: int,
    elapsed_seconds: float,
) -> dict:
    return {
        "arm": arm,
        "esn0_db": float(esn0_db),
        "blocks": int(blocks),
        "bit_errors": int(bit_errors),
        "ber": float(bit_errors / (blocks * PAYLOAD_BITS)),
        "true_payload_block_errors": int(true_block_errors),
        "true_payload_bler": float(true_block_errors / blocks),
        "true_payload_bler_wilson95": wilson_interval(true_block_errors, blocks),
        "crc_detected_block_errors": int(crc_block_errors),
        "crc_detected_bler": float(crc_block_errors / blocks),
        "crc_detected_bler_wilson95": wilson_interval(crc_block_errors, blocks),
        "undetected_errors": int(undetected_errors),
        "crc_fail_payload_correct": int(crc_fail_payload_correct),
        "elapsed_seconds": float(elapsed_seconds),
    }


def interpolate_log_bler_knee(rows: list[dict], target: float) -> float | None:
    """Interpolate Es/N0 linearly against log10(BLER) between bracketing points."""

    if not 0.0 < target < 1.0:
        raise ValueError("target BLER must lie in (0, 1)")
    points = sorted(
        (float(row["esn0_db"]), float(row["true_payload_bler"]))
        for row in rows if float(row["true_payload_bler"]) > 0.0
    )
    for (x0, p0), (x1, p1) in zip(points, points[1:]):
        if (p0 >= target >= p1) or (p1 >= target >= p0):
            if p0 == p1:
                return (x0 + x1) / 2.0
            log_target = math.log10(target)
            fraction = (
                (log_target - math.log10(p0))
                / (math.log10(p1) - math.log10(p0))
            )
            return x0 + fraction * (x1 - x0)
    return None
