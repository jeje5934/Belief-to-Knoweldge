"""Systematic per-source-symbol single-parity-check code."""

from __future__ import annotations

import numpy as np


def _validate_bits(values) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint8)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must contain only 0 and 1")
    return bits


def candidate_bits(bits_per_symbol: int) -> np.ndarray:
    """Return all symbol values in MSB-first bit order, shape [2**M, M]."""
    if bits_per_symbol <= 0:
        raise ValueError("bits_per_symbol must be positive")
    values = np.arange(1 << bits_per_symbol, dtype=np.uint64)[:, None]
    shifts = np.arange(bits_per_symbol - 1, -1, -1, dtype=np.uint64)[None, :]
    return ((values >> shifts) & 1).astype(np.uint8)


def encode_spc(payload_bits, bits_per_symbol: int) -> np.ndarray:
    """Encode each M-bit source symbol as [systematic M bits, XOR parity]."""
    payload = _validate_bits(payload_bits)
    if payload.ndim == 0 or payload.shape[-1] % bits_per_symbol:
        raise ValueError("payload length must be divisible by bits_per_symbol")
    symbol_count = payload.shape[-1] // bits_per_symbol
    systematic = payload.reshape(payload.shape[:-1] + (symbol_count, bits_per_symbol))
    parity = np.bitwise_xor.reduce(systematic, axis=-1, keepdims=True)
    codeword = np.concatenate([systematic, parity], axis=-1)
    return codeword.reshape(payload.shape[:-1] + (symbol_count * (bits_per_symbol + 1),))


def split_spc(codeword_bits, bits_per_symbol: int) -> tuple[np.ndarray, np.ndarray]:
    codeword = _validate_bits(codeword_bits)
    block = bits_per_symbol + 1
    if codeword.ndim == 0 or codeword.shape[-1] % block:
        raise ValueError("SPC codeword length must be divisible by M+1")
    symbol_count = codeword.shape[-1] // block
    shaped = codeword.reshape(codeword.shape[:-1] + (symbol_count, block))
    return shaped[..., :bits_per_symbol], shaped[..., bits_per_symbol]


def decode_systematic(codeword_bits, bits_per_symbol: int) -> np.ndarray:
    systematic, _ = split_spc(codeword_bits, bits_per_symbol)
    return systematic.reshape(systematic.shape[:-2] + (-1,))


def parity_is_valid(codeword_bits, bits_per_symbol: int) -> np.ndarray:
    systematic, parity = split_spc(codeword_bits, bits_per_symbol)
    return np.bitwise_xor.reduce(systematic, axis=-1) == parity
