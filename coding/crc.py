"""Bit-oriented CRC helpers.

The no-LDPC path uses MSB-first CRC-16-CCITT-FALSE by default:
polynomial 0x1021, initial register 0xFFFF, and xor-out 0x0000.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _bits(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.uint8)
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError("bits must contain only 0 and 1")
    return arr


@dataclass(frozen=True)
class CRC:
    """Configurable non-reflected, MSB-first CRC."""

    width: int
    polynomial: int
    init: int
    xor_out: int = 0
    name: str = "CRC"

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("CRC width must be positive")
        limit = 1 << self.width
        for field_name in ("polynomial", "init", "xor_out"):
            value = getattr(self, field_name)
            if not 0 <= value < limit:
                raise ValueError(f"{field_name} does not fit CRC width")

    @property
    def mask(self) -> int:
        return (1 << self.width) - 1

    def compute(self, message_bits) -> int:
        """Return the integer checksum for one one-dimensional bit vector."""
        data = _bits(message_bits)
        if data.ndim != 1:
            raise ValueError("compute expects one one-dimensional message")
        register = self.init & self.mask
        top_bit = 1 << (self.width - 1)
        for bit in data:
            feedback = bool(register & top_bit) ^ bool(bit)
            register = (register << 1) & self.mask
            if feedback:
                register ^= self.polynomial
        return register ^ self.xor_out

    def checksum_bits(self, message_bits) -> np.ndarray:
        value = self.compute(message_bits)
        shifts = np.arange(self.width - 1, -1, -1, dtype=np.int64)
        return ((value >> shifts) & 1).astype(np.uint8)

    def append(self, message_bits) -> np.ndarray:
        """Append CRC bits along the last axis; leading batch axes are preserved."""
        data = _bits(message_bits)
        if data.ndim == 0:
            raise ValueError("message must have a bit axis")
        flat = data.reshape(-1, data.shape[-1])
        checksums = np.stack([self.checksum_bits(row) for row in flat], axis=0)
        encoded = np.concatenate([flat, checksums], axis=-1)
        return encoded.reshape(data.shape[:-1] + (encoded.shape[-1],))

    def split(self, codeword_bits) -> tuple[np.ndarray, np.ndarray]:
        codeword = _bits(codeword_bits)
        if codeword.shape[-1] < self.width:
            raise ValueError("codeword is shorter than the CRC")
        return codeword[..., :-self.width], codeword[..., -self.width :]

    def check_parts(self, message_bits, checksum_bits) -> np.ndarray | bool:
        message = _bits(message_bits)
        checksum = _bits(checksum_bits)
        if message.shape[:-1] != checksum.shape[:-1]:
            raise ValueError("message and checksum batch shapes differ")
        if checksum.shape[-1] != self.width:
            raise ValueError("checksum has the wrong width")
        flat_message = message.reshape(-1, message.shape[-1])
        flat_checksum = checksum.reshape(-1, self.width)
        valid = np.array(
            [np.array_equal(self.checksum_bits(m), c)
             for m, c in zip(flat_message, flat_checksum)],
            dtype=bool,
        ).reshape(message.shape[:-1])
        return bool(valid) if valid.ndim == 0 else valid

    def check(self, codeword_bits) -> np.ndarray | bool:
        message, checksum = self.split(codeword_bits)
        return self.check_parts(message, checksum)


CRC16_CCITT = CRC(
    width=16,
    polynomial=0x1021,
    init=0xFFFF,
    xor_out=0x0000,
    name="CRC-16-CCITT-FALSE",
)
