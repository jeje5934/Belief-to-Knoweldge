"""Deterministic finite-precision arithmetic coder.

The probability table uses 16-bit integer frequencies while the interval
state uses 32 bits. Encoder and decoder receive the same quantized frequency
table at every source symbol.
"""

from __future__ import annotations

import numpy as np

PRECISION = 32
TOTAL_BITS = 16
TOTAL = 1 << TOTAL_BITS
FULL = (1 << PRECISION) - 1
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)
THREE_QUARTER = 3 * QUARTER
MASK = FULL


def quantize_pmf(pmf: np.ndarray) -> np.ndarray:
    """Map a floating PMF to positive integer frequencies summing to TOTAL."""

    n = pmf.shape[0]
    p = np.maximum(pmf.astype(np.float64), 0.0)
    total = p.sum()
    if total <= 0:
        p = np.ones(n, dtype=np.float64)
        total = float(n)
    p = p / total

    remaining = TOTAL - n
    scaled = p * remaining
    freq = np.floor(scaled).astype(np.int64) + 1
    deficit = TOTAL - int(freq.sum())
    if deficit > 0:
        fractions = scaled - np.floor(scaled)
        order = np.argsort(-fractions, kind="stable")
        for index in range(deficit):
            freq[order[index % n]] += 1
    elif deficit < 0:
        order = np.argsort(-freq, kind="stable")
        need = -deficit
        index = 0
        while need > 0:
            symbol = order[index % n]
            if freq[symbol] > 1:
                freq[symbol] -= 1
                need -= 1
            index += 1

    assert int(freq.sum()) == TOTAL
    assert freq.min() >= 1
    return freq


def _cdf_from_freq(freq: np.ndarray) -> np.ndarray:
    cdf = np.zeros(freq.shape[0] + 1, dtype=np.int64)
    np.cumsum(freq, out=cdf[1:])
    return cdf


class _BitWriter:
    def __init__(self) -> None:
        self.bits = bytearray()

    def write(self, bit: int) -> None:
        self.bits.append(bit & 1)

    def write_with_pending(self, bit: int, pending: int) -> None:
        self.write(bit)
        for _ in range(pending):
            self.write(bit ^ 1)

    def getvalue(self) -> np.ndarray:
        return np.frombuffer(bytes(self.bits), dtype=np.uint8).copy()


class ArithmeticEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = FULL
        self.pending = 0
        self.out = _BitWriter()

    def encode_symbol(self, symbol: int, freq: np.ndarray) -> None:
        cdf = _cdf_from_freq(freq)
        interval = self.high - self.low + 1
        self.high = self.low + interval * int(cdf[symbol + 1]) // TOTAL - 1
        self.low = self.low + interval * int(cdf[symbol]) // TOTAL

        while True:
            if self.high < HALF:
                self.out.write_with_pending(0, self.pending)
                self.pending = 0
            elif self.low >= HALF:
                self.out.write_with_pending(1, self.pending)
                self.pending = 0
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                return
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK

    def finish(self) -> np.ndarray:
        self.pending += 1
        if self.low < QUARTER:
            self.out.write_with_pending(0, self.pending)
        else:
            self.out.write_with_pending(1, self.pending)
        return self.out.getvalue()


class ArithmeticDecoder:
    def __init__(self, bitstream: np.ndarray) -> None:
        self.bits = np.asarray(bitstream, dtype=np.uint8)
        self.pos = 0
        self.low = 0
        self.high = FULL
        self.code = 0
        for _ in range(PRECISION):
            self.code = (self.code << 1) | self._next_bit()

    def _next_bit(self) -> int:
        if self.pos < self.bits.shape[0]:
            bit = int(self.bits[self.pos])
            self.pos += 1
            return bit
        return 0

    def decode_symbol(self, freq: np.ndarray) -> int:
        cdf = _cdf_from_freq(freq)
        interval = self.high - self.low + 1
        value = ((self.code - self.low + 1) * TOTAL - 1) // interval
        symbol = int(np.searchsorted(cdf, value, side="right") - 1)
        symbol = min(max(symbol, 0), freq.shape[0] - 1)

        self.high = self.low + interval * int(cdf[symbol + 1]) // TOTAL - 1
        self.low = self.low + interval * int(cdf[symbol]) // TOTAL
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK
            self.code = ((self.code << 1) | self._next_bit()) & MASK
        return symbol
