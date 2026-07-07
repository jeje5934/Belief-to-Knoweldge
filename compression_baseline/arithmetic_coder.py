"""
Bit-exact arithmetic coder (Witten-Neal-Cleary integer variant).

The model supplies, for each pixel, a 256-way probability vector.  We convert it
to a **16-bit quantized cumulative frequency table** (integers, total = 2**16)
and drive a 32-bit integer arithmetic coder off that table.  Because the encoder
and decoder build the CDF from *identical* integer frequencies (the decoder
reconstructs each pixel exactly before predicting the next, so the model inputs
match bit-for-bit), the round trip is provably lossless.

Nothing here touches floats during coding — the quantized integer table is the
single source of truth shared by encoder and decoder.

References: Witten, Neal & Cleary, "Arithmetic Coding for Data Compression",
CACM 1987; MacKay, ITILA ch. 6.
"""

import numpy as np

PRECISION = 32
TOTAL_BITS = 16                     # frequencies sum to 2**16
TOTAL = 1 << TOTAL_BITS
FULL = (1 << PRECISION) - 1
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)
THREE_QUARTER = 3 * QUARTER
MASK = FULL

# TOTAL must be < QUARTER so that the coder never overflows (WNC requirement).
assert TOTAL < QUARTER


def quantize_pmf(pmf: np.ndarray) -> np.ndarray:
    """Map a float pmf (len 256, sums ~1) to integer frequencies summing to TOTAL.

    Every symbol gets frequency >= 1 (so nothing is uncodable), the remainder is
    distributed by largest fractional part, and the result is deterministic given
    the input float array — encoder and decoder call this with identical inputs.
    """
    n = pmf.shape[0]
    p = np.maximum(pmf.astype(np.float64), 0.0)
    s = p.sum()
    if s <= 0:
        p = np.ones(n, dtype=np.float64)
        s = float(n)
    p = p / s

    # start every symbol at 1 to guarantee codability, share the rest by prob
    remaining = TOTAL - n
    scaled = p * remaining
    freq = np.floor(scaled).astype(np.int64) + 1
    deficit = TOTAL - int(freq.sum())
    if deficit > 0:
        # hand the leftover counts to the largest fractional remainders
        frac = scaled - np.floor(scaled)
        order = np.argsort(-frac, kind="stable")
        for i in range(deficit):
            freq[order[i % n]] += 1
    elif deficit < 0:
        # too many: remove from the largest freqs that can spare a count
        order = np.argsort(-freq, kind="stable")
        need = -deficit
        j = 0
        while need > 0:
            idx = order[j % n]
            if freq[idx] > 1:
                freq[idx] -= 1
                need -= 1
            j += 1
    assert int(freq.sum()) == TOTAL
    assert freq.min() >= 1
    return freq


def _cdf_from_freq(freq: np.ndarray) -> np.ndarray:
    cdf = np.zeros(freq.shape[0] + 1, dtype=np.int64)
    np.cumsum(freq, out=cdf[1:])
    return cdf


class BitWriter:
    def __init__(self):
        self.bits = bytearray()

    def write(self, bit):
        self.bits.append(bit & 1)

    def write_with_pending(self, bit, pending):
        self.write(bit)
        for _ in range(pending):
            self.write(bit ^ 1)

    def getvalue(self):
        return np.frombuffer(bytes(self.bits), dtype=np.uint8).copy()


class ArithmeticEncoder:
    def __init__(self):
        self.low = 0
        self.high = FULL
        self.pending = 0
        self.out = BitWriter()

    def encode_symbol(self, sym, freq):
        cdf = _cdf_from_freq(freq)
        rng = self.high - self.low + 1
        self.high = self.low + (rng * int(cdf[sym + 1])) // TOTAL - 1
        self.low = self.low + (rng * int(cdf[sym])) // TOTAL
        # renormalise
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
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK

    def finish(self):
        # flush: emit enough bits to disambiguate the final interval
        self.pending += 1
        if self.low < QUARTER:
            self.out.write_with_pending(0, self.pending)
        else:
            self.out.write_with_pending(1, self.pending)
        return self.out.getvalue()


class ArithmeticDecoder:
    def __init__(self, bitstream: np.ndarray):
        self.bits = bitstream
        self.pos = 0
        self.low = 0
        self.high = FULL
        self.code = 0
        for _ in range(PRECISION):
            self.code = (self.code << 1) | self._next_bit()

    def _next_bit(self):
        if self.pos < self.bits.shape[0]:
            b = int(self.bits[self.pos])
            self.pos += 1
            return b
        return 0   # past the end: pad with zeros

    def decode_symbol(self, freq):
        cdf = _cdf_from_freq(freq)
        rng = self.high - self.low + 1
        # locate the symbol whose cumulative interval contains `code`
        value = (((self.code - self.low) + 1) * TOTAL - 1) // rng
        sym = int(np.searchsorted(cdf, value, side="right") - 1)
        if sym < 0:
            sym = 0
        elif sym >= freq.shape[0]:
            sym = freq.shape[0] - 1

        self.high = self.low + (rng * int(cdf[sym + 1])) // TOTAL - 1
        self.low = self.low + (rng * int(cdf[sym])) // TOTAL
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
        return sym


def _selftest():
    """Round-trip random symbols under random pmfs to validate the coder."""
    rng = np.random.default_rng(0)
    enc = ArithmeticEncoder()
    syms, freqs = [], []
    for _ in range(5000):
        pmf = rng.dirichlet(np.ones(256) * rng.uniform(0.05, 2.0))
        freq = quantize_pmf(pmf)
        sym = int(rng.integers(0, 256))
        enc.encode_symbol(sym, freq)
        syms.append(sym)
        freqs.append(freq)
    stream = enc.finish()

    dec = ArithmeticDecoder(stream)
    ok = all(dec.decode_symbol(freqs[i]) == syms[i] for i in range(len(syms)))
    print(f"self-test round-trip: {'OK' if ok else 'FAIL'} "
          f"({len(syms)} symbols, {stream.shape[0]} bits, "
          f"{stream.shape[0]/len(syms):.3f} bits/symbol)")
    return ok


if __name__ == "__main__":
    assert _selftest()
