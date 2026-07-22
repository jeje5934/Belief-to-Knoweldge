"""Deterministic random interleaver with an exact inverse."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np


@dataclass
class DeterministicInterleaver:
    length: int
    seed: int = 20260722
    permutation: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("interleaver length must be positive")
        if self.permutation is None:
            rng = np.random.default_rng(self.seed)
            permutation = rng.permutation(self.length)
        else:
            permutation = np.asarray(self.permutation, dtype=np.int64)
        if permutation.shape != (self.length,):
            raise ValueError("permutation has the wrong shape")
        if not np.array_equal(np.sort(permutation), np.arange(self.length)):
            raise ValueError("permutation must contain every index exactly once")
        self.permutation = permutation.astype(np.int64, copy=True)
        self.inverse = np.empty(self.length, dtype=np.int64)
        self.inverse[self.permutation] = np.arange(self.length, dtype=np.int64)

    def interleave(self, values) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-1] != self.length:
            raise ValueError("input length does not match interleaver")
        return array[..., self.permutation]

    def deinterleave(self, values) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-1] != self.length:
            raise ValueError("input length does not match interleaver")
        return array[..., self.inverse]

    def metadata(self) -> dict:
        digest = hashlib.sha256(self.permutation.astype("<i8").tobytes()).hexdigest()
        return {
            "kind": "numpy_default_rng_permutation",
            "length": self.length,
            "seed": self.seed,
            "permutation_sha256": digest,
        }
