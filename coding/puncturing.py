"""Deterministic parity-only rate matching for systematic RSC codewords."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np


def uniform_parity_mask(trellis_length: int, target_length: int) -> np.ndarray:
    """Keep all systematic symbols and uniformly select the parity budget."""
    if trellis_length <= 0:
        raise ValueError("trellis_length must be positive")
    if target_length < trellis_length:
        raise ValueError("target length is shorter than mandatory systematic symbols")
    if target_length > 2 * trellis_length:
        raise ValueError("target length exceeds the unpunctured rate-1/2 length")
    parity_budget = target_length - trellis_length
    mask = np.zeros(trellis_length, dtype=bool)
    if parity_budget:
        indices = np.floor(
            (np.arange(parity_budget, dtype=np.float64) + 0.5)
            * trellis_length / parity_budget
        ).astype(np.int64)
        if np.unique(indices).size != parity_budget:
            raise AssertionError("uniform parity selection produced duplicates")
        mask[indices] = True
    return mask


def uniform_parity_indices(
    trellis_length: int,
    target_length: int,
    *,
    allow_repetition: bool = False,
) -> np.ndarray:
    """Return transmitted parity indices, optionally with named repetition.

    When the parity budget exceeds one full parity stream, every parity symbol
    is sent once and the remaining observations repeat uniformly spaced parity
    positions. Repeated LLRs are summed by :meth:`RateMatcher.depuncture_llr`.
    """
    if trellis_length <= 0:
        raise ValueError("trellis_length must be positive")
    if target_length < trellis_length:
        raise ValueError("target length is shorter than mandatory systematic symbols")
    parity_budget = target_length - trellis_length
    if parity_budget <= trellis_length:
        return np.flatnonzero(
            uniform_parity_mask(trellis_length, target_length)).astype(np.int64)
    if not allow_repetition:
        raise ValueError("target length requires parity repetition")
    repetitions = parity_budget - trellis_length
    extra = np.floor(
        (np.arange(repetitions, dtype=np.float64) + 0.5)
        * trellis_length / repetitions
    ).astype(np.int64)
    return np.concatenate([
        np.arange(trellis_length, dtype=np.int64), extra
    ])


@dataclass
class RateMatcher:
    trellis_length: int
    target_length: int
    parity_mask: np.ndarray | None = field(default=None, repr=False)
    allow_parity_repetition: bool = False

    def __post_init__(self) -> None:
        if self.parity_mask is None:
            self.parity_indices = uniform_parity_indices(
                self.trellis_length,
                self.target_length,
                allow_repetition=self.allow_parity_repetition,
            )
            self.parity_mask = np.zeros(self.trellis_length, dtype=bool)
            self.parity_mask[self.parity_indices] = True
        else:
            self.parity_mask = np.asarray(self.parity_mask, dtype=bool)
            if self.parity_mask.shape != (self.trellis_length,):
                raise ValueError("parity mask has the wrong shape")
            expected = self.target_length - self.trellis_length
            if int(self.parity_mask.sum()) != expected:
                raise ValueError("parity mask does not match target length")
            self.parity_indices = np.flatnonzero(self.parity_mask).astype(np.int64)

    @property
    def parity_symbols_kept(self) -> int:
        return int(self.parity_indices.size)

    @property
    def unique_parity_symbols(self) -> int:
        return int(np.unique(self.parity_indices).size)

    @property
    def repeated_parity_observations(self) -> int:
        return self.parity_symbols_kept - self.unique_parity_symbols

    def match(self, systematic_bits, parity_bits) -> np.ndarray:
        systematic = np.asarray(systematic_bits)
        parity = np.asarray(parity_bits)
        if systematic.shape != parity.shape:
            raise ValueError("systematic and parity arrays must have equal shape")
        if systematic.shape[-1] != self.trellis_length:
            raise ValueError("trellis length mismatch")
        matched = np.concatenate([systematic, parity[..., self.parity_indices]], axis=-1)
        if matched.shape[-1] != self.target_length:
            raise AssertionError("rate matcher produced the wrong length")
        return matched

    def depuncture_llr(self, transmitted_llr) -> tuple[np.ndarray, np.ndarray]:
        llr = np.asarray(transmitted_llr, dtype=np.float64)
        if llr.shape[-1] != self.target_length:
            raise ValueError("transmitted LLR length mismatch")
        systematic = llr[..., : self.trellis_length]
        parity_kept = llr[..., self.trellis_length :]
        parity = np.zeros(llr.shape[:-1] + (self.trellis_length,), dtype=llr.dtype)
        flat_parity = parity.reshape(-1, self.trellis_length)
        flat_kept = parity_kept.reshape(-1, self.parity_symbols_kept)
        for transmitted_index, parity_index in enumerate(self.parity_indices):
            flat_parity[:, parity_index] += flat_kept[:, transmitted_index]
        return systematic, parity

    def metadata(self) -> dict:
        digest = hashlib.sha256(
            self.parity_indices.astype("<i8").tobytes()).hexdigest()
        return {
            "policy": (
                "keep_all_systematic_uniform_parity_with_repetition"
                if self.repeated_parity_observations
                else "keep_all_systematic_uniform_parity_puncturing"),
            "trellis_length": self.trellis_length,
            "target_length": self.target_length,
            "systematic_symbols": self.trellis_length,
            "parity_symbols_kept": self.parity_symbols_kept,
            "unique_parity_symbols": self.unique_parity_symbols,
            "repeated_parity_observations": self.repeated_parity_observations,
            "parity_indices_sha256": digest,
        }
