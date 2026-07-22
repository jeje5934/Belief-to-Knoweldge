"""Composite score-prior/SPC source SISO decoder.

The categorical provider is responsible for consuming the systematic-bit
cavity exactly once. This module then adds only the parity-bit likelihood,
normalizes in the log domain, and marginalizes every systematic and parity bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from coding.spc import candidate_bits


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    result = safe_maximum + np.log(
        np.sum(np.exp(values - safe_maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _log_sigmoid(value: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -value)


class CategoricalProvider(Protocol):
    def __call__(self, systematic_llr: np.ndarray) -> np.ndarray:
        """Return normalized log P(symbol) with shape [..., symbols, 2**M]."""


class IndependentBitCategoricalProvider:
    """SPC-only baseline: construct P(symbol) from systematic cavity LLRs."""

    def __init__(self, bits_per_symbol: int):
        self.bits_per_symbol = bits_per_symbol
        self.bits = candidate_bits(bits_per_symbol).astype(np.float64)

    def __call__(self, systematic_llr: np.ndarray) -> np.ndarray:
        llr = np.asarray(systematic_llr, dtype=np.float64)
        if llr.shape[-1] != self.bits_per_symbol:
            raise ValueError("systematic LLR bit depth mismatch")
        log_p1 = _log_sigmoid(llr)
        log_p0 = _log_sigmoid(-llr)
        log_probability = np.sum(
            log_p1[..., None, :] * self.bits
            + log_p0[..., None, :] * (1.0 - self.bits),
            axis=-1,
        )
        return log_probability - _logsumexp(log_probability, axis=-1)[..., None]


class ScorePriorCategoricalProvider:
    """Adapter from the existing PyTorch score prior to categorical log PMFs."""

    def __init__(self, prior_model, sigma: float = 0.3, sigma_post: float | None = None):
        self.prior_model = prior_model
        self.sigma = float(sigma)
        self.sigma_post = (
            float(prior_model.sigma_post) if sigma_post is None else float(sigma_post))

    def __call__(self, systematic_llr: np.ndarray) -> np.ndarray:
        import torch

        llr = np.asarray(systematic_llr, dtype=np.float32)
        if llr.ndim < 2:
            raise ValueError("systematic LLRs need symbol and bit axes")
        symbol_count, bits_per_symbol = llr.shape[-2:]
        if symbol_count != self.prior_model.n_pixels:
            raise ValueError("score model pixel count does not match source symbols")
        if bits_per_symbol != self.prior_model.bpp:
            raise ValueError("score model bit depth does not match source symbols")
        leading_shape = llr.shape[:-2]
        batch = int(np.prod(leading_shape)) if leading_shape else 1
        device = next(self.prior_model.parameters()).device
        tensor = torch.as_tensor(
            llr.reshape(batch, symbol_count * bits_per_symbol),
            dtype=torch.float32,
            device=device,
        )
        sigma = torch.full((batch,), self.sigma, dtype=tensor.dtype, device=device)
        with torch.no_grad():
            soft_image = self.prior_model.llr_to_soft_field(tensor)
            denoised = self.prior_model.net(soft_image, sigma).clamp(0.0, 1.0)
            maximum_value = float((1 << bits_per_symbol) - 1)
            pixel_mean = denoised.reshape(batch, symbol_count) * maximum_value
            levels = self.prior_model.pixel_values.to(device=device, dtype=tensor.dtype)
            log_probability = -0.5 * (
                (pixel_mean[..., None] - levels) / self.sigma_post
            ) ** 2
            log_probability = log_probability - torch.logsumexp(
                log_probability, dim=-1, keepdim=True)
        output = log_probability.cpu().numpy()
        return output.reshape(leading_shape + (symbol_count, 1 << bits_per_symbol))


@dataclass
class SourceSISOResult:
    posterior_llr: np.ndarray
    extrinsic_llr: np.ndarray
    log_symbol_posterior: np.ndarray


class SourceCategoricalSISO:
    """Score-only source SISO for an uncoded M-bit source-symbol stream."""

    def __init__(self, bits_per_symbol: int, categorical_provider: CategoricalProvider):
        if bits_per_symbol <= 0:
            raise ValueError("bits_per_symbol must be positive")
        self.bits_per_symbol = bits_per_symbol
        self.categorical_provider = categorical_provider
        self.bits = candidate_bits(bits_per_symbol).astype(bool)

    def __call__(self, cavity_llr) -> SourceSISOResult:
        cavity = np.asarray(cavity_llr, dtype=np.float64)
        if cavity.ndim == 0 or cavity.shape[-1] % self.bits_per_symbol:
            raise ValueError("cavity length must be divisible by M")
        symbol_count = cavity.shape[-1] // self.bits_per_symbol
        systematic = cavity.reshape(
            cavity.shape[:-1] + (symbol_count, self.bits_per_symbol))
        log_posterior = np.asarray(
            self.categorical_provider(systematic), dtype=np.float64)
        expected = cavity.shape[:-1] + (symbol_count, 1 << self.bits_per_symbol)
        if log_posterior.shape != expected:
            raise ValueError(
                f"categorical provider returned {log_posterior.shape}, expected {expected}")
        log_posterior = log_posterior - _logsumexp(
            log_posterior, axis=-1)[..., None]
        bit_llrs = []
        for bit_index in range(self.bits_per_symbol):
            mask = self.bits[:, bit_index]
            bit_llrs.append(
                _logsumexp(log_posterior[..., mask], axis=-1)
                - _logsumexp(log_posterior[..., ~mask], axis=-1))
        posterior = np.stack(bit_llrs, axis=-1).reshape(cavity.shape)
        return SourceSISOResult(
            posterior_llr=posterior,
            extrinsic_llr=posterior - cavity,
            log_symbol_posterior=log_posterior,
        )


class SourceSPCSISO:
    """Exact SPC APP marginalization conditional on a categorical source PMF."""

    def __init__(self, bits_per_symbol: int, categorical_provider: CategoricalProvider):
        if bits_per_symbol <= 0:
            raise ValueError("bits_per_symbol must be positive")
        self.bits_per_symbol = bits_per_symbol
        self.categorical_provider = categorical_provider
        self.bits = candidate_bits(bits_per_symbol).astype(bool)
        self.parity = (self.bits.sum(axis=-1) % 2).astype(bool)

    def __call__(self, cavity_llr) -> SourceSISOResult:
        cavity = np.asarray(cavity_llr, dtype=np.float64)
        block_length = self.bits_per_symbol + 1
        if cavity.ndim == 0 or cavity.shape[-1] % block_length:
            raise ValueError("cavity length must be divisible by M+1")
        symbol_count = cavity.shape[-1] // block_length
        shaped = cavity.reshape(cavity.shape[:-1] + (symbol_count, block_length))
        systematic = shaped[..., : self.bits_per_symbol]
        parity_llr = shaped[..., self.bits_per_symbol]

        log_categorical = np.asarray(
            self.categorical_provider(systematic), dtype=np.float64)
        expected_shape = cavity.shape[:-1] + (symbol_count, 1 << self.bits_per_symbol)
        if log_categorical.shape != expected_shape:
            raise ValueError(
                f"categorical provider returned {log_categorical.shape}, "
                f"expected {expected_shape}")
        log_categorical = log_categorical - _logsumexp(
            log_categorical, axis=-1)[..., None]

        log_parity_one = _log_sigmoid(parity_llr)[..., None]
        log_parity_zero = _log_sigmoid(-parity_llr)[..., None]
        log_weight = log_categorical + np.where(
            self.parity, log_parity_one, log_parity_zero)
        log_posterior = log_weight - _logsumexp(log_weight, axis=-1)[..., None]

        systematic_llrs = []
        for bit_index in range(self.bits_per_symbol):
            mask = self.bits[:, bit_index]
            systematic_llrs.append(
                _logsumexp(log_posterior[..., mask], axis=-1)
                - _logsumexp(log_posterior[..., ~mask], axis=-1))
        systematic_app = np.stack(systematic_llrs, axis=-1)
        parity_app = (
            _logsumexp(log_posterior[..., self.parity], axis=-1)
            - _logsumexp(log_posterior[..., ~self.parity], axis=-1))
        posterior = np.concatenate(
            [systematic_app, parity_app[..., None]], axis=-1).reshape(cavity.shape)
        return SourceSISOResult(
            posterior_llr=posterior,
            extrinsic_llr=posterior - cavity,
            log_symbol_posterior=log_posterior,
        )
