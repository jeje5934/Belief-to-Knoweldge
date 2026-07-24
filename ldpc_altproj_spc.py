"""SPC-aware score source factor for the LDPC altproj experiment.

This is a standalone adapter; production ``decoder.py`` remains unchanged.
It transplants the no-LDPC branch's per-pixel SPC and exact 256-candidate
source-SISO equations into a Torch implementation suitable for the existing
TensorFlow/PyTorch bridge.

LLR convention: log P(bit=1)/P(bit=0), positive means bit 1.
Pixel layout: raster order, eight systematic bits MSB-first, then one XOR
parity bit, repeated for all 784 pixels.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import torch


BITS_PER_PIXEL = 8
PIXELS = 28 * 28
PAYLOAD_BITS = PIXELS * BITS_PER_PIXEL
SPC_BITS = PIXELS * (BITS_PER_PIXEL + 1)


def candidate_bits_numpy(bits_per_symbol=BITS_PER_PIXEL):
    values = np.arange(1 << bits_per_symbol, dtype=np.uint64)[:, None]
    shifts = np.arange(
        bits_per_symbol - 1, -1, -1, dtype=np.uint64
    )[None, :]
    return ((values >> shifts) & 1).astype(np.uint8)


def encode_spc_numpy(payload_bits, bits_per_symbol=BITS_PER_PIXEL):
    bits = np.asarray(payload_bits, dtype=np.uint8)
    if bits.shape[-1] % bits_per_symbol:
        raise ValueError("payload length must be divisible by bits_per_symbol")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("payload must be binary")
    symbols = bits.reshape(
        bits.shape[:-1] + (-1, bits_per_symbol)
    )
    parity = np.bitwise_xor.reduce(symbols, axis=-1, keepdims=True)
    return np.concatenate([symbols, parity], axis=-1).reshape(
        bits.shape[:-1] + (-1,)
    )


def decode_spc_systematic_numpy(spc_bits, bits_per_symbol=BITS_PER_PIXEL):
    bits = np.asarray(spc_bits, dtype=np.uint8)
    block = bits_per_symbol + 1
    if bits.shape[-1] % block:
        raise ValueError("SPC length must be divisible by M+1")
    shaped = bits.reshape(bits.shape[:-1] + (-1, block))
    return shaped[..., :bits_per_symbol].reshape(
        bits.shape[:-1] + (-1,)
    )


def parity_valid_numpy(spc_bits, bits_per_symbol=BITS_PER_PIXEL):
    bits = np.asarray(spc_bits, dtype=np.uint8)
    block = bits_per_symbol + 1
    shaped = bits.reshape(bits.shape[:-1] + (-1, block))
    expected = np.bitwise_xor.reduce(
        shaped[..., :bits_per_symbol], axis=-1
    )
    return expected == shaped[..., bits_per_symbol]


def spc_marginalize_torch(log_categorical, parity_llr, candidate_bits):
    """Exact score×SPC APP marginalization over all 256 pixel values."""
    bits = candidate_bits.bool()
    parity = (candidate_bits.sum(dim=-1) % 2).bool()
    log_categorical = log_categorical - torch.logsumexp(
        log_categorical, dim=-1, keepdim=True
    )
    log_p1 = torch.nn.functional.logsigmoid(parity_llr)[..., None]
    log_p0 = torch.nn.functional.logsigmoid(-parity_llr)[..., None]
    log_weight = log_categorical + torch.where(parity, log_p1, log_p0)
    log_posterior = log_weight - torch.logsumexp(
        log_weight, dim=-1, keepdim=True
    )
    apps = []
    for bit_index in range(candidate_bits.shape[-1]):
        mask = bits[:, bit_index]
        apps.append(
            torch.logsumexp(log_posterior[..., mask], dim=-1)
            - torch.logsumexp(log_posterior[..., ~mask], dim=-1)
        )
    systematic_app = torch.stack(apps, dim=-1)
    parity_app = (
        torch.logsumexp(log_posterior[..., parity], dim=-1)
        - torch.logsumexp(log_posterior[..., ~parity], dim=-1)
    )
    return systematic_app, parity_app, log_posterior


class SPCScoreExtrinsic(tf.keras.layers.Layer):
    """Drop-in altproj denoiser returning source/SPC posterior minus cavity."""

    def __init__(self, soft_denoiser, sigma_path=()):
        super().__init__(name="spc_score_extrinsic")
        self.inner = soft_denoiser
        self.path = tuple(float(value) for value in sigma_path)
        self.call_index = 0
        bits = candidate_bits_numpy()
        self._candidate_bits = torch.as_tensor(
            bits,
            dtype=torch.int64,
            device=self.inner._device,
        )
        lower = np.float32(1.0e-7)
        upper = np.float32(1.0 - 1.0e-7)
        one = np.float32(1.0)
        self._posterior_bounds = (
            float(np.log(lower / (one - lower))),
            float(np.log(upper / (one - upper))),
        )
        self.last_runtime = {}

    @property
    def prior_model(self):
        return self.inner.prior_model

    @property
    def sigma(self):
        return self.inner.sigma

    @sigma.setter
    def sigma(self, value):
        self.inner.sigma = value

    @property
    def sigma_post(self):
        return self.inner.sigma_post

    @sigma_post.setter
    def sigma_post(self, value):
        self.inner.sigma_post = value

    def reset_path(self, path):
        self.path = tuple(float(value) for value in path)
        self.call_index = 0

    def _sigma_tensor(self, sigma, batch, dtype, device):
        if self.path:
            if self.call_index >= len(self.path):
                raise RuntimeError(
                    "SPC source made more calls than the sigma path"
                )
            sigma = self.path[self.call_index]
            self.call_index += 1
        elif sigma is None:
            sigma = self.inner.sigma
        if isinstance(sigma, tf.Tensor):
            values = sigma.numpy()
        else:
            values = sigma
        output = torch.as_tensor(values, dtype=dtype, device=device).reshape(-1)
        if output.numel() == 1:
            output = output.expand(batch)
        if output.numel() != batch:
            raise ValueError("sigma must be scalar or have one value per block")
        return output

    def call(self, cavity_llr, sigma=None):
        import time

        started = time.perf_counter()
        cavity = torch.from_numpy(cavity_llr.numpy()).float().to(
            self.inner._device
        )
        if cavity.shape[-1] != SPC_BITS:
            raise ValueError(
                f"expected {SPC_BITS} SPC LLRs, got {cavity.shape[-1]}"
            )
        batch = cavity.shape[0]
        shaped = cavity.reshape(
            batch, PIXELS, BITS_PER_PIXEL + 1
        )
        systematic = shaped[..., :BITS_PER_PIXEL]
        parity_llr = shaped[..., BITS_PER_PIXEL]
        sigma_tensor = self._sigma_tensor(
            sigma, batch, cavity.dtype, cavity.device
        )

        with torch.no_grad():
            flat = systematic.reshape(batch, PAYLOAD_BITS)
            soft_image = self.prior_model.llr_to_soft_field(flat)
            denoised = self.prior_model.net(
                soft_image, sigma_tensor
            ).clamp(0.0, 1.0)
            pixel_mean = denoised.reshape(batch, PIXELS) * 255.0
            levels = self.prior_model.pixel_values.to(
                device=cavity.device, dtype=cavity.dtype
            )
            log_categorical = -0.5 * (
                (pixel_mean[..., None] - levels)
                / float(self.sigma_post)
            ) ** 2
            systematic_app, parity_app, _ = spc_marginalize_torch(
                log_categorical,
                parity_llr,
                self._candidate_bits,
            )
            posterior = torch.cat(
                [systematic_app, parity_app[..., None]], dim=-1
            ).reshape(batch, SPC_BITS)
            posterior = posterior.clamp(*self._posterior_bounds)
            extrinsic = posterior - cavity
        self.last_runtime = {
            "batch": int(batch),
            "seconds": float(time.perf_counter() - started),
            "sigma_min": float(sigma_tensor.min().item()),
            "sigma_max": float(sigma_tensor.max().item()),
            "mean_abs_extrinsic": float(extrinsic.abs().mean().item()),
        }
        return tf.constant(
            extrinsic.cpu().numpy(), dtype=cavity_llr.dtype
        )
