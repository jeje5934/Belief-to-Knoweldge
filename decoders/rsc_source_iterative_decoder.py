"""RSC/BCJR iterative source-channel decoder without LDPC."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod

import numpy as np

from coding.bcjr import bcjr_decode
from coding.crc import CRC, CRC16_CCITT
from coding.interleaver import DeterministicInterleaver
from coding.puncturing import RateMatcher
from coding.rsc import RSCTrellis
from coding.spc import decode_systematic, encode_spc


@dataclass(frozen=True)
class NoLDPCConfig:
    source_shape: tuple[int, ...] = (28, 28)
    source_bits_per_symbol: int = 8
    crc: CRC = CRC16_CCITT
    target_length: int = 12600
    feedback_polynomial: int = 0o13
    feedforward_polynomial: int = 0o15
    memory: int = 3
    termination_mode: str = "zero"
    bcjr_mode: str = "logmap"
    interleaver_seed: int = 20260722
    outer_iterations: int = 4
    alpha_schedule: tuple[float, ...] = (0.1,)
    llr_clip: float = 30.0
    sigma: float = 0.3
    sigma_post: float = 3.0
    channel_model: str = "awgn_bpsk"
    imperfect_csi_variance: float = 0.0
    use_spc: bool = True
    allow_parity_repetition: bool = False

    def __post_init__(self) -> None:
        if not self.source_shape or any(value <= 0 for value in self.source_shape):
            raise ValueError("source_shape must contain positive dimensions")
        if self.source_bits_per_symbol <= 0:
            raise ValueError("source_bits_per_symbol must be positive")
        if self.termination_mode != "zero":
            raise ValueError("the first implementation supports zero termination only")
        if self.bcjr_mode not in ("logmap", "maxlog"):
            raise ValueError("bcjr_mode must be logmap or maxlog")
        if self.outer_iterations <= 0:
            raise ValueError("outer_iterations must be positive")
        if not self.alpha_schedule:
            raise ValueError("alpha_schedule cannot be empty")
        if self.llr_clip <= 0:
            raise ValueError("llr_clip must be positive")

    @property
    def source_symbols(self) -> int:
        return int(prod(self.source_shape))

    @property
    def payload_length(self) -> int:
        return self.source_symbols * self.source_bits_per_symbol

    @property
    def source_coded_length(self) -> int:
        if not self.use_spc:
            return self.payload_length
        return self.payload_length + self.source_symbols

    @property
    def rsc_information_length(self) -> int:
        return self.source_coded_length + self.crc.width

    @property
    def trellis_length(self) -> int:
        return self.rsc_information_length + self.memory

    def alpha_at(self, iteration: int) -> float:
        return float(self.alpha_schedule[min(iteration, len(self.alpha_schedule) - 1)])


@dataclass
class EncodedFrame:
    transmitted_bits: np.ndarray
    information_bits: np.ndarray
    systematic_bits: np.ndarray
    parity_bits: np.ndarray
    metadata: dict


@dataclass
class DecodeResult:
    payload_bits: np.ndarray
    payload_llr: np.ndarray
    crc_bits: np.ndarray
    crc_llr: np.ndarray
    crc_valid: np.ndarray | bool
    channel_to_source_llr: np.ndarray
    last_a_priori_information: np.ndarray
    source_calls: int
    bcjr_passes: int


class RSCSourceIterativeDecoder:
    def __init__(self, config: NoLDPCConfig):
        self.config = config
        self.trellis = RSCTrellis(
            feedback_polynomial=config.feedback_polynomial,
            feedforward_polynomial=config.feedforward_polynomial,
            memory=config.memory,
        )
        self.interleaver = DeterministicInterleaver(
            config.rsc_information_length, config.interleaver_seed)
        self.rate_matcher = RateMatcher(
            config.trellis_length,
            config.target_length,
            allow_parity_repetition=config.allow_parity_repetition,
        )

    def encode(self, payload_bits) -> EncodedFrame:
        payload = np.asarray(payload_bits, dtype=np.uint8)
        if payload.ndim == 0 or payload.shape[-1] != self.config.payload_length:
            raise ValueError("payload length does not match configuration")
        if np.any((payload != 0) & (payload != 1)):
            raise ValueError("payload must contain only 0 and 1")
        source_coded = (
            encode_spc(payload, self.config.source_bits_per_symbol)
            if self.config.use_spc else payload.copy())
        crc_bits = self.config.crc.append(payload)[..., -self.config.crc.width :]
        information = np.concatenate([source_coded, crc_bits], axis=-1)
        interleaved = self.interleaver.interleave(information)
        encoded = self.trellis.encode(interleaved, terminate=True)
        transmitted = self.rate_matcher.match(encoded.systematic, encoded.parity)
        metadata = {
            "payload_length": self.config.payload_length,
            "source_coded_length": self.config.source_coded_length,
            "crc_length": self.config.crc.width,
            "rsc_information_length": self.config.rsc_information_length,
            "trellis_length": self.config.trellis_length,
            "transmitted_length": transmitted.shape[-1],
            "interleaver": self.interleaver.metadata(),
            "rate_matching": self.rate_matcher.metadata(),
        }
        return EncodedFrame(
            transmitted_bits=transmitted,
            information_bits=information,
            systematic_bits=encoded.systematic,
            parity_bits=encoded.parity,
            metadata=metadata,
        )

    def _payload_from_source_llr(self, source_llr: np.ndarray) -> np.ndarray:
        hard = (source_llr > 0.0).astype(np.uint8)
        if self.config.use_spc:
            return decode_systematic(hard, self.config.source_bits_per_symbol)
        return hard

    def _payload_llr_from_source(self, source_llr: np.ndarray) -> np.ndarray:
        if not self.config.use_spc:
            return source_llr
        block = self.config.source_bits_per_symbol + 1
        symbol_count = self.config.source_symbols
        shaped = source_llr.reshape(
            source_llr.shape[:-1] + (symbol_count, block))
        return shaped[..., : self.config.source_bits_per_symbol].reshape(
            source_llr.shape[:-1] + (self.config.payload_length,))

    def decode(
        self,
        transmitted_llr,
        source_siso=None,
    ) -> DecodeResult:
        systematic_llr, parity_llr = self.rate_matcher.depuncture_llr(transmitted_llr)
        leading_shape = systematic_llr.shape[:-1]
        prior_information = np.zeros(
            leading_shape + (self.config.rsc_information_length,), dtype=np.float64)
        zero_tail = np.zeros(leading_shape + (self.config.memory,), dtype=np.float64)
        source_calls = 0
        payload_llr = None
        crc_llr = None
        channel_to_source = None
        source_feedback_enabled = (
            source_siso is not None
            and any(abs(self.config.alpha_at(i)) > 0.0
                    for i in range(self.config.outer_iterations))
        )

        for iteration in range(self.config.outer_iterations):
            full_prior = np.concatenate([prior_information, zero_tail], axis=-1)
            channel_result = bcjr_decode(
                systematic_llr,
                parity_llr,
                self.trellis,
                a_priori_llr=full_prior,
                mode=self.config.bcjr_mode,
                start_state=0,
                end_state=0,
            )
            interleaved_message = channel_result.extrinsic_llr[
                ..., : self.config.rsc_information_length]
            channel_to_source = self.interleaver.deinterleave(interleaved_message)
            source_message = channel_to_source[..., : self.config.source_coded_length]
            crc_llr = channel_to_source[..., self.config.source_coded_length :]
            alpha = self.config.alpha_at(iteration)

            if source_feedback_enabled and alpha != 0.0:
                source_result = source_siso(source_message)
                source_calls += 1
                clipped = np.clip(
                    source_result.extrinsic_llr,
                    -self.config.llr_clip,
                    self.config.llr_clip,
                )
                scaled_extrinsic = alpha * clipped
                # The final decision must obey the same damping/scaling as the
                # feedback message. Using the raw source posterior here would
                # silently make the terminal decision alpha=1 even when the
                # iterative update uses a much smaller alpha.
                payload_llr = self._payload_llr_from_source(
                    source_message + scaled_extrinsic)
                source_extrinsic = np.concatenate(
                    [scaled_extrinsic, np.zeros_like(crc_llr)], axis=-1)
                prior_information = self.interleaver.interleave(source_extrinsic)
            else:
                payload_llr = self._payload_llr_from_source(source_message)
                prior_information = np.zeros_like(prior_information)

        if payload_llr is None or crc_llr is None or channel_to_source is None:
            raise AssertionError("decoder completed no iterations")
        payload = (payload_llr > 0.0).astype(np.uint8)
        crc_bits = (crc_llr > 0.0).astype(np.uint8)
        crc_valid = self.config.crc.check_parts(payload, crc_bits)
        return DecodeResult(
            payload_bits=payload,
            payload_llr=payload_llr,
            crc_bits=crc_bits,
            crc_llr=crc_llr,
            crc_valid=crc_valid,
            channel_to_source_llr=channel_to_source,
            last_a_priori_information=prior_information,
            source_calls=source_calls,
            bcjr_passes=self.config.outer_iterations,
        )

    @staticmethod
    def metrics(payload_true, decoded: DecodeResult) -> dict:
        truth = np.asarray(payload_true, dtype=np.uint8)
        if truth.shape != decoded.payload_bits.shape:
            raise ValueError("truth and decoded payload shapes differ")
        bit_errors = np.sum(truth != decoded.payload_bits, axis=-1)
        block_error = bit_errors > 0
        crc_valid = np.asarray(decoded.crc_valid, dtype=bool)
        return {
            "blocks": int(block_error.size),
            "bit_errors": int(bit_errors.sum()),
            "ber": float(bit_errors.sum() / truth.size),
            "true_payload_block_errors": int(block_error.sum()),
            "true_payload_bler": float(block_error.mean()),
            "crc_detected_block_errors": int((~crc_valid).sum()),
            "crc_detected_bler": float((~crc_valid).mean()),
            "undetected_errors": int((crc_valid & block_error).sum()),
        }

    def with_alpha(self, alpha: float, outer_iterations: int | None = None):
        config = replace(
            self.config,
            alpha_schedule=(float(alpha),),
            outer_iterations=(
                self.config.outer_iterations
                if outer_iterations is None else int(outer_iterations)),
        )
        return type(self)(config)
