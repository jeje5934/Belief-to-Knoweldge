"""Exact log-MAP and optional max-log-MAP BCJR for the RSC trellis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rsc import RSCTrellis


NEG_INF = -np.inf


def _combine(values, mode: str) -> float:
    if not values:
        return NEG_INF
    array = np.asarray(values, dtype=np.float64)
    if mode == "maxlog":
        return float(np.max(array))
    maximum = float(np.max(array))
    if np.isneginf(maximum):
        return NEG_INF
    return maximum + float(np.log(np.exp(array - maximum).sum()))


@dataclass
class BCJRResult:
    app_llr: np.ndarray
    extrinsic_llr: np.ndarray


def _decode_one(
    systematic_llr: np.ndarray,
    parity_llr: np.ndarray,
    a_priori_llr: np.ndarray,
    trellis: RSCTrellis,
    mode: str,
    start_state: int,
    end_state: int | None,
) -> np.ndarray:
    length = systematic_llr.size
    states = trellis.number_of_states
    alpha = np.full((length + 1, states), NEG_INF, dtype=np.float64)
    beta = np.full((length + 1, states), NEG_INF, dtype=np.float64)
    alpha[0, start_state] = 0.0
    if end_state is None:
        beta[length, :] = 0.0
    else:
        beta[length, end_state] = 0.0

    def gamma(time: int, state: int, bit: int) -> float:
        parity = int(trellis.parity_output[state, bit])
        input_sign = 1.0 if bit else -1.0
        parity_sign = 1.0 if parity else -1.0
        return 0.5 * (
            input_sign * (systematic_llr[time] + a_priori_llr[time])
            + parity_sign * parity_llr[time]
        )

    for time in range(length):
        candidates = [[] for _ in range(states)]
        for state in range(states):
            if np.isneginf(alpha[time, state]):
                continue
            for bit in (0, 1):
                next_state = int(trellis.next_state[state, bit])
                candidates[next_state].append(
                    alpha[time, state] + gamma(time, state, bit))
        for state in range(states):
            alpha[time + 1, state] = _combine(candidates[state], mode)
        normalizer = np.max(alpha[time + 1])
        if np.isfinite(normalizer):
            alpha[time + 1] -= normalizer

    for time in range(length - 1, -1, -1):
        for state in range(states):
            values = []
            for bit in (0, 1):
                next_state = int(trellis.next_state[state, bit])
                values.append(
                    gamma(time, state, bit) + beta[time + 1, next_state])
            beta[time, state] = _combine(values, mode)
        normalizer = np.max(beta[time])
        if np.isfinite(normalizer):
            beta[time] -= normalizer

    app = np.empty(length, dtype=np.float64)
    for time in range(length):
        scores = {0: [], 1: []}
        for state in range(states):
            if np.isneginf(alpha[time, state]):
                continue
            for bit in (0, 1):
                next_state = int(trellis.next_state[state, bit])
                scores[bit].append(
                    alpha[time, state]
                    + gamma(time, state, bit)
                    + beta[time + 1, next_state]
                )
        app[time] = _combine(scores[1], mode) - _combine(scores[0], mode)
    return app


def bcjr_decode(
    systematic_llr,
    parity_llr,
    trellis: RSCTrellis,
    a_priori_llr=None,
    *,
    mode: str = "logmap",
    start_state: int = 0,
    end_state: int | None = 0,
) -> BCJRResult:
    """Decode LLR=log(P(bit=1)/P(bit=0)); punctured parity LLRs are zero.

    The returned channel-to-source extrinsic is APP minus only the incoming
    source a-priori. The systematic-channel term is intentionally retained.
    """
    if mode not in ("logmap", "maxlog"):
        raise ValueError("mode must be 'logmap' or 'maxlog'")
    systematic = np.asarray(systematic_llr, dtype=np.float64)
    parity = np.asarray(parity_llr, dtype=np.float64)
    if systematic.shape != parity.shape or systematic.ndim == 0:
        raise ValueError("systematic and parity LLR arrays must have equal shape")
    if a_priori_llr is None:
        prior = np.zeros_like(systematic)
    else:
        prior = np.asarray(a_priori_llr, dtype=np.float64)
        if prior.shape != systematic.shape:
            raise ValueError("a-priori LLR shape mismatch")
    length = systematic.shape[-1]
    leading_shape = systematic.shape[:-1]
    flat_systematic = systematic.reshape(-1, length)
    flat_parity = parity.reshape(-1, length)
    flat_prior = prior.reshape(-1, length)
    decoded = np.stack([
        _decode_one(s, p, a, trellis, mode, start_state, end_state)
        for s, p, a in zip(flat_systematic, flat_parity, flat_prior)
    ], axis=0).reshape(leading_shape + (length,))
    return BCJRResult(app_llr=decoded, extrinsic_llr=decoded - prior)
