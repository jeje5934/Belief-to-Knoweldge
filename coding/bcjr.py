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


def _combine_axis(values: np.ndarray, mode: str, axis: int) -> np.ndarray:
    """Vectorized max-log or stable log-sum-exp reduction."""

    if mode == "maxlog":
        return np.max(values, axis=axis)
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    with np.errstate(invalid="ignore", divide="ignore"):
        shifted = np.where(finite, values - maximum, NEG_INF)
        combined = maximum + np.log(
            np.exp(shifted).sum(axis=axis, keepdims=True))
    return np.where(finite, combined, NEG_INF).squeeze(axis)


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


def _decode_batch(
    systematic_llr: np.ndarray,
    parity_llr: np.ndarray,
    a_priori_llr: np.ndarray,
    trellis: RSCTrellis,
    mode: str,
    start_state: int,
    end_state: int | None,
) -> np.ndarray:
    """Decode a flat batch while retaining the exact scalar BCJR equations."""

    batch, length = systematic_llr.shape
    states = trellis.number_of_states
    next_state = np.asarray(trellis.next_state, dtype=np.int64)
    parity_output = np.asarray(trellis.parity_output, dtype=np.float64)

    predecessor_state = np.empty((states, 2), dtype=np.int64)
    predecessor_bit = np.empty((states, 2), dtype=np.int64)
    predecessor_count = np.zeros(states, dtype=np.int64)
    for state in range(states):
        for bit in (0, 1):
            successor = next_state[state, bit]
            slot = predecessor_count[successor]
            if slot >= 2:
                raise ValueError("binary RSC trellis has more than two predecessors")
            predecessor_state[successor, slot] = state
            predecessor_bit[successor, slot] = bit
            predecessor_count[successor] += 1
    if np.any(predecessor_count != 2):
        raise ValueError("every RSC state must have exactly two predecessors")

    input_sign = np.asarray([-1.0, 1.0], dtype=np.float64)
    parity_sign = 2.0 * parity_output - 1.0
    gamma = 0.5 * (
        (systematic_llr + a_priori_llr)[:, :, None, None]
        * input_sign[None, None, None, :]
        + parity_llr[:, :, None, None] * parity_sign[None, None, :, :]
    )

    alpha = np.full((batch, length + 1, states), NEG_INF, dtype=np.float64)
    beta = np.full((batch, length + 1, states), NEG_INF, dtype=np.float64)
    alpha[:, 0, start_state] = 0.0
    if end_state is None:
        beta[:, length, :] = 0.0
    else:
        beta[:, length, end_state] = 0.0

    for time in range(length):
        incoming = (
            alpha[:, time, predecessor_state]
            + gamma[:, time, predecessor_state, predecessor_bit]
        )
        alpha[:, time + 1, :] = _combine_axis(incoming, mode, axis=-1)
        normalizer = np.max(alpha[:, time + 1, :], axis=-1, keepdims=True)
        alpha[:, time + 1, :] -= np.where(
            np.isfinite(normalizer), normalizer, 0.0)

    for time in range(length - 1, -1, -1):
        outgoing = gamma[:, time, :, :] + beta[:, time + 1, next_state]
        beta[:, time, :] = _combine_axis(outgoing, mode, axis=-1)
        normalizer = np.max(beta[:, time, :], axis=-1, keepdims=True)
        beta[:, time, :] -= np.where(
            np.isfinite(normalizer), normalizer, 0.0)

    app = np.empty((batch, length), dtype=np.float64)
    for time in range(length):
        scores = []
        for bit in (0, 1):
            scores.append(_combine_axis(
                alpha[:, time, :]
                + gamma[:, time, :, bit]
                + beta[:, time + 1, next_state[:, bit]],
                mode,
                axis=-1,
            ))
        app[:, time] = scores[1] - scores[0]
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
    decoded = _decode_batch(
        flat_systematic,
        flat_parity,
        flat_prior,
        trellis,
        mode,
        start_state,
        end_state,
    ).reshape(leading_shape + (length,))
    return BCJRResult(app_llr=decoded, extrinsic_llr=decoded - prior)
