"""Eight-state recursive systematic convolutional encoder.

Polynomial convention
---------------------
Octal polynomials are expanded least-significant bit first as coefficients of
D^0, D^1, ..., D^memory. State bit 0 is the most recent recursive register
value (D^1), followed by older values. With feedback 13_o and feedforward 15_o:

    feedback coefficients   [1, 1, 0, 1]
    feedforward coefficients [1, 0, 1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np


def _poly_coefficients(polynomial: int, memory: int) -> np.ndarray:
    coefficients = np.array(
        [(polynomial >> degree) & 1 for degree in range(memory + 1)],
        dtype=np.uint8,
    )
    if coefficients[0] != 1 or coefficients[-1] != 1:
        raise ValueError("polynomial must include D^0 and D^memory")
    return coefficients


@dataclass
class RSCEncoded:
    systematic: np.ndarray
    parity: np.ndarray
    tail_bits: np.ndarray

    @property
    def trellis_length(self) -> int:
        return int(self.systematic.shape[-1])


@dataclass
class RSCTrellis:
    feedback_polynomial: int = 0o13
    feedforward_polynomial: int = 0o15
    memory: int = 3
    next_state: np.ndarray = field(init=False, repr=False)
    parity_output: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.memory <= 0:
            raise ValueError("memory must be positive")
        self.feedback_coefficients = _poly_coefficients(
            self.feedback_polynomial, self.memory)
        self.feedforward_coefficients = _poly_coefficients(
            self.feedforward_polynomial, self.memory)
        states = 1 << self.memory
        self.next_state = np.zeros((states, 2), dtype=np.int64)
        self.parity_output = np.zeros((states, 2), dtype=np.uint8)
        for state in range(states):
            registers = np.array(
                [(state >> index) & 1 for index in range(self.memory)],
                dtype=np.uint8,
            )
            for input_bit in (0, 1):
                recursive = input_bit ^ int(
                    np.bitwise_xor.reduce(
                        registers * self.feedback_coefficients[1:]))
                parity = int(np.bitwise_xor.reduce(
                    np.concatenate([
                        np.array([recursive], dtype=np.uint8), registers
                    ]) * self.feedforward_coefficients))
                next_registers = np.concatenate([
                    np.array([recursive], dtype=np.uint8), registers[:-1]
                ])
                next_value = int(sum(
                    int(bit) << index for index, bit in enumerate(next_registers)))
                self.next_state[state, input_bit] = next_value
                self.parity_output[state, input_bit] = parity

    @property
    def number_of_states(self) -> int:
        return 1 << self.memory

    def transition(self, state: int, input_bit: int) -> tuple[int, int]:
        return (
            int(self.next_state[state, input_bit]),
            int(self.parity_output[state, input_bit]),
        )

    def termination_bits(self, state: int) -> np.ndarray:
        """Find the unique short input sequence that returns the trellis to zero."""
        for candidate in product((0, 1), repeat=self.memory):
            trial_state = state
            for bit in candidate:
                trial_state = int(self.next_state[trial_state, bit])
            if trial_state == 0:
                return np.asarray(candidate, dtype=np.uint8)
        raise RuntimeError("no zero-state termination sequence exists")

    def encode(self, information_bits, terminate: bool = True) -> RSCEncoded:
        bits = np.asarray(information_bits, dtype=np.uint8)
        if bits.ndim == 0:
            raise ValueError("information_bits must have a bit axis")
        if np.any((bits != 0) & (bits != 1)):
            raise ValueError("information_bits must contain only 0 and 1")
        leading_shape = bits.shape[:-1]
        information_length = bits.shape[-1]
        flat = bits.reshape(-1, information_length)
        tail_length = self.memory if terminate else 0
        systematic = np.empty((flat.shape[0], information_length + tail_length), dtype=np.uint8)
        parity = np.empty_like(systematic)
        tails = np.empty((flat.shape[0], tail_length), dtype=np.uint8)

        for row_index, row in enumerate(flat):
            state = 0
            sequence = list(int(bit) for bit in row)
            if terminate:
                for bit in sequence:
                    state = int(self.next_state[state, bit])
                tail = self.termination_bits(state)
                tails[row_index] = tail
                sequence.extend(int(bit) for bit in tail)
                state = 0
            for time, bit in enumerate(sequence):
                systematic[row_index, time] = bit
                state, parity_bit = self.transition(state, bit)
                parity[row_index, time] = parity_bit
            if terminate and state != 0:
                raise AssertionError("RSC termination failed")

        output_shape = leading_shape + (information_length + tail_length,)
        tail_shape = leading_shape + (tail_length,)
        return RSCEncoded(
            systematic=systematic.reshape(output_shape),
            parity=parity.reshape(output_shape),
            tail_bits=tails.reshape(tail_shape),
        )
