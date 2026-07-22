import unittest

import numpy as np

from coding.bcjr import bcjr_decode
from coding.rsc import RSCTrellis


class RSCTest(unittest.TestCase):
    def test_default_polynomial_convention(self):
        trellis = RSCTrellis()
        np.testing.assert_array_equal(
            trellis.feedback_coefficients, [1, 1, 0, 1])
        np.testing.assert_array_equal(
            trellis.feedforward_coefficients, [1, 0, 1, 1])

    def test_zero_state_termination(self):
        rng = np.random.default_rng(3)
        trellis = RSCTrellis()
        messages = rng.integers(0, 2, size=(32, 19), dtype=np.uint8)
        encoded = trellis.encode(messages, terminate=True)
        self.assertEqual(encoded.tail_bits.shape, (32, trellis.memory))
        for systematic in encoded.systematic:
            state = 0
            for bit in systematic:
                state = int(trellis.next_state[state, bit])
            self.assertEqual(state, 0)

    def test_noiseless_bcjr_recovery(self):
        rng = np.random.default_rng(5)
        trellis = RSCTrellis()
        messages = rng.integers(0, 2, size=(8, 23), dtype=np.uint8)
        encoded = trellis.encode(messages, terminate=True)
        result = bcjr_decode(
            (2.0 * encoded.systematic - 1.0) * 40.0,
            (2.0 * encoded.parity - 1.0) * 40.0,
            trellis,
        )
        decoded = (result.app_llr[..., : messages.shape[-1]] > 0.0).astype(np.uint8)
        np.testing.assert_array_equal(decoded, messages)


if __name__ == "__main__":
    unittest.main()
