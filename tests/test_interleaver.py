import unittest

import numpy as np

from coding.interleaver import DeterministicInterleaver


class InterleaverTest(unittest.TestCase):
    def test_inverse_and_permutation(self):
        interleaver = DeterministicInterleaver(101, seed=20260722)
        values = np.arange(202).reshape(2, 101)
        restored = interleaver.deinterleave(interleaver.interleave(values))
        np.testing.assert_array_equal(restored, values)
        np.testing.assert_array_equal(
            np.sort(interleaver.permutation), np.arange(101))

    def test_deterministic(self):
        first = DeterministicInterleaver(64, seed=7)
        second = DeterministicInterleaver(64, seed=7)
        np.testing.assert_array_equal(first.permutation, second.permutation)
        self.assertEqual(
            first.metadata()["permutation_sha256"],
            second.metadata()["permutation_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
