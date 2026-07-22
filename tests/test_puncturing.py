import unittest

import numpy as np

from coding.puncturing import RateMatcher, uniform_parity_mask


class PuncturingTest(unittest.TestCase):
    def test_length_and_depuncturing(self):
        matcher = RateMatcher(trellis_length=31, target_length=49)
        systematic = np.arange(62).reshape(2, 31)
        parity = systematic + 1000
        transmitted = matcher.match(systematic, parity)
        self.assertEqual(transmitted.shape, (2, 49))
        systematic_llr, parity_llr = matcher.depuncture_llr(transmitted)
        np.testing.assert_array_equal(systematic_llr, systematic)
        np.testing.assert_array_equal(
            parity_llr[..., matcher.parity_mask], parity[..., matcher.parity_mask])
        self.assertTrue(np.all(parity_llr[..., ~matcher.parity_mask] == 0))

    def test_systematic_mandatory_and_mask_deterministic(self):
        with self.assertRaises(ValueError):
            uniform_parity_mask(10, 9)
        first = uniform_parity_mask(100, 163)
        second = uniform_parity_mask(100, 163)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(int(first.sum()), 63)

    def test_named_parity_repetition_and_llr_combining(self):
        matcher = RateMatcher(
            trellis_length=10,
            target_length=23,
            allow_parity_repetition=True,
        )
        systematic = np.zeros((1, 10), dtype=np.uint8)
        parity = np.arange(10, dtype=np.uint8)[None, :]
        transmitted = matcher.match(systematic, parity)
        self.assertEqual(transmitted.shape[-1], 23)
        self.assertEqual(matcher.repeated_parity_observations, 3)
        _, parity_llr = matcher.depuncture_llr(np.ones((1, 23)))
        self.assertEqual(int(np.sum(parity_llr)), 13)
        self.assertEqual(int(np.count_nonzero(parity_llr == 2.0)), 3)


if __name__ == "__main__":
    unittest.main()
