import unittest

import numpy as np

from experiments.waterfall_common import (
    PAYLOAD_BITS,
    TRANSMITTED_BITS,
    awgn_llr,
    interpolate_log_bler_knee,
    paired_batches,
    wilson_interval,
)


class WaterfallCommonTest(unittest.TestCase):
    def test_paired_plan_is_repeatable(self):
        bank = np.arange(10 * PAYLOAD_BITS, dtype=np.uint8).reshape(10, PAYLOAD_BITS) % 2
        first = list(paired_batches(
            bank, blocks=5, batch_size=3, esn0_db=-2.5, seed=7))
        second = list(paired_batches(
            bank, blocks=5, batch_size=3, esn0_db=-2.5, seed=7))
        self.assertEqual(len(first), 2)
        for left, right in zip(first, second):
            for left_value, right_value in zip(left, right):
                np.testing.assert_array_equal(left_value, right_value)

    def test_awgn_llr_sign_without_noise(self):
        bits = np.zeros((1, TRANSMITTED_BITS), dtype=np.uint8)
        bits[:, 1::2] = 1
        llr = awgn_llr(bits, np.zeros_like(bits, dtype=float), 0.0)
        self.assertTrue(np.all(llr[:, 0::2] < 0.0))
        self.assertTrue(np.all(llr[:, 1::2] > 0.0))

    def test_wilson_and_log_knee(self):
        self.assertAlmostEqual(wilson_interval(0, 100)[0], 0.0)
        rows = [
            {"esn0_db": 0.0, "true_payload_bler": 0.1},
            {"esn0_db": 1.0, "true_payload_bler": 0.01},
        ]
        self.assertAlmostEqual(interpolate_log_bler_knee(rows, 0.1), 0.0)
        self.assertAlmostEqual(interpolate_log_bler_knee(rows, 0.01), 1.0)


if __name__ == "__main__":
    unittest.main()
