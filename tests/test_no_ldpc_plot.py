import unittest

from experiments.no_ldpc_plot import wilson_interval


class WilsonIntervalTest(unittest.TestCase):
    def test_zero_and_all_error_intervals_are_bounded(self):
        zero = wilson_interval(0, 8)
        all_error = wilson_interval(8, 8)
        self.assertEqual(zero[0], 0.0)
        self.assertLess(zero[1], 1.0)
        self.assertGreater(all_error[0], 0.0)
        self.assertEqual(all_error[1], 1.0)

    def test_midpoint_contains_estimate(self):
        lower, upper = wilson_interval(2, 10)
        self.assertLessEqual(lower, 0.2)
        self.assertGreaterEqual(upper, 0.2)


if __name__ == "__main__":
    unittest.main()
