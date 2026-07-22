import itertools
import unittest

import numpy as np

from coding.bcjr import bcjr_decode
from coding.rsc import RSCTrellis


def reduce_scores(values, mode):
    values = np.asarray(values, dtype=np.float64)
    if mode == "maxlog":
        return float(values.max())
    maximum = values.max()
    return float(maximum + np.log(np.exp(values - maximum).sum()))


def brute_force_app(trellis, information_length, systematic_llr, parity_llr, prior, mode):
    paths = []
    for information in itertools.product((0, 1), repeat=information_length):
        encoded = trellis.encode(np.asarray(information, dtype=np.uint8), terminate=True)
        systematic = encoded.systematic
        parity = encoded.parity
        score = 0.5 * np.sum(
            (2.0 * systematic - 1.0) * (systematic_llr + prior)
            + (2.0 * parity - 1.0) * parity_llr)
        paths.append((systematic, score))
    app = []
    for time in range(systematic_llr.size):
        one = [score for bits, score in paths if bits[time] == 1]
        zero = [score for bits, score in paths if bits[time] == 0]
        app.append(reduce_scores(one, mode) - reduce_scores(zero, mode))
    return np.asarray(app)


class BCJRBruteForceTest(unittest.TestCase):
    def setUp(self):
        self.trellis = RSCTrellis()
        rng = np.random.default_rng(9)
        self.information_length = 3
        self.length = self.information_length + self.trellis.memory
        self.systematic_llr = rng.normal(size=self.length)
        self.parity_llr = rng.normal(size=self.length)
        self.prior = rng.normal(scale=0.3, size=self.length)

    def _compare(self, mode):
        expected = brute_force_app(
            self.trellis,
            self.information_length,
            self.systematic_llr,
            self.parity_llr,
            self.prior,
            mode,
        )
        actual = bcjr_decode(
            self.systematic_llr,
            self.parity_llr,
            self.trellis,
            self.prior,
            mode=mode,
        )
        np.testing.assert_allclose(actual.app_llr, expected, atol=1e-10)
        np.testing.assert_allclose(
            actual.extrinsic_llr, expected - self.prior, atol=1e-10)

    def test_logmap_matches_bruteforce(self):
        self._compare("logmap")

    def test_maxlog_matches_bruteforce(self):
        self._compare("maxlog")

    def test_punctured_parity_and_llr_sign(self):
        parity = self.parity_llr.copy()
        parity[1::2] = 0.0
        expected = brute_force_app(
            self.trellis,
            self.information_length,
            self.systematic_llr,
            parity,
            self.prior,
            "logmap",
        )
        actual = bcjr_decode(
            self.systematic_llr, parity, self.trellis, self.prior)
        np.testing.assert_allclose(actual.app_llr, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
