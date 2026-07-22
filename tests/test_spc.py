import itertools
import unittest

import numpy as np

from coding.spc import candidate_bits, decode_systematic, encode_spc, parity_is_valid


class SPCTest(unittest.TestCase):
    def test_every_candidate_and_systematic_round_trip(self):
        for bits_per_symbol in (1, 2, 3, 4):
            candidates = candidate_bits(bits_per_symbol)
            encoded = encode_spc(candidates, bits_per_symbol)
            self.assertTrue(np.all(parity_is_valid(encoded, bits_per_symbol)))
            np.testing.assert_array_equal(
                decode_systematic(encoded, bits_per_symbol), candidates)

    def test_minimum_distance_is_two(self):
        bits_per_symbol = 4
        codewords = encode_spc(candidate_bits(bits_per_symbol), bits_per_symbol)
        distances = [
            np.count_nonzero(codewords[i] != codewords[j])
            for i, j in itertools.combinations(range(codewords.shape[0]), 2)
        ]
        self.assertGreaterEqual(min(distances), 2)


if __name__ == "__main__":
    unittest.main()
