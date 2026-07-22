import unittest

import numpy as np

from coding.crc import CRC16_CCITT


class CRCTest(unittest.TestCase):
    def test_known_check_vector(self):
        bits = np.unpackbits(np.frombuffer(b"123456789", dtype=np.uint8))
        self.assertEqual(CRC16_CCITT.compute(bits), 0x29B1)

    def test_round_trip_and_one_bit_corruption(self):
        message = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 3, dtype=np.uint8)
        codeword = CRC16_CCITT.append(message)
        self.assertTrue(CRC16_CCITT.check(codeword))
        corrupted = codeword.copy()
        corrupted[7] ^= 1
        self.assertFalse(CRC16_CCITT.check(corrupted))

    def test_batched_round_trip(self):
        rng = np.random.default_rng(1)
        message = rng.integers(0, 2, size=(5, 31), dtype=np.uint8)
        self.assertTrue(np.all(CRC16_CCITT.check(CRC16_CCITT.append(message))))


if __name__ == "__main__":
    unittest.main()
