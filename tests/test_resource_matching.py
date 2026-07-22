import unittest

import numpy as np

from decoders import NoLDPCConfig, RSCSourceIterativeDecoder


class ResourceMatchingTest(unittest.TestCase):
    def test_repository_default_lengths(self):
        config = NoLDPCConfig()
        self.assertEqual(config.source_bits_per_symbol, 8)
        self.assertEqual(config.payload_length, 6272)
        self.assertEqual(config.source_coded_length, 7056)
        self.assertEqual(config.rsc_information_length, 7072)
        self.assertEqual(config.trellis_length, 7075)
        decoder = RSCSourceIterativeDecoder(config)
        self.assertEqual(decoder.rate_matcher.parity_symbols_kept, 5525)
        payload = np.zeros((1, config.payload_length), dtype=np.uint8)
        frame = decoder.encode(payload)
        self.assertEqual(frame.transmitted_bits.shape[-1], 12600)
        np.testing.assert_array_equal(
            frame.transmitted_bits[..., : config.trellis_length],
            frame.systematic_bits,
        )

    def test_no_spc_baseline_uses_explicit_eighteen_repetitions(self):
        config = NoLDPCConfig(use_spc=False, allow_parity_repetition=True)
        self.assertEqual(config.trellis_length, 6291)
        decoder = RSCSourceIterativeDecoder(config)
        self.assertEqual(decoder.rate_matcher.parity_symbols_kept, 6309)
        self.assertEqual(decoder.rate_matcher.repeated_parity_observations, 18)
        payload = np.zeros((1, config.payload_length), dtype=np.uint8)
        frame = decoder.encode(payload)
        self.assertEqual(frame.transmitted_bits.shape[-1], 12600)


if __name__ == "__main__":
    unittest.main()
