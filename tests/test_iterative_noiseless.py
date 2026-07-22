import unittest

import numpy as np

from decoders import (
    IndependentBitCategoricalProvider,
    NoLDPCConfig,
    RSCSourceIterativeDecoder,
    SourceSPCSISO,
)


class IterativeNoiselessTest(unittest.TestCase):
    def setUp(self):
        self.config = NoLDPCConfig(
            source_shape=(2, 3),
            source_bits_per_symbol=3,
            target_length=67,
            outer_iterations=2,
            alpha_schedule=(0.15,),
        )
        self.decoder = RSCSourceIterativeDecoder(self.config)
        self.source_siso = SourceSPCSISO(
            3, IndependentBitCategoricalProvider(3))
        rng = np.random.default_rng(11)
        self.payload = rng.integers(
            0, 2, size=(5, self.config.payload_length), dtype=np.uint8)
        self.frame = self.decoder.encode(self.payload)
        self.noiseless_llr = (
            2.0 * self.frame.transmitted_bits.astype(np.float64) - 1.0) * 35.0

    def test_noiseless_true_payload_and_crc(self):
        result = self.decoder.decode(self.noiseless_llr, self.source_siso)
        np.testing.assert_array_equal(result.payload_bits, self.payload)
        self.assertTrue(np.all(result.crc_valid))
        self.assertFalse(np.any(~np.isfinite(result.payload_llr)))

    def test_alpha_zero_matches_rsc_only(self):
        zero_config = NoLDPCConfig(
            source_shape=self.config.source_shape,
            source_bits_per_symbol=self.config.source_bits_per_symbol,
            target_length=self.config.target_length,
            outer_iterations=2,
            alpha_schedule=(0.0,),
        )
        zero_decoder = RSCSourceIterativeDecoder(zero_config)
        with_source = zero_decoder.decode(self.noiseless_llr, self.source_siso)
        without_source = zero_decoder.decode(self.noiseless_llr, None)
        np.testing.assert_array_equal(with_source.payload_bits, without_source.payload_bits)
        np.testing.assert_allclose(
            with_source.channel_to_source_llr,
            without_source.channel_to_source_llr,
            atol=0.0,
        )
        self.assertEqual(with_source.source_calls, 0)

    def test_crc_positions_receive_zero_source_extrinsic(self):
        result = self.decoder.decode(self.noiseless_llr, self.source_siso)
        deinterleaved = self.decoder.interleaver.deinterleave(
            result.last_a_priori_information)
        self.assertTrue(np.all(deinterleaved[..., -self.config.crc.width :] == 0.0))

    def test_terminal_decision_uses_scaled_extrinsic(self):
        result = self.decoder.decode(self.noiseless_llr, self.source_siso)
        channel_source = result.channel_to_source_llr[
            ..., : self.config.source_coded_length]
        source_result = self.source_siso(channel_source)
        expected = channel_source + self.config.alpha_at(
            self.config.outer_iterations - 1) * np.clip(
                source_result.extrinsic_llr,
                -self.config.llr_clip,
                self.config.llr_clip,
            )
        expected_payload_llr = self.decoder._payload_llr_from_source(expected)
        np.testing.assert_allclose(result.payload_llr, expected_payload_llr)

    def test_one_outer_iteration_is_valid_and_finite(self):
        one_config = NoLDPCConfig(
            source_shape=self.config.source_shape,
            source_bits_per_symbol=self.config.source_bits_per_symbol,
            target_length=self.config.target_length,
            outer_iterations=1,
            alpha_schedule=(0.1,),
        )
        one_decoder = RSCSourceIterativeDecoder(one_config)
        result = one_decoder.decode(self.noiseless_llr, self.source_siso)
        self.assertEqual(result.bcjr_passes, 1)
        self.assertEqual(result.source_calls, 1)
        self.assertTrue(np.all(np.isfinite(result.payload_llr)))
        np.testing.assert_array_equal(result.payload_bits, self.payload)

    def test_no_spc_repetition_baseline_noiseless(self):
        config = NoLDPCConfig(
            source_shape=(2, 2),
            source_bits_per_symbol=2,
            target_length=57,
            outer_iterations=1,
            alpha_schedule=(0.0,),
            use_spc=False,
            allow_parity_repetition=True,
        )
        decoder = RSCSourceIterativeDecoder(config)
        rng = np.random.default_rng(15)
        payload = rng.integers(
            0, 2, size=(3, config.payload_length), dtype=np.uint8)
        frame = decoder.encode(payload)
        llr = (2.0 * frame.transmitted_bits.astype(np.float64) - 1.0) * 30.0
        result = decoder.decode(llr, None)
        np.testing.assert_array_equal(result.payload_bits, payload)
        self.assertTrue(np.all(result.crc_valid))


if __name__ == "__main__":
    unittest.main()
