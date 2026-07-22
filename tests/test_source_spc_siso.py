import unittest

import numpy as np

from decoders.source_spc_siso import (
    IndependentBitCategoricalProvider,
    SourceCategoricalSISO,
    SourceSPCSISO,
)


class FixedProvider:
    def __init__(self, probability):
        probability = np.asarray(probability, dtype=np.float64)
        self.log_probability = np.log(probability / probability.sum())

    def __call__(self, systematic_llr):
        shape = systematic_llr.shape[:-1] + (self.log_probability.size,)
        return np.broadcast_to(self.log_probability, shape)


class SourceSPCSISOTest(unittest.TestCase):
    def test_score_only_categorical_siso(self):
        probability = np.array([0.1, 0.2, 0.3, 0.4])
        decoder = SourceCategoricalSISO(2, FixedProvider(probability))
        cavity = np.array([[3.0, -2.0]])
        result = decoder(cavity)
        expected = np.array([[
            np.log((probability[2] + probability[3])
                   / (probability[0] + probability[1])),
            np.log((probability[1] + probability[3])
                   / (probability[0] + probability[2])),
        ]])
        np.testing.assert_allclose(result.posterior_llr, expected, atol=1e-12)
        np.testing.assert_allclose(result.extrinsic_llr, expected - cavity, atol=1e-12)

    def test_parity_likelihood_changes_marginal(self):
        decoder = SourceSPCSISO(1, FixedProvider([0.5, 0.5]))
        cavity = np.array([[0.0, 8.0]])
        result = decoder(cavity)
        self.assertGreater(result.posterior_llr[0, 0], 7.0)
        self.assertGreater(result.posterior_llr[0, 1], 7.0)

    def test_zero_parity_reduces_to_categorical(self):
        probability = np.array([0.05, 0.15, 0.30, 0.50])
        decoder = SourceSPCSISO(2, FixedProvider(probability))
        cavity = np.array([[4.0, -3.0, 0.0]])
        result = decoder(cavity)
        expected_bit0 = np.log((probability[2] + probability[3])
                               / (probability[0] + probability[1]))
        expected_bit1 = np.log((probability[1] + probability[3])
                               / (probability[0] + probability[2]))
        np.testing.assert_allclose(
            result.posterior_llr[0, :2], [expected_bit0, expected_bit1], atol=1e-12)

    def test_no_systematic_double_counting_and_extrinsic_definition(self):
        decoder = SourceSPCSISO(2, FixedProvider([0.1, 0.2, 0.3, 0.4]))
        first = np.array([[10.0, -10.0, 0.7]])
        second = np.array([[-4.0, 6.0, 0.7]])
        result_first = decoder(first)
        result_second = decoder(second)
        np.testing.assert_allclose(
            result_first.posterior_llr, result_second.posterior_llr, atol=1e-12)
        np.testing.assert_allclose(
            result_first.extrinsic_llr,
            result_first.posterior_llr - first,
            atol=1e-12,
        )

    def test_spc_only_matches_explicit_candidate_enumeration(self):
        provider = IndependentBitCategoricalProvider(3)
        decoder = SourceSPCSISO(3, provider)
        cavity = np.array([[0.4, -0.2, 1.1, -0.8]])
        result = decoder(cavity)
        self.assertTrue(np.all(np.isfinite(result.posterior_llr)))
        self.assertAlmostEqual(
            float(np.exp(result.log_symbol_posterior).sum()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
