import unittest

import numpy as np

from experiments.source_residual_diagnostic import summarize_diagnostics
from experiments.source_residual_diagnostic import (
    mask_payload_bit_planes,
    summarize_echo_probe,
    summarize_paired_block_accounting,
    summarize_stage_transitions,
)


class SourceResidualDiagnosticTest(unittest.TestCase):
    def test_conditional_correction_and_contamination_counts(self):
        truth = np.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
        cavity_hat = np.array([[1, 0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
        source_hat = np.array([[0, 1, 0, 0, 1, 1, 0, 1]], dtype=np.uint8)
        cavity = np.array(
            [[0.05, -0.05, -2.0, -2.0, 2.0, 2.0, 5.0, 2.0]])
        source = np.where(source_hat, 1.0, -1.0)
        summary = summarize_diagnostics(
            truth,
            cavity,
            source,
            np.array([False]),
            source_shape=(1, 1),
            bits_per_symbol=8,
        )
        conditional = summary["conditional"]
        self.assertEqual(conditional["useful_corrections"], 1)
        self.assertEqual(conditional["harmful_contaminations"], 2)
        self.assertEqual(
            conditional["source_accuracy_where_cavity_wrong"], 1.0)
        self.assertEqual(
            conditional["cavity_abs_llr_at_useful_correction_opportunities"][
                "quantiles"]["median"],
            0.05,
        )
        self.assertAlmostEqual(
            conditional["cavity_abs_llr_at_harmful_contamination_attempts"][
                "quantiles"]["median"],
            2.525,
        )
        injection = conditional["local_alpha_clip_injection"]
        self.assertEqual(injection["successful_corrections"], 1)
        self.assertEqual(injection["successful_contaminations"], 1)
        self.assertEqual(
            injection["effective_correction_to_contamination_ratio"], 1.0)
        self.assertEqual(
            summary["bit_planes_msb_to_lsb"][
                "source_posterior_accuracy_where_cavity_wrong"],
            [1.0, None, None, None, None, None, None, None],
        )
        self.assertEqual(summary["failure_block_bit_errors"]["values"], [1])
        self.assertEqual(
            summary["failure_bit_plane_errors_msb_to_lsb"]["counts"],
            [1, 0, 0, 0, 0, 0, 0, 0],
        )

    def test_echo_probe_identifies_identity_response(self):
        truth = np.array([[0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
        cavity = np.where(truth, 3.0, -2.0)
        masked_posteriors = [
            mask_payload_bit_planes(cavity, (plane,))
            for plane in range(8)
        ]
        lower_masked = mask_payload_bit_planes(cavity, (6, 7))
        summary = summarize_echo_probe(
            truth,
            cavity,
            cavity,
            masked_posteriors,
            lower_masked,
        )
        self.assertEqual(
            summary["output_minus_input_accuracy_msb_to_lsb"],
            [0.0] * 8,
        )
        lsb = summary["causal_single_plane_ablation_msb_to_lsb"][7]
        self.assertAlmostEqual(
            lsb["posterior_response_gain_from_removed_input"], 1.0)
        self.assertAlmostEqual(
            lsb["src_ext_response_gain_after_input_subtraction"], 0.0)
        self.assertAlmostEqual(
            lsb["src_ext_response_rms_over_input"], 0.0)

    def test_stage_transitions_locate_new_errors(self):
        truth = np.array([[0, 0, 1, 1]], dtype=np.uint8)
        stages = {
            "first": np.array([[-1.0, 1.0, 1.0, 1.0]]),
            "second": np.array([[-1.0, -1.0, -1.0, 1.0]]),
            "third": np.array([[1.0, -1.0, 1.0, 1.0]]),
        }
        summary = summarize_stage_transitions(truth, stages)
        self.assertEqual(
            summary["stage_wrong_bits"],
            {"first": 1, "second": 1, "third": 1},
        )
        self.assertEqual(
            summary["adjacent_transitions"]["first_to_second"],
            {"corrections": 1, "new_errors": 1, "net_wrong_bit_change": 0},
        )
        self.assertEqual(
            summary["adjacent_transitions"]["second_to_third"],
            {"corrections": 1, "new_errors": 1, "net_wrong_bit_change": 0},
        )

    def test_paired_block_accounting_separates_broken_and_rescued(self):
        truth = np.zeros((4, 2), dtype=np.uint8)
        score_off = np.array([
            [0, 0],
            [1, 0],
            [0, 0],
            [1, 0],
        ], dtype=np.uint8)
        score_on = np.array([
            [1, 0],
            [0, 0],
            [0, 0],
            [1, 1],
        ], dtype=np.uint8)
        summary = summarize_paired_block_accounting(
            truth,
            score_off,
            np.array([True, False, True, False]),
            score_on,
            np.array([False, True, True, False]),
        )
        payload = summary["true_payload"]
        self.assertEqual(payload["successful_blocks_broken_by_score"], 1)
        self.assertEqual(payload["failed_blocks_rescued_by_score"], 1)
        self.assertEqual(payload["both_failed"], 1)
        self.assertEqual(payload["both_succeeded"], 1)
        self.assertEqual(payload["net_bler_error_count_change"], 0)
        self.assertTrue(summary["true_crc_failure_masks_identical_score_off"])
        self.assertTrue(summary["true_crc_failure_masks_identical_score_on"])


if __name__ == "__main__":
    unittest.main()
