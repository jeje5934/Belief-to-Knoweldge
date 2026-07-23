import argparse
import unittest

import numpy as np

from decoders import NoLDPCConfig
from experiments.no_ldpc_schedule_search import (
    llr_diagnostics,
    parse_schedule,
    RecordingSourceSISO,
    summarize_rows,
    wilson_interval,
)


class NoLDPCScheduleSearchTest(unittest.TestCase):
    def test_selected_default_is_two_pass_exact_logmap(self):
        config = NoLDPCConfig()
        self.assertEqual(config.outer_iterations, 2)
        self.assertEqual(config.alpha_schedule, (0.1, 0.1))
        self.assertEqual(config.bcjr_mode, "logmap")

    def test_parse_schedule(self):
        self.assertEqual(parse_schedule("0.025,0.1,0.2"), (0.025, 0.1, 0.2))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_schedule("0.1,nan")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_schedule("-0.1")

    def test_wilson_interval_contains_observed_rate(self):
        lower, upper = wilson_interval(5, 100)
        self.assertLess(lower, 0.05)
        self.assertGreater(upper, 0.05)
        self.assertAlmostEqual(wilson_interval(0, 100)[0], 0.0)

    def test_llr_diagnostics_reports_real_clip_activity(self):
        recorder = RecordingSourceSISO(lambda _: None)
        recorder.cavity_llr = [np.array([[1.0, 2.0]]), np.array([[4.0, 5.0]])]
        recorder.extrinsic_llr = [
            np.array([[10.0, 40.0]]),
            np.array([[20.0, 60.0]]),
        ]
        diagnostics = llr_diagnostics(recorder, (0.1, 0.1), 30.0)
        self.assertEqual(diagnostics["clip_exceed_count"], 2)
        self.assertEqual(diagnostics["clip_value_count"], 4)
        np.testing.assert_allclose(
            diagnostics["final_bcjr_app_abs"]["max"], 8.0)

    def test_summary_uses_errors_before_runtime(self):
        rows = [
            {
                "schedule": "0.1", "alpha_schedule": [0.1], "bcjr_passes": 1,
                "blocks": 4, "true_payload_block_errors": 2, "bit_errors": 4,
                "true_payload_bler": 0.5, "crc_detected_bler": 0.5,
                "elapsed_seconds": 1.0, "source_siso_calls": 1,
                "score_model_calls": 1,
            },
            {
                "schedule": "0.05,0.1", "alpha_schedule": [0.05, 0.1],
                "bcjr_passes": 2, "blocks": 4,
                "true_payload_block_errors": 1, "bit_errors": 8,
                "true_payload_bler": 0.25, "crc_detected_bler": 0.25,
                "elapsed_seconds": 5.0, "source_siso_calls": 2,
                "score_model_calls": 2,
            },
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary[0]["schedule"], "0.05,0.1")
        self.assertEqual(summary[0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
