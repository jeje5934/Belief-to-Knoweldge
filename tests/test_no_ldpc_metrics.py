import unittest

import numpy as np

from experiments.no_ldpc_metrics import image_metrics, payload_bits_to_uint8_images


class ImageMetricsTest(unittest.TestCase):
    def test_exact_round_trip_has_infinite_psnr_and_unit_ssim(self):
        image = np.array([[[0, 64], [128, 255]]], dtype=np.uint8)
        bits = np.unpackbits(image.reshape(1, -1), axis=-1)
        restored = payload_bits_to_uint8_images(bits, (2, 2), 8)
        np.testing.assert_array_equal(restored, image)
        metrics = image_metrics(bits, bits, (2, 2), 8, np.array([True]))
        self.assertEqual(metrics["psnr_db"], "inf")
        self.assertAlmostEqual(metrics["ssim_mean"], 1.0)
        self.assertIsNone(metrics["crc_failure_psnr_db"])

    def test_crc_failure_conditional_distortion(self):
        reference_image = np.zeros((1, 8, 8), dtype=np.uint8)
        estimate_image = reference_image.copy()
        estimate_image[0, 0, 0] = 255
        reference = np.unpackbits(reference_image.reshape(1, -1), axis=-1)
        estimate = np.unpackbits(estimate_image.reshape(1, -1), axis=-1)
        metrics = image_metrics(
            reference, estimate, (8, 8), 8, np.array([False]))
        self.assertEqual(metrics["crc_failure_blocks_for_distortion"], 1)
        self.assertIsInstance(metrics["crc_failure_psnr_db"], float)
        self.assertLess(metrics["ssim_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
