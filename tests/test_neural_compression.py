import unittest
from pathlib import Path

import numpy as np
import torch

from compression_baseline.arithmetic_coder import (
    TOTAL,
    ArithmeticDecoder,
    ArithmeticEncoder,
    quantize_pmf,
)
from compression_baseline.data import to_model_input
from compression_baseline.codec import _step_pmf, decode_batch, encode_batch, load_model
from compression_baseline.pixelcnn import GatedPixelCNN


class UniformAutoregressiveModel(torch.nn.Module):
    def forward(self, canvas):
        return torch.zeros(
            canvas.shape[0], 256, canvas.shape[2], canvas.shape[3],
            device=canvas.device,
        )


class BatchSensitiveAutoregressiveModel(torch.nn.Module):
    def forward(self, canvas):
        logits = torch.zeros(
            canvas.shape[0], 256, canvas.shape[2], canvas.shape[3],
            device=canvas.device,
        )
        logits[:, 0] = float(canvas.shape[0])
        return logits


class NeuralCompressionTest(unittest.TestCase):
    def test_probability_quantization_is_deterministic_and_positive(self):
        pmf = np.linspace(0.0, 1.0, 256, dtype=np.float64)
        first = quantize_pmf(pmf)
        second = quantize_pmf(pmf)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(int(first.sum()), TOTAL)
        self.assertGreaterEqual(int(first.min()), 1)

    def test_arithmetic_coder_round_trip_with_changing_pmfs(self):
        rng = np.random.default_rng(11)
        symbols = rng.integers(0, 256, size=200)
        frequencies = [quantize_pmf(rng.random(256)) for _ in symbols]
        encoder = ArithmeticEncoder()
        for symbol, freq in zip(symbols, frequencies):
            encoder.encode_symbol(int(symbol), freq)
        stream = encoder.finish()

        decoder = ArithmeticDecoder(stream)
        decoded = [decoder.decode_symbol(freq) for freq in frequencies]
        np.testing.assert_array_equal(decoded, symbols)

    def test_checkpoint_architecture_and_causal_shape(self):
        model = GatedPixelCNN(n_channels=4, n_layers=2, k=3)
        image = torch.zeros(2, 1, 28, 28, dtype=torch.uint8)
        logits = model(to_model_input(image))
        self.assertEqual(tuple(logits.shape), (2, 256, 28, 28))
        self.assertTrue(torch.isfinite(logits).all())

    def test_codec_round_trip_without_external_checkpoint(self):
        rng = np.random.default_rng(17)
        image = torch.from_numpy(
            rng.integers(0, 256, size=(1, 1, 28, 28), dtype=np.uint8)
        )
        model = UniformAutoregressiveModel().eval()
        streams, lengths = encode_batch(model, image, "cpu")
        decoded = decode_batch(model, streams, "cpu")
        self.assertEqual(lengths, [6274])
        torch.testing.assert_close(decoded, image, rtol=0, atol=0)

    def test_probability_evaluation_is_batch_partition_invariant(self):
        model = BatchSensitiveAutoregressiveModel().eval()
        canvas = torch.zeros(3, 1, 28, 28, dtype=torch.uint8)
        together = _step_pmf(model, canvas, 0, 0)
        separately = np.concatenate(
            [_step_pmf(model, canvas[index : index + 1], 0, 0)
             for index in range(canvas.shape[0])],
            axis=0,
        )
        np.testing.assert_array_equal(together, separately)

    @unittest.skipUnless(
        Path("compression_baseline/results/pixelcnn_fmnist.pt").is_file(),
        "external neural-compression checkpoint is unavailable",
    )
    def test_recovered_checkpoint_is_architecture_compatible(self):
        model, saved = load_model(
            "compression_baseline/results/pixelcnn_fmnist.pt", "cpu"
        )
        self.assertEqual(saved["cfg"], {"n_channels": 72, "n_layers": 12, "k": 7})
        self.assertEqual(sum(p.numel() for p in model.parameters()), 7_312_504)


if __name__ == "__main__":
    unittest.main()
