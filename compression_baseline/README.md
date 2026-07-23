# Lossless neural-compression baseline

This package contains the restored source for the Fashion-MNIST gated PixelCNN
and deterministic arithmetic codec used by the seventh paired comparison arm.

The pretrained checkpoint is an external, gitignored artifact expected at:

```text
compression_baseline/results/pixelcnn_fmnist.pt
```

- size: `57,179,486` bytes
- SHA-256: `61654e4dd4647bfd568ebc81494bbd8da9344b94f53a828d89e7caf0f74312cb`
- architecture: 72 channels, 12 gated layers, kernel size 7
- parameters: 7,312,504
- checkpoint-reported Fashion-MNIST test rate: 3.0304374691 bits/pixel

The fixed experimental container is 4,873 bits, followed by CRC-16, and is
encoded by conventional 5G LDPC to exactly 12,600 transmitted bits. Streams
that exceed the container are reported and rejected; they are never truncated.
CRC-failed blocks are treated as outages and are not entropy-decoded.

Probability evaluation is deliberately performed one image at a time. Batched
GPU convolution changed quantized PMFs when batch shape changed, which can
desynchronize arithmetic decoding if only a CRC-passing subset is decoded. The
canonical per-image path is stable across batch partitions and processes.

Run the isolated end-to-end smoke test with:

```bash
python3 experiments/no_ldpc_baseline_matrix.py \
  --blocks 1 --esn0-db 12 --schemes neural_compression_ldpc --device cuda
```
