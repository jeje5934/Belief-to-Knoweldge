# Compression baseline — Phase 1 report (compressor build + rate diagnostic)

Comparison baseline for the transmission study: the classical **separation**
pipeline — *lossless neural compression → CRC → low-rate LDPC → same N=12600
channel → BP → CRC*. This phase builds the compressor, verifies bit-exact
losslessness, measures the compressed-bit distribution, and judges whether the
freed bits can actually be spent on parity given Sionna's LDPC rate limits.
**Channel simulation / head-to-head BLER is the next phase.**

All new code is under `compression_baseline/`. No existing decoder code was
modified (read-only, as required).

---

## 1. Fairness setup

| Property | Compressor (this work) | Transmission denoiser (existing) |
|---|---|---|
| Source knowledge | FMNIST **train** 60K only | FMNIST train (EDM SongUNet) |
| Capacity | **7.31M** params | 6.99M params |
| Data representation | 8-bit pixels 0..255 (= the 6272-bit payload) | same |
| General codecs | **none** (gzip measured only as a context number) | — |

Same source data, same ~7M capacity band, same 8-bit payload the channel sends.
The compressor is a **Gated PixelCNN** (two-stack, no blind spot;
`n_channels=72, n_layers=12, k=7`), autoregressive in raster order, output =
**256-way softmax per pixel**.

---

## 2. Training & test NLL (learning curve: `results/learning_curve.png`)

- Objective: NLL in bits/pixel on FMNIST train 60K; Adam, cosine LR, 30 epochs.
- **Converged test NLL = 3.030 bits/pixel = 2375.9 bits/image** (held-out 10K).
- Curve: 8.02 bpp (untrained, = log2 256) → 3.06 @ep17 → **3.030 @ep30**, flat.

**On the prompt's expected 1.5–2.5 bpp range — measured 3.03 is above it, and
this is correct, not a defect.** Diagnosis:
- FMNIST test is **50% exact-zero background pixels** (measured). Those code to
  near-0 bits. The other ~50% are garment interior — genuine 8-bit texture,
  high entropy (~6 bpp). Mean ≈ 3.0 bpp is the direct consequence.
- This matches the published FMNIST autoregressive/flow literature (~2.7–3.5
  bpd); the 1.5–2.5 guess was optimistic for 8-bit (vs binarized) FMNIST.
- Not-a-bug evidence: (a) causal receptive field verified (changing a pixel
  never moves any earlier-in-raster prediction); (b) untrained bpp = 8.000
  exactly; (c) arithmetic-coded bits match the model cross-entropy to +2.5
  bits/image (below); (d) 128/128 bit-exact round trips.

Consequence: the higher-than-expected NLL is exactly what pushes R_c up near the
LDPC floor (§4) — so it is the pivotal number, not a nuisance.

---

## 3. Entropy coder & bit-exact round trip

- **Arithmetic coder**: Witten–Neal–Cleary 32-bit integer coder driven by a
  **16-bit quantized CDF** (frequencies sum to 2^16, every symbol ≥1). Encoder
  and decoder build the CDF from identical integer frequencies.
- **Bit-exactness guarantee**: encode and decode call the *identical* sequential
  forward (`_step_pmf`) on the *identical* canvas (true past, zero future) → the
  quantized CDF is bit-identical on both sides by construction. (A single
  teacher-forced pass is faster but its pmf can differ from the sequential pmf by
  ~1e-7 — enough to flip a 16-bit boundary and desync the coder; that failure
  mode was observed and designed out. Decoding is 784 model calls/image — slow,
  correctness first, as specified.)
- **Round trip: 128 / 128 test images bit-exact** (≥100 required). ✓
- **Coder optimality**: arith-coded B_c vs the model's own cross-entropy on the
  same 2500 images → overhead **+2.5 bits/image (3.2 milli-bits/pixel)**.
  Near-ideal; confirms `B_c ≈ NLL×784 + small overhead`.

---

## 4. Compressed-bit distribution B_c (test, 2500 images)

| stat | bits/image | bpp |
|---|---|---|
| mean | **2366.9** | 3.02 |
| median | 2361.0 | 3.01 |
| p10 | 1475.8 | 1.88 |
| p90 | 3274.3 | 4.18 |
| p99 | 4143.0 | 5.28 |
| max | 4873 | 6.22 |
| min | 653 | 0.83 |
| std | 697.0 | — |

Sanity ✓: mean B_c 2366.9 ≈ NLL×784 (2375.9, full-set) — the 9-bit gap is the
first-2500 subset being slightly easier; on the *same* images overhead is +2.5.

Context only (general codec, **not** part of the baseline): gzip -9 on the raw
784-byte payload = **3740.6 bits (4.77 bpp)**. The neural compressor is **37%
smaller** than gzip — it uses source structure gzip cannot. Raw payload = 6272
bits (8 bpp); mean compression ratio **2.65×**.

---

## 5. Same-resource rate diagnostic

Raw system: k = 6272 + 24 CRC = **6296** info bits over **N = 12600** →
rate **0.4997**. Separation baseline: same N, k = B_c + 24 → R_c = (B_c+24)/12600.

**Sionna `LDPC5GEncoder` hard floor (measured): rate ≥ 1/5 = 0.2** — any k <
2520 raises `Unsupported coderate (r<1/5)`. So r ≥ 0.2 ⇔ k ≥ 2520 ⇔ B_c ≥ 2496
bits (3.184 bpp).

Per-image R_c: mean **0.190**, median 0.189, p90 0.262, p99 0.331, max 0.389.
**57% of images have per-image R_c < 0.2** (below the floor).

### Feasibility by variable-length strategy

**(ii) per-image adaptive rate** — infeasible directly: 57% of images want
rate < 0.2 and cannot be LDPC-encoded without repetition. Also the more complex
option. **Not recommended.**

**(i) fixed container padded to a percentile** (prompt-preferred for
simplicity/fairness — and it turns out to *also* fix feasibility):

| container | B_c\* (bits) | k_fixed | rate R_c | LDPC | overflow | mean pad |
|---|---|---|---|---|---|---|
| p90 | 3274 | 3298 | 0.262 | **supported** | 10.0% | 907 b |
| p99 | 4143 | 4167 | 0.331 | **supported** | 1.0% | 1776 b |
| max | 4873 | 4897 | 0.389 | **supported** | 0.0% | 2506 b |

Because the container is sized to a **high percentile / worst case**, k_fixed is
large enough to clear the 0.2 floor while still being **far below the raw 0.5**
— i.e. the freed bits genuinely become parity, and **no LDPC hacks (repetition,
N reduction) are needed.**

### Failure-mode fairness (confirmed)

Both paths are all-or-nothing at the CRC block: one surviving bit error → CRC24A
fails → whole image lost, identically on both sides. "CRC pass = success" applies
symmetrically ⇒ automatically fair. A container overflow (image doesn't fit) is
also just a declared failure — same success criterion, still fair.

---

## 6. Recommendation for the next (channel) phase

1. **Variable length: fixed container, Strategy (i).** Simplest, fairest, and it
   is what makes the scheme feasible under Sionna.
2. **Container size = max B_c over the test set (+ a small safety margin), k ≈
   4897–5000, rate R_c ≈ 0.389.** Gives **0% structural overflow** (no built-in
   BLER floor) and a supported LDPC rate, while still handing the channel
   ~1400 more parity bits than the raw system (6296→~4900 info bits over the same
   12600). A leaner **p99 container (rate 0.331)** is a valid alternative if the
   1% overflow is scored as failures — more parity, tiny structural loss.
3. **Do NOT** weaken the compressor to hit a target rate (option c) — that
   destroys the source-knowledge fairness. Not needed anyway.
4. **Do NOT** shrink N (option b) — unnecessary here, and it would force a
   re-measurement of the raw system at the new N to keep resources equal.

Net: the baseline is **feasible as a fixed-container, rate-≈0.39 LDPC over the
existing N=12600**, bit-exact lossless, at matched source knowledge and model
capacity. Ready for channel integration and head-to-head BLER vs the
denoiser-in-the-loop system.

---

## Files (`compression_baseline/`)

| file | role |
|---|---|
| `data.py` | FMNIST loader, 8-bit payload representation |
| `pixelcnn.py` | Gated PixelCNN (7.31M), 256-way softmax |
| `arithmetic_coder.py` | WNC integer coder, 16-bit quantized CDF (+ self-test) |
| `train.py` | training loop, learning-curve logging |
| `compress.py` | encode/decode, round-trip verify, B_c measurement, gzip ref |
| `rate_analysis.py` | R_c + LDPC-floor feasibility |
| `plot_curve.py` | learning-curve figure |
| `results/` | `pixelcnn_fmnist.pt`, `train_log.json`, `learning_curve.png`, `bc_stats.json`, `bc_lengths.npy`, `rate_analysis.json`, stdout logs |
