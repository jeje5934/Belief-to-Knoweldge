# No-LDPC Design and Decision Log

## Status

**Closed after the final decision experiment.** This branch implements and
audits a canonical score-prior-aided iterative source-channel decoder with a
per-source-symbol SPC outer factor and an RSC/BCJR inner factor, but it is not
the selected production/research direction. Development returns to the 5G LDPC
path; no further no-LDPC tuning is planned.

## Final decision and closure

At CRC-BLER `1e-2`, the measured Es/N0 knees are:

| Arm | Knee |
| --- | ---: |
| A. 5G LDPC only, BP-100 | `-2.240 dB` |
| B. 5G LDPC + score, legacy `[5]x20` | **`-2.661 dB`** |
| C. RSC/BCJR + score + SPC | `+3.662 dB` |

C trails A by **5.90 dB** and B by **6.32 dB** under the same 6,272-bit
payload, `N=12,600`, payload rate, Es/N0 definition, and AWGN/perfect-CSI
channel. B is the final best system. The no-LDPC architecture does not beat the
conventional baseline and is therefore retired.

### Root cause

Code strength dominates. The weaker RSC moves its useful operating region to
about `+2.6 dB`; there, the first-BCJR source cavity is already about `99.99%`
bit-correct. The frozen source posterior has little useful uncertainty left to
resolve. Hard correction-to-contamination opportunities are approximately
`1:3,675` for RSC versus `1:7.82` for LDPC. Source usefulness is controlled by
the channel decoder's residual BER and whether those errors are dense enough
to be visible in image space, not by absolute SNR alone.

The apparent conflict between the 64-block diagnostic and final performance
was a condition mismatch. `14/64` was the hard result at the **first BCJR,
before any source/SPC update**. The score-off `2/256` result was the **final
second-pass output after IndependentBit+SPC updates**. Both used seed
`20260730`, and true-payload/CRC failure masks agreed, but the decoder stage and
source factor were different. The first-pass value remains valid for component
wiring and LLR-distribution diagnostics; it is not a score-off BLER baseline.

After aligning the complete decoders on the same 256 payloads and noise
realizations, the pre-audit canonical score path gives:

| Paired final-block transition | Count |
| --- | ---: |
| score-off success -> score-on failure | **22** |
| score-off failure -> score-on success | **0** |
| both fail | 2 |
| both succeed | 232 |

Thus `22 broken - 0 rescued = +22`, exactly matching `2/256 -> 24/256`.
Payload and CRC failure masks match in both arms, and total payload bit errors
increase from 6 to 67. The final BLER reversal is therefore caused by the score
path destroying 22 blocks that the SPC-only path decoded successfully while
rescuing none. Local or average bit improvements do not represent the final
block metric.

The `24/256` waterfall point used the original unbounded categorical adapter.
The later LDPC-compatible marginal bound fixed a real numerical-interface bug,
but its paired check was `28/256`, so the bug was not the performance cause.

### Rejected mechanisms

- payload/SPC/CRC wiring, sign, interleaver, MSB order, or raster-index errors;
- marginal-bound mismatch as the cause (real bug, no performance recovery);
- persistent `llr_clip=30` saturation;
- net contamination at the first damped source update
  (the local effective correction:contamination ratio was `14:1`);
- denoiser cavity echo or extrinsic double counting;
- net bit-error explosion in the second BCJR pass.

### Measurements reusable on the LDPC path

- Source-posterior bit-plane accuracy: LDPC is U-shaped
  (`94.9 -> 63.5 -> 73.1%`), while RSC falls approximately
  `95.6 -> 41.0%`.
- Conditional accuracy on cavity-error positions, correction/contamination
  opportunity counts, cavity `|LLR|`, actual sign flips, and final paired-block
  transitions form the reusable source-value diagnostic.
- A source prior is useful only where the channel decoder leaves sufficiently
  dense, image-visible residual uncertainty.

### Deliberately unresolved

The LDPC purity result `minus < legacy (D<C)` remains open. Cavity echo was
directly rejected and cannot explain the advantage of the incomplete legacy
cavity. This lies outside the closed no-LDPC branch and is not pursued here.

## Repository audit

| Item | Audited value |
| --- | --- |
| Source dataset/shape | Fashion-MNIST, one grayscale channel, `28 x 28` |
| Quantization | Dataset pixels are `uint8` in `[0, 255]`; score input is normalized by `x / 255` |
| Source bit depth | `M = 8` bits per pixel in the current checkpoint and scripts |
| Bit order | `numpy.unpackbits` default, MSB first within each pixel |
| LLR convention | `L = log P(bit=1) / P(bit=0)`; positive LLR hard-decides to bit 1 |
| Original payload | `K = 28 * 28 * 8 = 6,272` bits |
| Existing channel length | `N = 12,600` coded bits |
| Existing CRC | Sionna `CRC24A`, appended before 5G LDPC |
| New branch CRC | CRC-16-CCITT-FALSE (`poly=0x1021`, `init=0xFFFF`, MSB first) over the original payload |
| Existing channel models | BPSK/AWGN and QPSK fast Rayleigh fading with perfect or additive-error imperfect CSI |
| Imperfect CSI | `h_hat = h + e`, `e ~ CN(0, sigma_e2)`; receiver LLR uses `h_hat` |
| Colored-noise option | Stationary AR(1) noise with deliberately mismatched white-noise LLR |
| Score interface | Systematic payload LLR -> soft image -> one EDM denoiser call -> categorical pixel readout |
| Existing score defaults | fixed `sigma=0.3` in the principal experiments; `sigma_post=3.0` pixel units |
| Existing LLR clipping | `llr_max=30.0` in the principal LDPC experiments |
| Existing feedback | experiment-dependent `alpha`; legacy `beta` exists only in the LDPC path and is not used here |
| Result formats | JSON/CSV metrics and PNG plots under `results/`; top-level Python experiment entry points |
| Neural-compression baseline | An ignored PixelCNN checkpoint and legacy bytecode/results were recovered; source was restored and a fixed-container rule was implemented |

The new coding code is generic in `M`. The existing pretrained Fashion-MNIST
score model is structurally tied to `M=8`, `28 x 28`, and normalization by 255.

## Dependency map

```text
uint8 source symbols (MSB-first bits)
  -> per-symbol SPC(M+1, M)
  -> CRC-16 over original payload
  -> deterministic interleaver(seed=20260722)
  -> terminated RSC(feedback=13o, parity=15o)
  -> keep all systematic + uniformly selected parity
  -> exactly N=12600 channel bits
  -> channel LLR, log P(1)/P(0)
  -> log-MAP BCJR
  -> deinterleave(APP - incoming source prior)
  -> categorical score readout + exact SPC APP
  -> clip, alpha-scale, interleave, next BCJR pass
```

CRC positions receive zero source extrinsic. CRC is used only for the final
validation in this implementation.

## Exact dimensions for the default path

| Quantity | Value |
| --- | ---: |
| Original source payload `K` | 6,272 |
| Number of source symbols `K/M` | 784 |
| SPC-coded source `K_B = K + K/M` | 7,056 |
| CRC length | 16 |
| RSC information length `K_in` | 7,072 |
| Zero-termination inputs | 3 |
| Trellis/systematic length | 7,075 |
| Unpunctured parity length | 7,075 |
| Parity symbols retained | 5,525 |
| Parity symbols punctured | 1,550 |
| Final transmitted length | 12,600 |
| Original-payload rate `K/N` | 0.497777... |
| RSC-information rate `K_in/N` | 0.561269... |

The actual no-SPC comparison arm has `K_in=6,288`, trellis length 6,291,
and an unpunctured rate-1/2 length of 12,582. It reaches the same `N=12,600`
with 18 explicitly named, uniformly spaced repeated parity observations. The
receiver sums repeated LLRs. Its original-payload rate remains `K/N`; its
RSC-information rate is `6,288/12,600 = 0.499047...`.

## Comparison rate table

| Scheme | Original payload | Channel information before termination | Transmitted length | Payload rate | Rate-matching note |
| --- | ---: | ---: | ---: | ---: | --- |
| Existing 5G LDPC only | 6,272 | 6,296 | 12,600 | 0.497778 | Sionna 5G LDPC, CRC24A |
| Existing warm-start LDPC + score | 6,272 | 6,296 | 12,600 | 0.497778 | Same LDPC/CRC24A frame |
| RSC/BCJR only, no SPC | 6,272 | 6,288 | 12,600 | 0.497778 | 18 uniform parity repetitions |
| RSC/BCJR + SPC, score off | 6,272 | 7,072 | 12,600 | 0.497778 | 1,550 parity punctures |
| RSC/BCJR + score, no SPC | 6,272 | 6,288 | 12,600 | 0.497778 | 18 uniform parity repetitions |
| Full RSC/BCJR + score/SPC | 6,272 | 7,072 | 12,600 | 0.497778 | 1,550 parity punctures |
| Lossless neural compression + LDPC | 6,272 | 4,889 | 12,600 | 0.497778 | 4,873-bit zero-padded entropy container + CRC-16; effective LDPC rate 0.388016 |

The final three-arm decision uses the same original payload, transmitted length,
Es/N0, and indexed standard-noise realization. A/B retain their production
CRC24A while the RSC arms use CRC-16; both CRC-BLER and true payload BLER are
reported. Earlier internal no-LDPC matrix experiments that imposed a common
CRC-16 are not the final A/B waterfall. The channel-information column excludes
the three RSC termination inputs where applicable.

## Neural-compression resource matching

The seventh arm uses the recovered 7,312,504-parameter gated PixelCNN as a
frozen autoregressive probability model and a deterministic 16-bit-frequency,
32-bit-state arithmetic coder. Each 28x28 image is encoded sequentially in
raster order. The variable-length bitstream is placed at the start of a fixed
4,873-bit container and the remainder is zero padded. CRC-16 covers the entire
container, after which conventional 5G LDPC maps 4,889 information bits to the
same `N=12,600` channel symbols.

The container capacity is the observed maximum from the recovered 2,500-image
measurement resource. It is a fixed experimental policy, not a proof that all
Fashion-MNIST images fit. The implementation never truncates: any selected
stream longer than 4,873 bits stops the run and reports the overflowing indices
and lengths. Since arithmetic decoding reconstructs exactly 784 symbols, it
does not require the stream length and safely ignores trailing zero padding.

Probability-model inference is canonicalized to one image per forward pass.
Recovered archived streams showed that batched GPU convolution could move PMFs
across a 16-bit quantization boundary when the batch shape changed. Per-image
evaluation makes the stream independent of how encoder and decoder partition
the batch, including when only a CRC-passing subset is decoded.

At the receiver, CRC-failed blocks are declared outages and are not passed to
the entropy decoder. CRC-passing blocks are decoded, including the rare
undetected-error case, and the metrics separately count container errors,
CRC outages, attempted entropy decodes, corrupted entropy outputs, and exact
decompressions. Distortion metrics use an explicitly labeled all-zero outage
placeholder for CRC-failed blocks.

## RSC convention

The trellis has three memory bits and eight states. Octal polynomials are
expanded least-significant bit first as coefficients of `D^0, D^1, D^2, D^3`:

- feedback `13_o`: `[1, 1, 0, 1]`;
- feedforward parity `15_o`: `[1, 0, 1, 1]`.

State bit zero stores the newest recursive register value (`D^1`). For input
`u`, recursive input `r` and parity `p` are

```text
r = u xor m1 xor m3
p = r xor m2 xor m3
next_state = [r, m1, m2]
```

Termination inputs are found by exact enumeration of the `2^3` tail candidates
and must return every codeword to state zero.

## BCJR convention

For `L = log P(1)/P(0)`, branch metric is

```text
gamma = 0.5 * (2u-1) * (L_sys + L_A)
      + 0.5 * (2p-1) * L_parity
```

The channel-to-source message is exactly

```text
L_C_to_S = L_APP - L_A
```

The systematic-channel term is not subtracted. A punctured parity observation
has zero LLR. Exact log-MAP is the default; max-log-MAP is an explicit ablation.

## Source/SPC categorical update

The score categorical distribution consumes the systematic cavity through the
soft image and frozen denoiser. It is therefore combined with only the parity
likelihood:

```text
log P(a | Q) = normalize(log P_score(a | Q_sys)
                         + log P_Q(parity(a)))
```

Systematic and parity APP LLRs are exact `logsumexp` marginalizations over the
`2^M` candidates. The outgoing source message is posterior minus the complete
incoming source cavity. The implementation has a separate independent-bit
categorical provider for the SPC-only baseline; this provider consumes each
systematic likelihood exactly once.

The score provider also preserves the numerical contract of the legacy LDPC
`SourcePriorDenoiser`: marginal bit probabilities are bounded to
`[1e-7, 1-1e-7]` before the posterior logit is exposed. This bound is applied
after the exact categorical/SPC marginalization and only for providers that
declare it; the independent-bit SPC baseline remains unbounded. A wiring audit
found that the first categorical adapter had omitted this final marginal bound,
allowing hundreds of LLR units for the same denoiser input for which the LDPC
wrapper emits about 16. The bound restores direct module agreement but did not
improve the paired `+2.6 dB` result (`24/256` before, `28/256` after), so it was
not the cause of the score-path loss. Layout, interleaver direction, CRC/SPC
exclusion, bit sign, MSB-first order, and raster order all passed. The remaining
failure is an information-quality mismatch: throughout the usable waterfall
the BCJR cavity is 99.8--100% bit-correct, while the source posterior is only
about 69--75% bit-correct. Mixing the weaker estimate into the stronger one
therefore degrades decoding. A 32-block screen at `1.0`, `1.5`, and `2.0 dB`
found no crossover before BLER saturation. See
`RSC_SOURCE_WIRING_AUDIT_KO.md`.

## Configuration defaults

- interleaver seed: `20260722`;
- BCJR: exact log-MAP;
- termination: zero-state;
- outer iterations: configurable, default 2;
- alpha schedule: configurable, default `(0.1, 0.1)`;
- source LLR clipping: 30;
- score `sigma`: 0.3;
- categorical readout `sigma_post`: 3.0 pixel units;
- channel label: `awgn_bpsk`;
- imperfect-CSI variance: 0.

## Simulation budget while tuning

Until a candidate fixes the source-message failure, use 32 blocks for the first
paired screen, 64--128 only for survivors, and 256 for a single confirmation
point. Reserve 1,024--3,200-block runs for an explicitly requested final
waterfall or decision gate. This keeps diagnosis iterations short without
turning exploratory counts into performance claims.

## Detected conflicts and explicit holds

1. The existing LDPC path uses CRC24A. The no-LDPC path intentionally uses the
   requested CRC-16-CCITT and reports this difference instead of changing the
   LDPC branches.
2. The requested RSC-only/no-SPC comparison has `6,272 + 16 + 3 = 6,291`
   trellis inputs, so an unpunctured rate-1/2 RSC produces only `12,582` symbols,
   18 fewer than `N=12,600`. After recording this conflict, the comparison arm
   was implemented with an explicit `allow_parity_repetition` policy: send every
   parity once, repeat 18 uniformly spaced parity positions, and sum their LLRs.
   The full SPC path remains parity-puncturing-only and unchanged. The separate
   smoke-test `bcjr_only` arm still decodes the common SPC-expanded frame without
   applying the SPC factor and exists only as the exact `alpha=0` invariant.
3. The neural-compression source files had been removed while ignored `.pyc`,
   checkpoint, and result artifacts remained. The PixelCNN and arithmetic codec
   were reconstructed from the captured bytecode API/disassembly and checkpoint
   structure. Old batched archive streams are not used as canonical streams:
   their PMFs depend on the historical GPU batch shape. The restored codec uses
   per-image inference and passes independent-process round trips. The 57 MB
   checkpoint remains an external ignored resource and is not committed.
4. `pytest` is not installed in the remote container. Tests use standard-library
   `unittest` and remain pytest-compatible.

## Verification gates

- CRC known vector and corruption detection;
- deterministic interleaver and exact inverse;
- SPC parity and minimum distance by exhaustive enumeration;
- RSC zero termination;
- log-MAP and max-log BCJR equal brute-force path enumeration;
- source/SPC marginalization and no systematic double counting;
- CRC source-extrinsic positions remain zero;
- noiseless exact payload recovery and CRC pass;
- `alpha=0` equals BCJR-only behavior;
- default transmitted length is exactly 12,600;
- all emitted LLRs are finite in smoke tests;
- PSNR, 7x7 uniform-window SSIM, expected normalized MSE, and CRC-failure
  conditional PSNR are emitted by all new experiment entry points.
- the restored checkpoint loads with the reconstructed 7,312,504-parameter
  architecture, and arithmetic coding passes independent-process round trips;
- per-image probability evaluation is invariant to batch partitioning;
- the neural-compression arm passes bit-exact end-to-end recovery and exercises
  the CRC-outage path without invoking entropy decoding;
- all seven paired AWGN arms execute together with common CRC-16 and `N=12,600`.

## Alpha schedule selection

The paired small-sample search selected the conservative canonical setting
`outer_iterations=2`, `alpha_schedule=(0.1, 0.1)`, and exact log-MAP BCJR.
Across two independent validation seeds, `(0.1, 0.1)` and `(0.125, 0.1)` were
effectively tied in block errors; `(0.1, 0.1)` had fewer bit errors and avoids
the seed-dependent first-pass increase. A third pass did not reproduce its
fine-search gain on the validation seed. Max-log remains an optional throughput
mode, not the canonical accuracy default. Full details and commands are in
`NO_LDPC_ALPHA_SCHEDULE_REPORT_KO.md`.

A later recalibration moved tuning to the actual 3.0 dB knee and jointly
screened `sigma_post`, `llr_clip`, outer passes, and constant alpha. The best
observed candidate was `sigma_post=3`, `llr_clip=45`, four passes, and
`alpha=0.1`, but its aggregate 11/256 BLER 95% Wilson interval
`[0.0242, 0.0753]` overlapped the current canonical 13/256 interval
`[0.0299, 0.0849]`. It also doubled decoder runtime. Under the predeclared
CI-first rule, the candidate is recorded but not promoted; the configuration
defaults above remain canonical. See
`NO_LDPC_OPERATING_POINT_RECALIBRATION_KO.md`.

## Cross-branch common-resource verdict

A later decision-grade waterfall compared this canonical path against the
`practical_sigma` BP-100 and legacy score-aided 5G-LDPC arms at common payload
6,272, N=12,600, explicit Es/N0, and paired payload/noise plans. At CRC-BLER
`1e-1`, this path required 4.98 dB more than BP-100 and 5.54 dB more than the
legacy score-aided LDPC arm; at `1e-2`, the gaps were 5.90 and 6.32 dB.
The result does not support replacing the LDPC architecture. Full confidence
intervals, true-vs-CRC BLER, the alpha-collapse sweep, and auxiliary RSC
decomposition are in `THREE_ARM_WATERFALL_REPORT_KO.md`.
