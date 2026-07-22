# No-LDPC Design and Decision Log

## Status

This branch implements a canonical score-prior-aided iterative source-channel
decoder with a per-source-symbol SPC outer factor and an RSC/BCJR inner factor.
The pretrained score model remains frozen. Large experiments are blocked until
the component and smoke-test invariants pass.

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
| Neural-compression baseline | No implementation or resource-matching rule was found in the audited repository |

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
| Existing 5G LDPC only | 6,272 | 6,288 | 12,600 | 0.497778 | Sionna 5G LDPC, common CRC-16 bits |
| Existing warm-start LDPC + score | 6,272 | 6,288 | 12,600 | 0.497778 | Same LDPC frame |
| RSC/BCJR only, no SPC | 6,272 | 6,288 | 12,600 | 0.497778 | 18 uniform parity repetitions |
| RSC/BCJR + SPC, score off | 6,272 | 7,072 | 12,600 | 0.497778 | 1,550 parity punctures |
| RSC/BCJR + score, no SPC | 6,272 | 6,288 | 12,600 | 0.497778 | 18 uniform parity repetitions |
| Full RSC/BCJR + score/SPC | 6,272 | 7,072 | 12,600 | 0.497778 | 1,550 parity punctures |
| Lossless neural compression + LDPC | unresolved | unresolved | intended 12,600 | unresolved | no implementation/resource found |

All implemented matrix arms use the same original payload, CRC length, Es/N0,
and indexed standard-noise realization. The channel-information column excludes
the three RSC termination inputs where applicable.

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

## Configuration defaults

- interleaver seed: `20260722`;
- BCJR: exact log-MAP;
- termination: zero-state;
- outer iterations: configurable, default 4;
- alpha schedule: configurable, default `(0.1,)`;
- source LLR clipping: 30;
- score `sigma`: 0.3;
- categorical readout `sigma_post`: 3.0 pixel units;
- channel label: `awgn_bpsk`;
- imperfect-CSI variance: 0.

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
3. No lossless neural-compression implementation exists in this repository.
   That baseline remains unresolved and is not fabricated.
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
