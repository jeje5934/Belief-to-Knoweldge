# Compression baseline — Phase 2 report (channel integration + same-resource BLER)

Head-to-head of the **separation baseline** (source knowledge in the *compressor*,
pure-BP channel decoder) against the study's **denoiser-in-the-loop** system
(source knowledge in the *decoder*), at **equal total energy and equal channel
resource** (same N=12600 BPSK symbols, same noise variance / same Es/N0).

Pipeline (per image): PixelCNN+AC encode (B_c bits) → zero-pad to fixed container
K → CRC24A → LDPC5GEncoder(k=K+24, n=12600) → BPSK → AWGN → **LDPC5GDecoder pure
BP, 100 iters, no denoiser** → CRC24A. CRC pass = success (same criterion as our
system). 3200 codewords/point, Wilson 95% CI. TF-on-GPU (torch absent in this
process — no CUDA-context clash, docs/COMPUTE_LESSONS §5).

---

## 1. Energy-equivalence axis (as specified)

Main axis = **same Es/N0** (same noise variance `no` applied to both systems).
Derived from the raw system's Eb/N0 points at its rate 0.4997:
`Es/N0 = Eb/N0 + 10·log10(0.4997) = Eb/N0 − 3.011 dB`.

| raw Eb/N0 | Es/N0 (dB) | no | base Eb/N0 @ MAX (r0.389) |
|---|---|---|---|
| 0.5 | −2.513 | 1.7836 | 1.59 |
| 0.6 | −2.413 | 1.7430 | 1.69 |
| 0.7 | −2.313 | 1.7034 | 1.79 |

The baseline's lower rate spends the compression saving as ~1 dB more Eb/N0 per
info bit at the same Es/N0. Waterfall extension: Es/N0 down to −3.7 dB.

---

## 2. Corrected LDPC feasibility (updates the Phase-1 floor claim)

Phase 1 reported only the BG2 floor (r ≥ 1/5). Measuring the **base-graph
selection** at n=12600 exposes a stricter, two-sided constraint:

- **BG2** (used when k ≤ 3824): rate down to ~0.2, but **k ≤ 3824** (payload ≤ 3800).
- **BG1** (used when k > 3824): requires **rate > 1/3** (k ≥ 4200).
- **Unsupported gap: rate ∈ (0.3035, 0.3333)** — no base graph covers it at
  n=12600. Sionna: *"Only coderate>1/3 supported for BG1. Repetition not supported."*

The intended **p99 container (k=4171, rate 0.331) falls exactly in this gap** and
is **infeasible**. We therefore compare three *feasible* containers:

| container | k_ldpc | rate | base graph | overflow (of 3200) |
|---|---|---|---|---|
| **MAX** | 4896 | 0.3886 | BG1 | **0 (0.00%)** |
| **BG1LEAN** (rate=1/3, p99 stand-in) | 4200 | 0.3333 | BG1 | 29 (0.91%) |
| **BG2LEAN** (largest BG2) | 3824 | 0.3035 | BG2 | 90 (2.81%) |

---

## 3. Results — same-Es/N0 BLER (3200 cw, Wilson 95% CI)

Reference (ours) quoted verbatim from pure-EP_practical final table
(EP_RESEARCH_SUMMARY §F), budget-100, rate 0.4997, mapped to Es/N0.

| Es/N0 | rawEb | ours legacy [5]×20 | ours EP [5]×20 | BP-100 ceiling | **base MAX** (r0.389) | base BG1LEAN (r0.333) | base BG2LEAN (r0.304) |
|---|---|---|---|---|---|---|---|
| −2.513 | 0.5 | 0.0009 [.0003,.0028] | 0.0106 [.0076,.0148] | 0.565 | **0/3200 [0,.0012]** | 0.0091 [.0063,.0130] | 0.0281 [.0229,.0344] |
| −2.413 | 0.6 | 0/3200 [0,.0012] | 0.0006 [.0002,.0023] | 0.217 | **0/3200 [0,.0012]** | 0.0091 [.0063,.0130] | 0.0281 [.0229,.0344] |
| −2.313 | 0.7 | 0/3200 [0,.0012] | 0.0006 [.0002,.0023] | 0.055 | **0/3200 [0,.0012]** | 0.0091 [.0063,.0130] | 0.0281 [.0229,.0344] |

Curve: `results/channel_waterfall.png`. Table: `results/channel_table.txt`.

### Failure decomposition (overflow vs LDPC/CRC), per container

| container | overflow (all SNR) | LDPC-fail @ −2.5…−2.3 | LDPC waterfall knee |
|---|---|---|---|
| MAX | 0 | **0 at all 3 points** | Es/N0 ≈ −3.7 (37/3200); −3.3 gave 1/3200 (Poisson noise) |
| BG1LEAN | 29 | **0 everywhere, down to −3.7** | not reached above −3.7 |
| BG2LEAN | 90 | **0 everywhere, down to −3.7** | not reached above −3.7 |

**The baseline's channel decoder never fails at the comparison points.** BLER is
set *entirely* by container overflow. Lowering the rate (more parity) does not
lower BLER — it only raises overflow. So the baseline's limiting quantity is the
**variance** of the compressed length, not the channel.

### Lossless end-to-end

200/200 images bit-exact through encode→AC-decode under the canonical codec
(cuDNN-off → deterministic **and** batch-invariant). A CRC24A pass guarantees
exact container recovery, so every CRC-passed image is reconstructed bit-exact.

---

## 4. Verdict (as measured — no adornment)

1. **base MAX ≡ our best (legacy).** At all three Es/N0 points MAX = 0/3200
   (CI ≤ 0.0012), identical to legacy (0/3200 at 0.6/0.7; 0.0009 [.0003,.0028] at
   0.5, whose CI overlaps MAX). **Not separable at 3200 cw → statistically
   equivalent.**
2. **base MAX > our EP at 0.5 dB.** MAX 0/3200 [0,.0012] vs EP 0.0106
   [.0076,.0148] — **disjoint CIs, MAX better**. At 0.6/0.7 both ~0 (overlap).
3. **The separation baseline works — but only with a worst-case (zero-overflow)
   container.** Trimming to a leaner rate hits an overflow BLER floor
   (BG1LEAN 0.0091, BG2LEAN 0.0281), both **worse than our best** (~10⁻³–10⁻⁴)
   with **disjoint CIs**. And the natural p99 rate (0.331) is LDPC-infeasible.
4. **Both approaches are conditionally equivalent at these operating points.**
   Putting source knowledge in the *compressor* (MAX) reaches the same ~0 BLER as
   putting it in the *decoder* (legacy), at the same total energy — neither wins
   decisively. The baseline's distinctive weakness is variable-length overflow;
   the decoder-in-the-loop system has no such structural floor.

### Honest caveats
- The comparison points sit in the **floor region** where both systems are ~0.
  0/3200 vs 0.0009 are not separable at 3200 cw; distinguishing MAX from legacy
  would need more codewords or lower SNR. MAX's own LDPC waterfall is ~1.2 dB
  below (Es/N0 ≈ −3.7, base Eb/N0 ≈ 0.4), i.e. it has ample channel margin here.
- Reference numbers are the study's previously-recorded values (TF-CPU, their
  3200 draw); this run is TF-GPU on a disjoint 3200 draw — equivalent within
  float epsilon per COMPUTE_LESSONS. Both are 3200 cw / Wilson CI at matched
  Es/N0.
- MAX rate 0.389 vs raw 0.5: the baseline does *not* need the freed rate here
  (the channel is already solved at these SNRs); the saving would matter at lower
  Es/N0, below both systems' floors.

---

## Files (`compression_baseline/`)
`channel_encode.py` (stage A, torch/GPU) · `channel_experiment.py` (stage B,
TF/Sionna) · `channel_verify_ac.py` (stage C, torch, lossless check) ·
`plot_channel.py` (table+curve). Results: `channel_bler.json`,
`channel_table.txt`, `channel_waterfall.png`, `channel_streams.npz` (gitignored
binaries; regenerate via the scripts).
