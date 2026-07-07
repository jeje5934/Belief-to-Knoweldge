# Compression baseline — Phase 2 report (channel integration + same-resource BLER)

Head-to-head of the **separation baseline** (source knowledge in the *compressor*,
pure-BP channel decoder) against the study's **denoiser-in-the-loop** system
(source knowledge in the *decoder*), at **equal total energy and equal channel
resource** (same N=12600 BPSK symbols, same noise variance / same Es/N0).

Pipeline (per image): PixelCNN+AC encode (B_c bits) → zero-pad to fixed container
K → CRC24A → LDPC5GEncoder(k=K+24, n=12600) → BPSK → AWGN → **LDPC5GDecoder pure
BP, 100 iters, no denoiser** → CRC24A. CRC pass = success (same criterion as our
system). 3200 codewords/point, Wilson 95% CI.

Both systems measured fresh on the **same Es/N0 grid** (−2.5 … −4.0 dB, 0.25 dB
steps). The denoiser system is the pure-EP_practical final-table config
(legacy/EP `[5]×20`, budget-100), imported **read-only from a git worktree**
(decoder.py etc. never modified) and run under the §5 recipe (TF on CPU, torch
denoiser on GPU). The baseline runs TF-on-GPU with torch absent — the two never
share a CUDA context (docs/COMPUTE_LESSONS §5).

---

## 1. Energy-equivalence axis (as specified)

Main axis = **same Es/N0** (same noise variance `no` applied to both systems).
`Es/N0 = Eb/N0 + 10·log10(rate)`; unit-energy BPSK ⇒ `no = 10^(−Es/N0/10)`.
Both systems are driven by the identical `no` at each grid point.

Because the systems carry **different numbers of info bits**, the same Es/N0 maps
to a different Eb/N0 for each — this is annotated in every table so a deeper knee
is read as the **rate cost**, not a free lunch:

| system | info bits k | rate | Eb/N0 = Es/N0 − 10log10(rate) |
|---|---|---|---|
| ours (legacy/EP) | 6296 | 0.4997 | Es/N0 + 3.01 |
| base MAX | 4896 | 0.3886 | Es/N0 + 4.10 |
| base BG1LEAN | 4200 | 0.3333 | Es/N0 + 4.77 |

Both deliver the **same 784-pixel image**; the baseline just needs fewer info
bits to do so (≈2380 compressed + padding vs 6272 raw), converting the surplus
into coding gain at equal energy.

---

## 2. Corrected LDPC feasibility (updates the Phase-1 floor claim)

Phase 1 reported only the BG2 floor (r ≥ 1/5). The **base-graph selection** at
n=12600 is a stricter, two-sided constraint (measured):

- **BG2** (k ≤ 3824): rate down to ~0.2, but **k ≤ 3824** (payload ≤ 3800).
- **BG1** (k > 3824): requires **rate > 1/3** (k ≥ 4200).
- **Unsupported gap: rate ∈ (0.3035, 0.3333)** — no base graph at n=12600.

The intended p99 container (k=4171, rate 0.331) falls in the gap and is
**infeasible**. Feasible containers used:

| container | k_ldpc | rate | base graph | overflow /3200 |
|---|---|---|---|---|
| **MAX** | 4896 | 0.3886 | BG1 | **0 (0.00%)** |
| **BG1LEAN** (rate=1/3) | 4200 | 0.3333 | BG1 | 29 (0.91%) |
| BG2LEAN (largest BG2) | 3824 | 0.3035 | BG2 | 90 (2.81%) |

---

## 3. Full waterfall — same Es/N0, 3200 cw, Wilson 95% CI

`results/channel_waterfall.png`, `results/channel_table.txt`.

| Es/N0 | ours legacy [5]×20 | ours EP [5]×20 | **base MAX** (r0.389) | **base BG1LEAN** (r0.333) | Eb/N0 ours/MAX/BG1 |
|---|---|---|---|---|---|
| −2.50 | 0.0019 [.0009,.0041] | 0.0097 [.0068,.0137] | **0/3200 [0,.0012]** | 0.0091 [.0063,.0130] | 0.51 / 1.61 / 2.27 |
| −2.75 | 0.0213 [.0168,.0269] | 0.2528 [.238,.268] | **0/3200 [0,.0012]** | 0.0091 [.0063,.0130] | 0.26 / 1.36 / 2.02 |
| −3.00 | 0.1978 [.184,.212] | 0.8194 [.806,.832] | **0/3200 [0,.0012]** | 0.0091 [.0063,.0130] | 0.01 / 1.11 / 1.77 |
| −3.25 | 0.6434 [.627,.660] | 0.9988 [.997,.9995] | **0/3200 [0,.0012]** | 0.0091 [.0063,.0130] | −0.24 / 0.86 / 1.52 |
| −3.50 | 0.9619 [.955,.968] | 1.0000 | **0/3200 [0,.0012]** | 0.0091 [.0063,.0130] | −0.49 / 0.61 / 1.27 |
| −3.75 | 0.9994 | 1.0000 | 0.0284 [.023,.035] | **0.0091 [.0063,.0130]** | −0.74 / 0.36 / 1.02 |
| −4.00 | 1.0000 | 1.0000 | 0.4788 [.462,.496] | **0.0091 [.0063,.0130]** | −0.99 / 0.11 / 0.77 |

Reading the curves:
- **ours legacy** knee ≈ −2.9 dB (0.002 → 0.20 → 0.96 across −2.5 … −3.5).
  **ours EP** is ~0.5 dB worse than legacy throughout (0.25 at −2.75 where legacy
  is 0.021) — consistent with the study's low-SNR legacy edge.
- **base MAX** holds **0/3200 down to −3.5 dB**, then its own LDPC waterfall hits:
  0.028 at −3.75, 0.48 at −4.0. Knee ≈ −3.85 dB.
- **base BG1LEAN** (more parity, rate 1/3) has **zero LDPC failures across the whole
  −2.5 … −4.0 range**; its BLER is pinned at the **0.91 % overflow floor**.

**Baseline failure decomposition:** overflow is SNR-independent (MAX 0, BG1LEAN
29). All baseline BLER above the floor is LDPC/CRC failure, which appears only at
MAX's −3.75/−4.0 knee; BG1LEAN never reaches it.

---

## 4. Failure-quality comparison (graceful vs catastrophic)

`results/failure_montage.png`. ~20 CRC-failed blocks per system, PSNR of the
reconstruction vs the original image.

| system | sample point | BLER there | recon PSNR (mean / median / min–max) | mode |
|---|---|---|---|---|
| ours (legacy) | Es/N0 −3.0 | 0.198 | **20.15 / 20.27 / 16.0–29.6 dB** | **graceful** |
| base (MAX) | Es/N0 −4.0 | 0.479 | **8.26 / 7.67 / 5.8–11.5 dB** | **catastrophic** |

- **Ours** payload *is* the raw image, so a decode failure is scattered pixel
  errors — the garment stays clearly recognizable (montage row 2).
- **Baseline** arithmetic coding is a sequential dependent code; a failed block
  carries ~590 bit errors, and the **first** one desyncs the decoder — the image
  is correct down to that point, then collapses into streak noise (montage row 4).
- Control: AC-decoding the *true* (error-free) container bits reconstructs
  **20/20 bit-exact** under the canonical (cuDNN-off, deterministic +
  batch-invariant) codec — confirming the failure is channel-induced, not a codec
  artefact.

So at comparable per-block failure rates the two failure *modes* differ in kind:
ours loses image quality gradually; the baseline loses the whole image at once.

---

## 5. Verdict (revised — resolution now sufficient)

Phase 2's provisional "conditional equivalence" was drawn from three points that
sit in the shared floor (both ~0, not separable at 3200 cw). The completed
waterfall resolves it — and **overturns it**:

1. **On BLER-vs-energy the separation baseline WINS across the waterfall region.**
   At equal total energy / channel, delivering the same image:
   - **base MAX beats ours legacy by ~0.8–1.0 dB.** At −3.0 dB MAX = 0/3200 [0,.0012]
     vs legacy 0.198 [.184,.212] — **disjoint CIs**. MAX holds 0/3200 through −3.5
     where legacy is already 0.96.
   - **base BG1LEAN is the most channel-robust** (0 LDPC failures to −4.0); below
     −3.5 it is the best of all four (0.0091 vs MAX 0.028/0.48 vs legacy ~1.0),
     limited only by its 0.91 % overflow floor.
   - This is the classical **source/channel-separation gain**, quantified:
     compression frees rate (0.389/0.333 vs 0.500) → coding gain → ~1 dB. The
     Eb/N0 columns show the mechanism (the baseline carries fewer info bits).

2. **On failure mode the denoiser-in-the-loop system wins.** Its failures are
   graceful (PSNR ~20 dB, recognizable image); the baseline's are catastrophic
   (PSNR ~8 dB, image destroyed after AC desync). The baseline has a structural
   all-or-nothing weakness the joint system does not.

3. **Net:** the two are **not equivalent** — they trade off. Separation gives
   better BLER-vs-energy (more images delivered bit-exact for the same energy) but
   catastrophic failures; the denoiser-in-the-loop system delivers fewer images
   correctly but degrades gracefully when it fails. Which is preferable depends on
   whether the application demands bit-exact delivery (favours the baseline) or
   tolerates degraded images (favours the joint decoder).

### Honest caveats
- The baseline's advantage is on the task's **main axis (same energy/channel,
  same delivered image)**. Per **info bit** (Eb/N0) the baseline operates 1.1–1.8
  dB higher — it carries less information; the win is that the *image* needs less
  information after compression.
- The baseline requires worst-case container sizing (MAX) for 0 overflow; the
  natural p99 rate is LDPC-infeasible (§2), and leaner BG2 sizing trades channel
  robustness for a larger overflow floor.
- ours legacy vs ours EP: legacy is uniformly better in this low-SNR waterfall,
  as the study's §F already found.

---

## Files (`compression_baseline/`)
Stage A `channel_encode.py` (torch/GPU) · stage B `channel_experiment.py`
(TF/Sionna, baseline waterfall) · `channel_our_waterfall.py` (ours, read-only
worktree import) · stage C `channel_verify_ac.py` (lossless, 200/200 bit-exact) ·
failure demo `channel_fail_ours.py`, `channel_fail_base_encode.py`,
`channel_fail_base.py`, `channel_fail_base_decode.py` · plots `plot_channel.py`,
`plot_failures.py`. Results: `channel_bler.json`, `our_waterfall.json`,
`channel_table.txt`, `channel_waterfall.png`, `failure_montage.png`,
`fail_*.npz` (gitignored binaries; regenerate via the scripts).
