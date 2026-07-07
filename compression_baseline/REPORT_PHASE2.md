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

## 6. Commercial-codec (gzip) control — was the ~1 dB a *learned*-compressor effect?

The neural baseline's advantage assumed a compressor with the **same source
knowledge and capacity** as the denoiser (PixelCNN 7.3M, learned on FMNIST). To
isolate that, we run the *identical* separation pipeline with a general codec
that has **no source knowledge**: gzip (DEFLATE, level 9). No new training.

**[1] gzip B_g distribution** (test[:2500], `results/gzip_stats.json`): mean
**3740.6 bits** (= Phase-1 gzip ref exactly), median 3824, p10 2551, p90 4848,
p99 5456, max 6088, min 1168 → **4.8 bpp** (vs PixelCNN 3.03 bpp). Roundtrip
bit-exact 20/20.

**[2] Containers** (channel set 3200; k=B_g+24; both feasible, no gap adjust):

| container | k | rate | vs neural MAX 0.389 | vs raw 0.500 | overflow |
|---|---|---|---|---|---|
| gzip-MAX | 6112 | **0.4851** | +0.096 | **−0.015 (≈ raw!)** | 0 |
| gzip-P99 | 5545 | 0.4401 | +0.051 | −0.060 | 32/3200 (1.0%) |

gzip frees almost no rate — gzip-MAX (0.485) is within 3 % of the raw rate 0.5,
so the separation has **almost no coding gain to spend**. Eb/N0 = Es/N0 −
10log10(rate): gzip-MAX +3.14, gzip-P99 +3.56 (vs ours +3.01).

**[3] Waterfall** (same grid, 3200 cw, Wilson CI; unified figure
`results/channel_waterfall.png`, the paper's representative figure — 6 curves,
legend = compressor/rate/info-bits). Knees (Es/N0 @ BLER = 0.1):

| system | compressor (bpp) | rate | k bits | knee Es/N0 | vs ours legacy |
|---|---|---|---|---|---|
| base MAX | PixelCNN (3.03) | 0.389 | 4896 | **−3.79** | **+0.93 dB** |
| base BG1LEAN | PixelCNN (3.03) | 0.333 | 4200 | < −4.0 (floor 0.0091) | — |
| gzip-P99 | gzip (4.8) | 0.440 | 5545 | −3.04 (floor 0.0100) | +0.18 dB* |
| **ours legacy** | raw (source in decoder) | 0.500 | 6296 | **−2.86** | — |
| ours EP | raw (source in decoder) | 0.500 | 6296 | −2.59 | −0.27 dB |
| **gzip-MAX** | gzip (4.8) | 0.485 | 6112 | **−2.53** | **−0.33 dB** |

*gzip-P99's deeper LDPC knee is bought by a 1.0 % overflow floor + catastrophic
failure (below), so it is dominated. Decomposition: gzip-MAX BLER is pure LDPC
failure (0 overflow); gzip-P99 sits on its overflow floor until its LDPC knee.

**[4] gzip failure mode** (`gzip_fail.py`): DEFLATE has a CRC32 trailer, so **a
single surviving bit error makes gzip.decompress raise — 20/20 total loss, no
image at all** (worse than PixelCNN's garbage-image; strictly no graceful path).

### Verdict — the coding gain scales with the source knowledge invested in compression

Measured, same pipeline / same pure-BP decoder, delivering the same image at
equal energy:

- **Learned compression (PixelCNN, 3.03 bpp) → base MAX knee −3.79 = +0.93 dB**
  over the joint denoiser system.
- **Commercial compression (gzip, 4.8 bpp) → gzip-MAX knee −2.53 = −0.33 dB**
  (i.e. *worse* than the joint system): gzip frees almost no rate, so separation
  has no gain to give.
- **Learned vs commercial, identical pipeline: −3.79 vs −2.53 = 1.26 dB** — the
  pure effect of compression strength (3.03 vs 4.8 bpp), nothing else changed.
- Investing the **same source knowledge in the decoder** (ours legacy, rate 0.5)
  lands **between** the two separation baselines (−2.86) and **uniquely fails
  gracefully** (PSNR ~20 dB) where both separation codecs are catastrophic
  (PixelCNN ~8 dB garbage; gzip: no image).

So the neural baseline's ~1 dB was **contingent on the learned compressor**: it is
the compression the source knowledge bought, not a property of "separation" per
se. Strip the source knowledge (gzip) and separation loses to the joint decoder.

---

## 7. Image-specific lossless codecs (PNG, WebP) — the fair "commercial" tier

**Why gzip alone is insufficient.** gzip is a general-purpose *byte* compressor,
blind to 2-D image structure — it is the "no knowledge" extreme, not a fair
representative of commercial lossless *image* coding. A fair middle tier is a
codec that knows image structure but is not trained on this source: **PNG**
(per-row predictive filters + zlib) and **WebP-lossless** (spatial prediction +
entropy). This is the "general image-structure knowledge" tier, between the
learned PixelCNN and gzip. (JPEG-LS was intended too but is unavailable here:
`pyjpegls` requires numpy≥2.0, incompatible with this env's sionna/TF numpy<2.0
— installing it breaks the stack; WebP-lossless stands in.)

**[1] bpp** (test[:2500], `results/png_stats.json`; roundtrip PNG 20/20, WebP
20/20):

| compressor | bpp | note |
|---|---|---|
| PixelCNN (learned) | **3.03** | source knowledge |
| WebP-lossless | **4.53** | best image codec here |
| PNG, IDAT-only (b) | 4.59 | fixed overhead removed |
| gzip | 4.77 | general |
| PNG, full file (a) | **5.17** | +57 B fixed chunk/header overhead **> gzip** at 28×28 |

At 28×28 the image codecs barely beat gzip — the tile is too small for row/spatial
prediction to pay off, and PNG's fixed overhead actually makes the whole file
*worse* than gzip. Container design uses the full file (a), conservatively.

**[2] containers** (channel set 3200; k=B+24; feasibility-checked, no gap hit):

| container | k | rate | vs raw 0.500 | overflow |
|---|---|---|---|---|
| WebP-MAX | 5832 | 0.4629 | −0.037 | 0 |
| gzip-MAX | 6112 | 0.4851 | −0.015 | 0 |
| PNG-MAX | 6584 | **0.5225** | **+0.023 (above raw!)** | 0 |
| (p99: WebP 0.417, gzip 0.440, PNG 0.469 — floored variants, tabulated only) |

PNG-MAX rate 0.523 is *above* the raw 0.5 — its overhead means the "compressed"
container carries **more** info bits than the raw image, so separation has
negative rate budget. All three rates differ from gzip-MAX by ≥0.02, so all were
measured (same grid, 3200 cw, Wilson CI).

**[3] the 3-tier spectrum** (MAX container, 0 overflow — `channel_waterfall.png`,
the paper's representative figure; knee = Es/N0 @ BLER 0.1):

| tier | system | compressor bpp | rate | k bits | knee | vs ours legacy |
|---|---|---|---|---|---|---|
| **learned source knowledge** | base MAX | PixelCNN 3.03 | 0.389 | 4896 | **−3.79** | **+0.93 dB** |
| source in the DECODER | ours legacy | raw | 0.500 | 6296 | −2.86 | — |
| general image structure | WebP-MAX | WebP 4.53 | 0.463 | 5832 | −2.80 | −0.06 dB |
| source in the DECODER | ours EP | raw | 0.500 | 6296 | −2.59 | −0.27 dB |
| no knowledge | gzip-MAX | gzip 4.77 | 0.485 | 6112 | −2.53 | −0.33 dB |
| general image structure | PNG-MAX | PNG 5.17 | 0.523 | 6584 | −2.03 | −0.83 dB |

The knee tracks compression strength almost monotonically: **~0.82 dB of coding
gain per bpp** of compression (−3.79 at 3.03 bpp → −2.03 at 5.17 bpp). Only the
**learned** compressor (3.03 bpp) buys a real margin over the joint decoder
(+0.93 dB). The best **image codec** (WebP, 4.53 bpp) merely **ties** it
(−0.06 dB); gzip and PNG are worse.

**Failure mode:** a single surviving bit error destroys every non-learned
container too — PNG 0/20 usable (16/20 decode-error, 4/20 garbage), WebP 0/20
(7/20 error, 13/20 garbage), gzip 0/20 (all decode-error). All separation codecs
are catastrophic; only the joint decoder degrades gracefully (PSNR ~20 dB).

**Verdict (measured, full spectrum).** The separation coding gain scales with the
source knowledge invested in compression: learned (PixelCNN) **+0.93 dB**, image
codecs **−0.06 to −0.83 dB**, general (gzip) **−0.33 dB** vs the joint decoder.
Put differently, the joint denoiser system (source in the *decoder*, rate 0.5) is
**Pareto-dominant over every non-learned separation baseline** — equal-or-better
BLER-vs-energy (only the learned PixelCNN beats it) *and* uniquely graceful
failure. General image-structure knowledge (PNG/WebP) is not enough at 28×28 to
change that; it takes a *learned* source model to make separation pay.

---

## Files (`compression_baseline/`)
gzip baseline: `gzip_encode.py`, `gzip_channel.py`, `gzip_fail.py`.
image codecs: `png_encode.py`, `png_channel.py` (PNG + WebP-lossless). 
Stage A `channel_encode.py` (torch/GPU) · stage B `channel_experiment.py`
(TF/Sionna, baseline waterfall) · `channel_our_waterfall.py` (ours, read-only
worktree import) · stage C `channel_verify_ac.py` (lossless, 200/200 bit-exact) ·
failure demo `channel_fail_ours.py`, `channel_fail_base_encode.py`,
`channel_fail_base.py`, `channel_fail_base_decode.py` · plots `plot_channel.py`,
`plot_failures.py`. Results: `channel_bler.json`, `our_waterfall.json`,
`channel_table.txt`, `channel_waterfall.png`, `failure_montage.png`,
`fail_*.npz` (gitignored binaries; regenerate via the scripts).
