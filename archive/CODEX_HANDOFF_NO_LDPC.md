# Codex Handoff — `no-LDPC`: RSC/BCJR + Source-Side SPC ISCD

> **Archived:** 이 문서는 구현 전 원 작업 명세이며 더 이상 활성
> 지시서가 아니다. 완료 결과와 최종 판정은
> [`../NO_LDPC_SUMMARY.md`](../NO_LDPC_SUMMARY.md)를 참조한다.

You are working on the repository containing the current `pure-EP_practical` implementation.

## Mission

Create a new branch named:

```text
no-LDPC
```

The purpose of this branch is to build a canonical iterative source-channel decoder without 5G LDPC:

\[
\text{score-based source SISO + high-rate source-side SPC}
\;\leftrightarrow\;
\text{RSC/BCJR channel SISO}.
\]

The top priority is **CRC-verifiable exact-recovery BLER**, not perceptual quality alone.

Do not modify the existing LDPC branches. Reuse the pretrained score model, source representation, channel models, datasets, evaluation utilities, and compression baseline wherever possible.

Report progress and results in Korean. Code, comments, docstrings, equations, and configuration names should be in English.

---

# 1. Work style

Do not begin with large-scale experiments.

Proceed in this order:

1. Audit the current repository.
2. Write an implementation plan and dependency map.
3. Implement CPU/unit-testable coding components.
4. Verify every component independently.
5. Integrate the iterative decoder.
6. Run only small smoke tests.
7. Add fair-comparison experiments.
8. Run larger experiments only after all invariants pass.

At the end of each phase, provide:

- files added/modified;
- exact mathematical update implemented;
- tests run and results;
- unresolved risks;
- next phase.

Do not launch long training. The existing score model must initially remain frozen.

---

# 2. First audit

Before coding, inspect and report:

- current source bit depth per pixel \(M\);
- image tensor shape and quantization rule;
- LLR sign convention;
- bit ordering within each pixel;
- current channel-coded length \(N\), expected to be around 12,600 if unchanged;
- current payload length \(K\);
- CRC implementation, if any;
- channel models and imperfect-CSI implementation;
- score denoiser input/output interface;
- `sigma`, `sigma_post`, source LLR clipping, and damping parameters;
- current neural-compression baseline resource-matching rule;
- existing experiment entry points and result formats.

Do not assume \(M=8\). Derive \(M\) from the current code and make the new implementation generic.

If the current code conflicts with the defaults below, preserve mathematical intent and report the conflict before making a structural change.

---

# 3. Target factorization

Let the original source payload be \(u\), with \(M\) quantization bits per source symbol/pixel.

For each source symbol \(u_k\), apply a systematic single-parity-check code:

\[
C_B:(M+1,M),\qquad
p_k=\bigoplus_{m=1}^{M}u_{k,m}.
\]

The local source-block codeword is

\[
v_k=[u_{k,1},\ldots,u_{k,M},p_k].
\]

Concatenate all \(v_k\), append a global CRC over the original payload \(u\), interleave, and encode with an RSC:

\[
u
\rightarrow C_B
\rightarrow [v,\mathrm{CRC}(u)]
\rightarrow \pi
\rightarrow C_{\rm RSC}
\rightarrow c.
\]

The receiver iterates:

\[
\text{BCJR channel SISO}
\leftrightarrow
\text{score-prior/SPC source SISO}.
\]

The CRC is used only for final exact-recovery validation in the first implementation. Do not feed CRC constraints into the iterative decoder yet.

---

# 4. Default design decisions

Use these defaults unless repository constraints make them impossible.

## 4.1 Source-side block code

Use one SPC bit per source symbol:

\[
p_k=u_{k,1}\oplus\cdots\oplus u_{k,M}.
\]

Properties:

- systematic;
- rate \(R_B=M/(M+1)\);
- local minimum distance \(d_{\min}=2\);
- exact candidate enumeration over \(2^M\) source values.

Do not use a hard parity detection rule inside the source decoder. Use exact local APP marginalization.

## 4.2 Global CRC

Use CRC-16-CCITT by default.

- Compute CRC over the original payload \(u\).
- Append CRC bits after the per-symbol SPC-coded image bits.
- CRC bits are RSC information bits but are not modeled by the score source decoder.
- The source decoder must return zero extrinsic LLR for CRC positions.
- Report both:
  - CRC-detected BLER;
  - true payload BLER from direct simulation equality.
- Also report undetected-error events, even if zero.

Keep CRC length configurable.

## 4.3 Interleaver

Use a deterministic random interleaver with a stored seed.

Default seed:

```text
20260722
```

Requirements:

- exact inverse;
- fixed across compared schemes unless an ablation explicitly changes it;
- save permutation metadata with results;
- interleave all RSC information bits, including CRC;
- source decoding occurs after deinterleaving.

Optionally add an S-random interleaver later, but not in the first implementation.

## 4.4 RSC

Use a standard 8-state systematic RSC as the first implementation.

Recommended default polynomial convention:

- feedback polynomial: \(13_{\rm oct}\);
- feedforward parity polynomial: \(15_{\rm oct}\);
- rate \(1/2\) before puncturing;
- zero-state termination initially;
- exact log-MAP BCJR as default;
- max-log-MAP as an optional ablation.

Document the bit/state convention carefully and test it against brute-force MAP on short sequences.

Do not silently choose a different polynomial convention. If the existing library uses reversed notation, document the exact mapping.

## 4.5 Fixed channel-resource matching

Preserve the existing transmitted coded length \(N\).

Let:

- \(K\): original payload bits;
- \(r_{\rm CRC}\): CRC bits;
- \(K_B=K+K/M\): SPC-coded image bits;
- \(K_{\rm in}=K_B+r_{\rm CRC}\): RSC information bits.

The RSC encoder produces systematic and parity outputs plus termination bits. Construct a deterministic puncturing/rate-matching mask so the final transmitted length is exactly \(N\).

Initial policy:

- keep all systematic RSC symbols;
- puncture parity symbols only;
- spread punctures uniformly;
- include termination overhead in the length accounting;
- assert that \(N\) is not smaller than the number of mandatory systematic/termination symbols.

Normalize energy consistently across all schemes.

## 4.6 Iteration

Each outer iteration consists of:

1. one full BCJR pass;
2. one score/SPC source-SISO update.

No BP warm-start state exists in this branch.

Initial outer iteration sweep:

```text
1, 2, 4, 6, 8
```

Initial source-feedback scaling sweep:

```text
alpha = 0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0
```

Do not add a separate `beta` feedback term. BCJR already produces the channel-to-source message directly.

---

# 5. BCJR SISO mathematics

Let \(L_A^{(t)}\) be the source-to-channel a-priori LLR on the RSC information bits after interleaving.

The BCJR decoder receives:

- systematic channel observations;
- parity channel observations;
- puncturing mask;
- channel state/estimate;
- \(L_A^{(t)}\).

It computes APP LLRs:

\[
L_{\rm APP}^{(t)}(b_i)
=
\log
\frac{P(b_i=1\mid y,L_A^{(t)})}
     {P(b_i=0\mid y,L_A^{(t)})}.
\]

The channel-to-source message is

\[
L_{C\to S}^{(t)}
=
L_{\rm APP}^{(t)}-L_A^{(t)}.
\]

Do **not** subtract the systematic-channel term again. In this serial factorization, all channel observations belong to the inner RSC/channel factor, so the message to the outer source factor is posterior minus only the incoming source a-priori.

Deinterleave \(L_{C\to S}^{(t)}\) before source decoding.

CRC positions may be hard-decoded for final output but must not enter the score source model.

---

# 6. Source/SPC SISO mathematics

For source symbol \(k\), let the incoming channel-to-source cavity LLRs be

\[
Q_k=[Q_{k,1},\ldots,Q_{k,M},Q_{k,p}],
\]

where \(Q_{k,p}\) is the SPC parity-bit LLR.

## 6.1 Score-model input

Use only the \(M\) systematic source-bit LLRs to construct the soft image:

\[
\tilde x
=
\operatorname{LLRToSoftImage}(Q_{\rm sys}).
\]

Run the existing one-step score/EDM denoiser:

\[
\mu_{\rm den}
=
D_\theta(\tilde x;\sigma).
\]

Do not implement multistep diffusion in the first branch.

## 6.2 Categorical source posterior

Use the existing readout logic to construct a normalized categorical posterior over all \(2^M\) quantized values:

\[
P_{\rm den}(a\mid Q_{\rm sys}),\qquad
a\in\{0,\ldots,2^M-1\}.
\]

This distribution already depends on the systematic-bit cavity through the denoiser input.

Therefore, to avoid double counting, combine it with the **parity-bit likelihood only**:

\[
P_k(a\mid Q_k)
\propto
P_{\rm den}(a\mid Q_{k,\rm sys})
\,
P_Q\!\left(B_{k,p}=p(a)\right),
\]

where

\[
p(a)=\bigoplus_{m=1}^{M} b_m(a).
\]

Use log-domain computation and `logsumexp`.

Do not multiply the systematic-bit likelihoods a second time unless an explicitly named ablation is added.

## 6.3 Posterior LLRs

For each systematic bit and the SPC bit, compute:

\[
L^{S,\rm post}_{k,\ell}
=
\log
\frac{
\sum_{a:b_\ell(a)=1}P_k(a\mid Q_k)
}{
\sum_{a:b_\ell(a)=0}P_k(a\mid Q_k)
}.
\]

The source-to-channel extrinsic is

\[
L_{S\to C,k,\ell}
=
L^{S,\rm post}_{k,\ell}
-
Q_{k,\ell}.
\]

Apply configurable damping/scaling and clipping:

\[
L_{A,k,\ell}^{(t+1)}
=
\alpha_t
\operatorname{clip}
\left(
L_{S\to C,k,\ell},
-L_{\max},
L_{\max}
\right).
\]

Interleave this message before the next BCJR pass.

CRC extrinsic entries remain zero.

---

# 7. Important interpretation

This is a strict SISO iterative source-channel decoder:

\[
\text{source/SPC outer factor}
\leftrightarrow
\text{RSC/channel inner factor}.
\]

The SPC is not a separate hard decoder. It is incorporated into the source APP calculation through candidate enumeration.

The score model is an approximate source-posterior module. The SPC part is exact conditional on the denoiser-derived categorical distribution.

Do not call the full algorithm exact Bayesian decoding. Use:

```text
score-prior-aided iterative source-channel decoding
```

or:

```text
RSC/BCJR ISCD with a composite score-prior/SPC source SISO decoder
```

---

# 8. Software architecture

Add modular interfaces rather than embedding everything in one decoder.

Recommended structure:

```text
coding/
    crc.py
    interleaver.py
    spc.py
    rsc.py
    bcjr.py
    puncturing.py

decoders/
    source_spc_siso.py
    rsc_source_iterative_decoder.py

experiments/
    no_ldpc_smoke.py
    no_ldpc_awgn.py
    no_ldpc_fading.py
    no_ldpc_imperfect_csi.py
    no_ldpc_compare.py

tests/
    test_crc.py
    test_interleaver.py
    test_spc.py
    test_rsc.py
    test_bcjr_bruteforce.py
    test_puncturing.py
    test_source_spc_siso.py
    test_iterative_noiseless.py
    test_resource_matching.py
```

Adapt names to repository conventions, but preserve modularity.

Provide a configuration object for:

- source bit depth;
- CRC type/length;
- RSC generators;
- termination mode;
- BCJR mode;
- interleaver seed;
- puncturing mask;
- outer iterations;
- alpha schedule;
- LLR clipping;
- `sigma`;
- `sigma_post`;
- channel model;
- imperfect-CSI parameters.

---

# 9. Mandatory tests

## 9.1 CRC

- known test vectors;
- detect one-bit corruption;
- encode/check round trip.

## 9.2 Interleaver

- `deinterleave(interleave(x)) == x`;
- deterministic across runs;
- no duplicate/missing indices.

## 9.3 SPC

For every candidate \(a\):

- encoder parity is correct;
- generated codeword satisfies parity;
- minimum distance is at least 2.

For a small \(M\), compare SISO marginalization with brute-force enumeration.

## 9.4 RSC

- noiseless encode/decode exact;
- termination reaches zero state;
- puncturing/depuncturing consistency;
- systematic positions preserved.

## 9.5 BCJR

On very short sequences:

- enumerate all possible input sequences;
- compute exact posterior probabilities;
- compare BCJR APP LLRs to brute-force values;
- test both signs of LLR convention;
- test punctured parity symbols;
- test max-log against log-MAP.

## 9.6 Source/SPC SISO

- with uniform denoiser posterior and reliable parity LLR, parity constraint changes bit marginals correctly;
- with parity LLR zero, result reduces to denoiser posterior;
- outgoing extrinsic equals posterior minus incoming cavity;
- no double counting of systematic cavity;
- CRC positions receive zero source extrinsic.

## 9.7 End-to-end

- noiseless channel: 100% true payload recovery and CRC pass;
- `alpha=0`: identical to RSC/BCJR-only baseline;
- source module disabled: no source feedback;
- one outer iteration: valid output;
- transmitted length exactly \(N\);
- same source image and channel instance across paired schemes;
- no NaN/Inf LLRs.

---

# 10. Baseline matrix

Implement paired comparisons under exactly matched:

- original image/payload;
- transmitted coded length \(N\);
- \(E_s/N_0\);
- channel realization;
- imperfect-CSI realization;
- CRC length;
- number of test blocks.

Required schemes:

1. Existing 5G LDPC only.
2. Existing warm-start LDPC + score source decoder.
3. RSC/BCJR only, no SPC, no score.
4. RSC/BCJR + SPC, score disabled.
5. RSC/BCJR + score, SPC disabled.
6. Full RSC/BCJR + joint score/SPC source decoder.
7. Lossless neural compression + conventional LDPC + CRC.

If a configuration cannot use the exact same original payload length or channel length, stop and report rather than silently changing fairness.

---

# 11. Metrics

Primary:

- true payload BLER;
- CRC-detected BLER;
- undetected error rate;
- BER.

Secondary:

- PSNR/SSIM for all outputs;
- PSNR conditional on CRC failure;
- expected distortion;
- decoding runtime;
- number of score-model calls;
- number of BCJR passes;
- transmitted source rate and effective total rate.

For compression failures, explicitly distinguish:

- CRC failure/outage;
- attempted entropy decode producing corrupted output;
- successful exact decompression.

---

# 12. Initial experiment plan

## Stage A: smoke

- 8–32 blocks;
- AWGN;
- one or two SNR points;
- no large plotting;
- verify all invariants.

## Stage B: component ablation

- outer iterations: 1, 2, 4, 6, 8;
- alpha sweep;
- SPC on/off;
- score on/off;
- log-MAP/max-log;
- clipping sweep.

## Stage C: main comparison

- AWGN;
- fading with perfect CSI;
- fading with imperfect CSI;
- identical paired channel realizations;
- enough blocks for stable BLER confidence intervals.

Do not claim an SNR knee from very small samples.

---

# 13. Decision log to maintain

Create a markdown file such as:

```text
NO_LDPC_DESIGN.md
```

Record:

- actual source bit depth \(M\);
- actual payload \(K\);
- fixed transmitted length \(N\);
- CRC polynomial;
- RSC polynomial convention;
- termination method;
- puncturing mask;
- interleaver seed;
- LLR convention;
- categorical readout equation;
- source/SPC double-counting decision;
- outer schedule;
- all deviations from this specification.

---

# 14. Non-goals for the first implementation

Do not add yet:

- multistep diffusion;
- source-aware custom SPC design;
- Hamming/BCH outer codes;
- tail-biting BCJR;
- joint CRC decoding;
- deep unfolding;
- DDECC;
- list decoding;
- learned puncturing/interleaving;
- long retraining.

First establish a correct and auditable canonical RSC/BCJR + SPC ISCD implementation.

---

# 15. Final deliverables

Provide:

1. working `no-LDPC` branch;
2. component tests;
3. end-to-end smoke tests;
4. `NO_LDPC_DESIGN.md`;
5. a short Korean implementation report;
6. exact commands to reproduce tests and experiments;
7. a table of all scheme rates and transmitted lengths;
8. initial BLER comparison plots only after correctness tests pass.

The implementation is complete only when:

- BCJR matches brute-force MAP on short sequences;
- source/SPC APP matches brute-force enumeration;
- noiseless CRC recovery is perfect;
- all schemes transmit exactly the same \(N\) symbols;
- `alpha=0` reproduces RSC-only behavior;
- paired channel realizations are verified.
