# Belief-to-Knowledge — LDPC + score-denoiser decoding as Expectation Propagation

> **CRC correction (2026-07-28):** historical CRC-BLER/NACK values in this
> branch may include soft-logit CRC misuse. Current hard-CRC primary results are
> in [`HARD_CRC_REMEASUREMENT_REPORT_KO.md`](HARD_CRC_REMEASUREMENT_REPORT_KO.md);
> fixed-latency has been regenerated with hard CRC in
> [`FIXED_LATENCY_SNR_REPORT_KO.md`](FIXED_LATENCY_SNR_REPORT_KO.md), while
> fading remains pending remeasurement. The discussion entry point is
> [`DISCUSSION_SUMMARY_KO.md`](DISCUSSION_SUMMARY_KO.md); the printable one-page
> handout is [`output/pdf/PROFESSOR_DISCUSSION_ONEPAGE_KO.pdf`](output/pdf/PROFESSOR_DISCUSSION_ONEPAGE_KO.pdf).

> **MSE experiment branch:** BP-50 `[10]×5`에서 hard-image MSE를 기준으로
> source 파라미터를 다시 훑은 소표본 견적은
> [`MSE_BUDGET50_ESTIMATE_KO.md`](MSE_BUDGET50_ESTIMATE_KO.md)에 있고,
> 512-block concealment 공정 비교·PixelCNN/raw 대조·crossover 최종 결과는
> [`MSE_FAIRNESS_EXTENSION_KO.md`](MSE_FAIRNESS_EXTENSION_KO.md)에 있다.
> BLER canonical을 대체하는 결과가 아니라 `codex/mse-optimization`의 별도
> distortion-oriented 실험이다.

> **Lossy-JSCC positioning branch:** D²-JSCC/NTSCC/DeepJSCC와의 자원 정의,
> SNR 환산, 직접 비교 실현성, 클래스 경계는
> [`JSCC_POSITIONING_REPORT_KO.md`](JSCC_POSITIONING_REPORT_KO.md)에 정리했다.
> 핵심 결론은 ours(`r=8.04`)와 대표 lossy JSCC(`r≈.02~.167`)가 48~129배 다른
> 자원 영역에 있어 현재 수치를 head-to-head 성능 비교로 읽을 수 없다는 것이다.

> **Multiview receiver-side experiment:** 송신단이 통신하지 않는 상관 카메라의
> 무손실 side view를 수신단에서만 쓰는 게이팅·altproj 통합 결과는
> [`MULTIVIEW_REPORT_KO.md`](MULTIVIEW_REPORT_KO.md)에 정리했다. 재학습 없는
> early-gated side pull은 BLER 0.1 knee를 약 0.19 dB 개선했지만, 무비용 side라는
> 낙관 조건에서도 PixelCNN-MAX까지의 격차 0.69 dB가 남았다.

5G-LDPC decoding of **Fashion-MNIST images** over an AWGN channel, with a
learned EDM **score denoiser** supplying an image-domain prior. This branch
(`pure-EP_ada-sigma`) reformulates the decoder as **Expectation Propagation
(EP)** on a three-factor graph and documents, honestly, where pure EP works and
where it does not.

> **no-LDPC 갈래 종료:** 동일 자원 최종 비교에서 RSC/BCJR + score +
> SPC가 5G LDPC 기준선에 뒤져 이 갈래를 종료했다. 최종 판정과 전체
> 문서 지도는 [`NO_LDPC_SUMMARY.md`](NO_LDPC_SUMMARY.md)를 참조한다.

> ### Status — read this first
> **EP-aligned decoder. `full_ep` (α=1, the source site reflected in full) is the
> EP-fidelity *reference point* — but it does NOT decode (BLER 1.0).** The
> denoiser is an over-confident, un-calibrated factor, and trusting its site
> fully corrupts the belief every round. The **working configuration is
> damped EP** — a damped source site (`ep_source_power ≈ 0.02` on the
> `[2]×15` schedule → BLER ≈ 0.004 at 0.8 dB). The legacy turbo `α≈0.1` is, in
> hindsight, exactly this damped EP (an implicit precision discount on the
> denoiser). Do not read "EP" here as "it just works": the honest result is that
> *pure* EP is a reference point and *damped* EP is the practical decoder.
>
> Terminology: the α<1 path is **damped EP** (`ep_update="damped_ep"`; the site
> update is EMA-blended). It is **not** true power/fractional EP (which tempers the
> factor by `f^η` in the projection — a distinct, unimplemented method). The old
> `"fractional_ep"` flag name was a misnomer, kept only as a deprecated alias.
> See `docs/EP_SCHEDULING_EXPERIMENT.md` §5, `docs/EP_APPENDIX.md` §A.6,
> `docs/EP_THEORY.md` §4.4.

---

## 1. System overview (EP framework)

The posterior over a codeword is a product of three factors (details:
`docs/EP_THEORY.md`):

```
p(x | y) ∝ p(y|x) · 1{Hx=0} · p_src(image(x_p))
          └channel┘ └ code ┘ └──── source ────┘
```

EP keeps one **site** per factor; the global belief is their product, which in
the LLR (natural) parameterisation is a **sum**:

```
posterior_LLR(payload) = channel_site + code_site + src_site
                          (frozen)      (= BP)      (denoiser)
```

Each factor is refined by the EP cycle **cavity → tilted → projection → site**:

* **channel factor** — exact; `channel_site` is the frozen channel LLR.
* **code factor** — EP with a fully-factorised family *is* sum-product BP
  (Minka 2001 §4); `code_site` is the LDPC `msg_v2c` state.
* **source factor** — the score denoiser is the amortised projection (Tweedie:
  the denoiser output is the posterior mean of the clean image). The cavity fed
  to it is `BP_post − src_site` (removing the source's own previous message —
  the double-count fix).

Full derivation: `docs/EP_THEORY.md`, `docs/EP_APPENDIX.md`. Block diagram:
`docs/EP_SYSTEM_BLOCK.md`.

## 2. BP iterations are a refinement *schedule*

Each parity check is its own factor, so a BP iteration is an EP refinement step
of the check factors — the iteration count is a **schedule**, not an
approximation. Sweeping schedules (`docs/EP_SCHEDULING_EXPERIMENT.md`) finds a
sweet spot **`[2]×15`** (few BP steps, source often): many-BP-then-source
(`[50]×3`) drives the check messages into a limit cycle, while too-few-BP
(`[1]×20`) under-refines the cavity and the source drifts. `[2]×15` reaches a
stable outer orbit.

## 3. Key finding (honest)

Convergence *dynamics* (Δsite, logZ) do **not** imply decode quality. Measuring
BLER/BER (0.8 dB, CRC-checked; harness validated by turbo reproducing prior
results):

| config | BLER | note |
|---|---|---|
| baseline BP, 30 it (no prior) | 0.50 | same BP budget as `[2]×15` |
| baseline BP, 100 it | 0.008 | BP-limit reference |
| **`full_ep` `[2]×15`** | **1.00** | pure EP — corrupts from round 1 |
| **damped EP, α≈0.02 `[2]×15`** | **0.004** | working decoder |
| turbo α=0.1 β=0.1 `[10]×3` (legacy) | 0.016 | = damped EP, non-accumulating |

Mechanism: a partly-converged BP cavity fed to the denoiser yields an
over-confident *wrong* image; `full_ep` trusts that site fully and the belief is
poisoned every round (proven by per-round BER and cleanup-BP tests,
`docs/EP_SCHEDULING_EXPERIMENT.md` §5.1). The damping weight is an **implicit
precision discount** on the miscalibrated denoiser factor. This is the sharpest
form of the project's **EP-fidelity vs performance** tension.

## 4. This branch: source-projection variance via adaptive σ

The EP source projection formally needs the **cavity variance**. This branch
supplies it through the existing **syndrome-ratio adaptive σ** scheduler: the
syndrome ratio (fraction of unsatisfied parity checks) is a cheap proxy for the
cavity's **global, per-image** uncertainty (distance from the code manifold),
and selects the denoiser σ per chunk (`FixedSigmaScheduler` /
`HandcraftedLookupScheduler` / `CalibratedLookupScheduler`, see
`syndrome_sigma_schedule.py`). The pixel→bit read-out uses a **fixed
`sigma_post`** — a per-pixel precision **slot** exists
(`source_prior.py`: `projected_pixel_precision`, `forward(return_precision=)`)
but is not filled here. (The sibling branch
`pure-EP_tweedie-2nd-diagonal-precision` fills it with a real diagonal Tweedie
2nd moment; see there.)

## 4b. Practical branch (`pure-EP_practical`) — low-SNR optimization

This branch optimizes the working **damped-EP** decoder in the SNR region where
BP struggles, to measure and maximize the **source-knowledge gain** over BP.
(High SNR ≥0.8 dB is uninformative — BP-100 alone reaches BLER≈0.008 there.)

**Low-SNR baseline** (`practical_lowsnr_baseline.py`, 3200 codewords/point,
batch 64 × rounds 50, EP = `damped_ep, α_ep=0.02, [2]×15, adaptive σ`; baselines
share the EP BP budget K=30, ceiling = BP-100):

| Eb/N0 | BP-30 (same budget) | **EP (α=0.02)** | BP-100 (ceiling) | source gain vs BP-30 | EP below ceiling |
|---|---|---|---|---|---|
| 0.4 | 1.000 | 0.619 | 0.859 | +0.381 | yes (0.72×) |
| 0.5 | 0.999 | 0.352 | 0.557 | +0.647 | yes (0.63×) |
| 0.6 | 0.983 | **0.129** | 0.249 | **+0.854** | **yes (0.52×)** |
| 0.7 | 0.864 | 0.037 | 0.050 | +0.827 | yes (0.73×) |

EP (source prior) beats the **BP-100 ceiling at every low SNR** — source
knowledge reaches BLER that BP cannot at any iteration count. Focus SNR for
subsequent steps = **0.6 dB** (max absolute + relative gain, stable statistics,
visible headroom). Numbers are Wilson-95%-CI stable.

**Denoiser training-state probe** (`denoiser_training_probe.py`): the active
`checkpoints/denoiser.pt` is **well trained, not under-trained**. Held-out
FashionMNIST test:

* **unweighted per-pixel denoising MSE = 0.0105** (already ~10× below the "0.1"
  figure, which is actually the *weighted DSM objective* ≈ 0.125, not pixel MSE).
* A fresh run plateaus by **epoch ~10–20**: test MSE 0.0106 (ep5) → 0.0096
  (ep45), i.e. training 40 more epochs buys <10% — this is a **data /
  EDM-DSM difficulty floor, not under-training or capacity** (~7M params suffice).
* Locating `denoiser.pt` by its test MSE puts it at **~epoch 5–8** (consistent
  with the README default `--epochs 5`).

**Consequence:** the decode-quality failures (correlated hallucination §Part C,
gray-mush OOD input §Part D) are **not** a denoiser-training problem — "train the
denoiser longer" is not a performance lever. See `docs/COMPUTE_LESSONS.md` for
the CPU/GPU run notes from these experiments.

### 4b.1 Chunk-wise α_ep schedule (Opt-1/Opt-2) — not a BLER lever

`ep_source_power` (α_ep, the source site damping rate) can be scheduled per chunk
via `ep_source_power_schedule` (list), and the last M chunks can be truly
source-off via `ep_source_off_tail=M` (both options, default off). Sweeping
decreasing (Opt-1) and increasing (Opt-2) schedules against the constant α=0.02
anchor (0.6 dB, 3200 cw, Wilson CI; `practical_promptOpt{1,2}_*.py`):

- **Neither decreasing nor increasing α beats constant α=0.02 on BLER.** The α
  magnitude sweet spot is 0.02 (const 0.03 → 0.194, const 0.05 → 0.537); schedule
  shaping is not an effective BLER lever. True source-off tails and α=0 freeze
  tails give at most a marginal, CI-overlapping change.
- **Key finding — BER/BLER dissociation.** An increasing schedule (low early,
  high late) **cuts BER 3×** (0.0064 → 0.0018) via late-cavity source gain — the
  per-round trajectory confirms the theory (mid-chunk BER stays higher under low
  early-α, then late high-α drives the final BER below constant-α). But it
  **worsens BLER** (0.124 → 0.489). Mechanism: strong late source fixes a residual
  handful of **correlated** wrong bits per block — killing CRC while most bits are
  clean; constant 0.02's gentle accumulation keeps more blocks *fully* clean.
- This is the direct experimental confirmation that **correlated errors are
  invisible to average metrics (BER) but catastrophic to block metrics
  (BLER/CRC)**: damping trades average accuracy for block-level cleanliness, and
  α=0.02 constant is the robust BLER optimum.

### 4b.2 Research conclusion

Pure EP (`full_ep`, α=1) fails by an **inter-pixel correlated hallucination** in
the source projection; every attempt to remove it post-hoc is excluded (§C), so
the working strategy is **gentle damping** that never forms the clump. Two
requirements make source injection stable (§F): **(a)** gentle steps over a
**warm-started** fine-grained schedule (the mathematically exact `true_turbo`,
which drops warm-start, fails at BLER 0.55); **(b)** keeping the source consistent
across chunks — purifying the denoiser input *hurts* (0.114 vs 0.055) for a reason
that is **still open** (magnitude and mode-locking hypotheses both refuted).

**Given a matched BP budget and schedule (`[5]×20`, budget 100), the principled
EP and the heuristic turbo are close** — both reach BLER ~10⁻³–10⁻⁴ at 0.6/0.7 dB,
far below the BP-100 ceiling; legacy keeps a small edge, clear only at 0.5 dB. The
earlier large "heuristic ≫ principled" gap was a **schedule artefact** (`[2]×15`'s
2-iter chunks under-refine EP). Full narrative, final table, and the honest
interpretation-update history: **`docs/EP_RESEARCH_SUMMARY.md` §F**.

## 5. Documentation (`docs/`)

| file | role |
|---|---|
| `EP_RESEARCH_SUMMARY.md` | **whole-study narrative** (paper skeleton): EP formalisation → full-EP failure → exclusion chain → BER/BLER dissociation → damping necessity → performance → future work |
| `EP_THEORY.md` | factor graph, per-factor EP cycle, LLR algebra, damped EP (vs true power EP), scheduling |
| `EP_MIGRATION_PLAN.md` | the double-count fix (turbo → EP), line-level |
| `EP_SYSTEM_BLOCK.md` | corrected block diagram + before/after |
| `EP_DIAGNOSTICS.md` | convergence/evidence metrics (Z_i, Δsite), syndrome↔EP mapping |
| `EP_SCHEDULING_EXPERIMENT.md` | schedule sweep + **decode-quality** experiments (§5: full EP fails, damping needed) |
| `EP_APPENDIX.md` | full probabilistic derivation (A.1–A.6), all approximations named |
| `COMPUTE_LESSONS.md` | CPU vs GPU run notes: bridge is CPU-bound, GPU gain is small, long GPU training is crash-prone (checkpoint/resume) |

## 6. Running

Smoke-sized runs (the host has had GPU lockups on large sweeps — prefer CPU or
cap GPU memory; see `safe_sweep.sh`):

```bash
# α sweep (turbo / damped-EP weighting) at β=0
CUDA_VISIBLE_DEVICES=0 python3 experiment.py --alpha 0.1 --ebno 0.8 --batch 16 --rounds 1

# baseline vs adaptive-σ comparison
python3 plot_comparison.py --alpha 0.1 --beta 0.1 --sigma 0.3 --ebno 0.8 \
    --batch 16 --rounds 1 --compare-adaptive --no-save-plot

# Pure-EP vs baseline-BP Eb/N0 sweep (set the EP curve to a WORKING config —
# damped_ep, small ep_source_power — not full_ep, which is BLER 1.0)
python3 ep_snr_sweep.py --batch 32 --rounds 24
```

The EP decoder (`decoder.py::LDPC5GDecoder_soft`) is driven by kwargs:
`ep_mode=True`, `ep_update="full_ep" | "damped_ep"`, `ep_source_power`,
`bp_schedule=[2]*15`, `adaptive_sigma=True`. `ep_mode=False` selects the legacy
turbo path.

**Dependencies**: Python 3.8+, TensorFlow 2.x, PyTorch, Sionna (LDPC/mapper/AWGN),
NumPy, Matplotlib, torchvision. The comms chain runs in TensorFlow; the denoiser
runs in PyTorch; `denoiser.py` bridges via NumPy.

## 7. Legacy turbo path (preserved)

The original turbo update `new_input = channel + β·bp_ext + α·src_ext` is kept
as `ep_mode=False` (knobs `alpha`, `beta`). It is the practical decoder's
non-accumulating equivalent of damped EP (best known: `α=β=0.1`, `σ=0.3`).
Design notes: `adaptive_sigma_review.md`; context handoff: `HANDOFF.md`.
