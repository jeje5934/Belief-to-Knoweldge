# Learned-prior iterative decoding as Expectation Propagation — research summary

> **2026-07-28 CRC 정정:** 이 문서의 과거 CRC-BLER/NACK 및 CRC 조건부
> 비교는 soft-logit CRC 오용 때문에 무효 보류다. sigma canonical, AWGN
> 3-way, 저예산 codec 대결의 hard-CRC 대체 수치는
> [`HARD_CRC_REMEASUREMENT_REPORT_KO.md`](../HARD_CRC_REMEASUREMENT_REPORT_KO.md)에
> 있다. fixed-latency는 후속 hard-CRC 재생성을 완료했으며
> [`FIXED_LATENCY_SNR_REPORT_KO.md`](../FIXED_LATENCY_SNR_REPORT_KO.md)와
> [`DISCUSSION_SUMMARY_KO.md`](../DISCUSSION_SUMMARY_KO.md)를 진입점으로 쓴다.
> fading은 아직 무효 보류다. BER/LLR/RMSE/syndrome처럼 CRC 비의존 계측은
> 별도로 유효하다.

A single narrative over the whole study: formulating an LDPC + score-denoiser
decoder as Expectation Propagation (EP), showing where and *why* pure EP fails,
excluding every attempt to repair it, and establishing the one strategy that
works. Every claim is backed by a named experiment (script + result file) with
quantitative numbers (BLER/BER, Wilson 95% CI, 3200 codewords/point unless noted,
Fashion-MNIST over AWGN, 5G-LDPC, CRC24A).

This is the paper skeleton. The headline is **not** "it failed" — it is:
**(1)** the failure has a precise cause (inter-pixel *correlated* hallucination),
**(2)** we characterise that cause by an exhaustive exclusion chain, and
**(3)** damped EP is not a hack but the *necessary* consequence — "if you cannot
fix the correlated clump, don't create it."

---

## A. EP formalisation of the decoder

The posterior over a codeword factorises into three factors:

```
p(x | y) ∝ p(y|x) · 1{Hx=0} · p_src(image(x_p))
          └channel┘ └ code ┘ └──── source ────┘
```

EP keeps one **site** per factor; the global belief is their product, a **sum**
in the LLR (natural) parameterisation:

```
posterior_LLR(payload) = channel_site + code_site + src_site
                          (frozen)      (= BP)      (denoiser)
```

Each factor is refined by the EP cycle **cavity → tilted → projection → site**:

- **channel** — exact; `channel_site` is the frozen channel LLR.
- **code** — EP with a fully-factorised family *is* sum-product BP (Minka 2001
  §4); `code_site` is the LDPC `msg_v2c` state. BP iteration count is a
  refinement **schedule**, not an approximation (`docs/EP_THEORY.md` §4.5–4.6).
- **source** — the score/EDM denoiser is the amortised projection (Tweedie: the
  denoiser output is the posterior mean of the clean image). The cavity fed to it
  is `BP_post − src_site` (removing the source's own previous message).

**Two facts that frame everything below.**
1. The legacy **turbo extrinsic** (`new_input = channel + β·bp_ext + α·src_ext`)
   is EP with an *incomplete cavity* and a *damped* site update; the double-count
   was removed (`docs/EP_MIGRATION_PLAN.md`).
2. **α (`ep_source_power`) is the source site DAMPING rate**, applied as an EMA
   accumulation `src_site ← (1−α)·src_site + α·src_full`. This is **damped EP**,
   NOT true power/fractional EP (which tempers the factor by `f^η` in the
   projection — a distinct, unimplemented method). The old name `fractional_ep`
   is a deprecated alias (`decoder.py`, `docs/EP_THEORY.md` §4.4).

Code: `decoder.py::LDPC5GDecoder_soft`, `source_prior.py`, `denoiser.py`.

## B. Pure EP fails — correlated hallucination

`full_ep` (α_ep = 1, the source site reflected in full — the EP-fidelity
alignment target) **does not decode**:

| decoder | BLER | BER | per-round BER (chunk 0 → 1, first source injection) |
|---|---|---|---|
| `full_ep`, multi-cat denoiser | **1.00** | 0.45 | 0.13 → **0.46**, stays ~0.45 |
| `full_ep`, Trouser denoiser | **1.00** | 0.49 | 0.14 → **0.52**, stays ~0.49 |
| pure BP (source off), reference | — | — | 0.14 → 0.04 (monotone descent) |

The source projection, fed a partly-converged BP cavity, returns an
over-confident **wrong image**; trusting that site fully poisons the belief from
the first injection. The error is an **inter-pixel correlated** hallucination
(the wrong garment), not independent per-pixel noise.
Experiments: `practical_promptB_fullep_diag.py` (per-round probe,
`ep_track_payload_hist`); `docs/EP_SCHEDULING_EXPERIMENT.md` §5.

## C. Exclusion chain — every attempt to remove the correlated error fails

Each row is an axis on which one might hope to catch/repair the correlated
hallucination, and the experiment that rules it out.

| # | attempt (axis) | result | why it fails | evidence |
|---|---|---|---|---|
| 1 | **BP warm-start double-count** | no hidden double-count | resetting `msg_v2c` each chunk gives the *same* explosion (0.13→0.46); control: reset cripples pure BP (0.004→0.13), proving reset is a strong intervention, yet `full_ep` is unchanged → the poison is the `src_site` prior, not BP state | `bp_reset_each_chunk` (Part C) |
| 2 | **denoiser under-training** | not under-trained | test denoising MSE 0.0105 (RMSE 0.0995); a fresh run plateaus by epoch ~20 (0.0106@ep5 → 0.0096@ep45) — a data/EDM-DSM **difficulty floor**, not capacity (~7M params) or epochs | `denoiser_training_probe.py` |
| 3a | **read-out precision — diagonal Tweedie** (input-dependent `σ²·diag ∂D/∂x̃`) | `full_ep` still BLER 1.0; damped **worse** (0.474 vs 0.342 at α=0.02) | a diagonal (per-pixel, local Jacobian) precision reports *false confidence* exactly where the denoiser hallucinates (54.6% of Trouser pixels sit at the variance floor); it cannot see inter-pixel correlation | `practical_promptD_*` (sibling branch `pure-EP_tweedie_practical`) |
| 3b | **read-out precision — global ε** (`sigma_post = 255·√MSE`) | bit-plane stratification confirmed, but damped **worse** (0.711 vs 0.114) | ε softens all but the top 1–2 bit-planes → source becomes inert; residuals are heavy-tailed (excess kurtosis 12.8 multi / 20.4 Trouser; >3σ tail 2.8% vs Gaussian 0.27%) so ε cannot capture the tail (the hallucination) | `practical_promptC_eps_bitplane.py`, `practical_promptC_sweep.py` |
| — | *(3a + 3b together)* | both axes fail | **the problem does not live on the per-bit / per-pixel precision axis** | — |
| 4 | **data domain — single category (Trouser)** | `full_ep` still explodes (0.14→0.49, BLER 1.0), even with diagonal Tweedie | the Trouser diagonal variance is genuinely *smaller/concentrated* (mean 8.8 < 13.0, 54.6% at floor) — yet it still fails, so the failure is the **diagonal *form* missing off-diagonal components**, not diagonal inaccuracy; intra-category shape variation remains multimodal | `practical_promptB_fullep_diag.py`, `practical_promptD_*` |
| 5 | **damping schedule** (chunk-wise α) | no schedule beats const α=0.02 | decreasing (Opt-1) and increasing (Opt-2) both fail on BLER; α magnitude sweet spot is 0.02 (const 0.03→0.194, 0.05→0.537) | `practical_promptOpt1_alphasched.py`, `practical_promptOpt2_incalpha.py` |
| 6 | **code cleanup** (source-off + pure BP tail) | cleanup cuts BER 4× but BLER 2–3× worse; true source-off worst | pure BP removes *scattered* errors but the **correlated clump survives** — the code constraint cannot fix it; removing the source reverts toward pure-BP low-SNR failure (0.89–0.93) | `practical_promptOpt3_hybrid.py` |

The chain forms a 2×2 exclusion for the read-out/precision axis (multi/Trouser ×
`sigma_post=3.0`/diagonal Tweedie — **all four BLER 1.0**) plus the domain,
schedule, and code-cleanup axes. **Correlated hallucination is removable by none
of them.**

## D. The BER/BLER dissociation (key finding)

Opt-2 (increasing α: low early, high late) exposes the mechanism directly.

| schedule | BER | BLER |
|---|---|---|
| const α=0.02 | 0.0064 | **0.124** |
| increasing α (0 → 0.03) | **0.0018** (3× lower) | 0.489 (worse) |

The per-round trajectory confirms the theory: under low early-α the mid-chunk BER
stays higher (less early poison), then late high-α exploits the now-reliable
cavity to drive the **final BER below constant-α**. But BLER *worsens*: the strong
late source freezes a residual handful of **correlated** wrong bits in a few
blocks — killing CRC while most bits are clean. Constant 0.02's gentle
accumulation instead keeps more blocks *fully* clean.

**This is the direct experimental proof that correlated errors are invisible to
average metrics (BER) but catastrophic to block metrics (BLER/CRC).** Damping is
the lever that trades average accuracy for block-level cleanliness.
(Opt-3 reinforces it: a pure-BP cleanup drives BER 4× *below* constant-α — pre
0.062 → post 0.0014 — yet BLER stays 0.28–0.38, because the clump survives.)

## E. The one strategy that works — gentle constant damping

Every attempt to *remove* the correlated clump post-hoc fails (§C). The clump is
invisible on the per-bit axis (§C 3), survives category restriction (§C 4), any α
schedule (§C 5), and code cleanup (§C 6); it is a block-level object (§D).

Therefore the only working strategy is to **never form the clump**: a gentle
damping α (small enough that the accumulated source site stays a fraction of the
belief) lets the over-confident source *guide* BP without *dictating* a wrong
codeword. **"If you cannot fix the correlated clump, don't create it."** Damped EP
is thus not a heuristic patch but the necessary consequence of the exclusion
chain. (The specific *best* `(schedule, α)` is budget-dependent — `[2]×15`, α≈0.02
at budget 30; **`[5]×20`, α≈0.01 at budget 100 → BLER ~10⁻³–10⁻⁴**; see §F.)

## F. Performance — two requirements for stable decoding, and where each decoder meets them

Source knowledge is a large win over BP at low SNR, but *only* when injected the
right way. Two requirements emerged; the final comparison shows which decoders
meet them. (`practical_final_table.py`, 3200 cw, Wilson CI, 0.6/0.5/0.7 dB; EP
uses adaptive σ, the turbo family fixed σ=0.3 — each at its own best.)

**Requirement (a): gentle injection over a warm-started, fine-grained schedule.**
A small per-chunk step with the code state *carried across chunks* (warm-start).
Evidence: `true_turbo` (mathematically exact — fresh BP each chunk, no warm-start)
**fails**, best BLER **0.55** (`practical_true_turbo_search.py`): fresh short BP
under-refines and the source exchange explodes at the first injection. Warm-start
is not an impurity to remove; it is load-bearing.

**Requirement (b): keep the source consistent across chunks (mechanism OPEN).**
Removing the explicit injected source feedback from the denoiser input (a "more
correct" cavity) **hurts** — legacy turbo β=0: `bp_post` C=0.055 vs purified
`minus` D=0.114, CIs disjoint (`practical_source_purity.py`). *Why* is unresolved:
the magnitude hypothesis is refuted (|src_ext| ratio 0.98; larger α *explodes*
`minus`, 0.1→0.108→0.2→0.919, not compensates —
`practical_purity_discriminate.py`), and the direction-consistency ("mode-locking")
hypothesis is **also refuted** (`bp_post` chunk-to-chunk cosine 0.906 is *not*
higher than `minus` 0.929 — `practical_direction_diag.py`). So input-level source
retention helps for a reason not yet pinned down. Recorded honestly as open.

**Schedule/budget is a large lever, and it reconciles the earlier "heuristic beats
principled" result.** Giving every decoder the same BP budget (100) and a
coarser-per-chunk schedule `[5]×20`:

| Eb/N0 | BP-100 (ceiling) | legacy `[2]×15` (bud-30) | **legacy `[5]×20` (bud-100)** | EP `[2]×15` (bud-30) | **EP `[5]×20` (bud-100)** | true_turbo | minus |
|---|---|---|---|---|---|---|---|
| 0.6 | 0.217 | 0.0141 | **0/3200** (CI ≤ .0012) | 0.120 | **0.0006** [.0002,.0023] | 0.554 | 0.109 |
| 0.5 | 0.565 | 0.065 | **0.0009** [.0003,.0028] | — | **0.0106** [.0076,.0148] | — | — |
| 0.7 | 0.055 | 0.0044 | **0/3200** (CI ≤ .0012) | — | **0.0006** [.0002,.0023] | — | — |

(`0/3200` is an **upper bound** — Wilson 95% CI ≤ 0.0012 — not a point estimate;
at 3200 cw it does not separate legacy from EP.)

Reading it:
- **Both** legacy and EP, at budget-100 `[5]×20`, reach **~10⁻³–10⁻⁴** — far below
  the BP-100 ceiling (0.217/0.565/0.055). Source knowledge decodes where BP cannot
  at any iteration count.
- **At matched budget + schedule, legacy ≈ EP.** At 0.6 and 0.7 dB the two are
  **not separable** at 3200 cw — legacy scores 0/3200 (CI ≤ 0.0012), which only
  bounds it above and overlaps EP's 0.0006 [.0002,.0023]; calling one better there
  is unsupported. The **only** SNR where they separate is **0.5 dB**, where legacy
  (0.0009 [.0003,.0028]) is below EP (0.0106 [.0076,.0148]) with disjoint CIs — so
  the heuristic keeps a small edge **at 0.5 dB only**. EP's earlier apparent
  inferiority (0.12 vs 0.014 at `[2]×15`) was a **schedule artefact** — `[2]×15`'s
  2-iter chunks under-refine the cavity; give EP `[5]×20` and it matches the
  heuristic (equivalent at 0.6/0.7 dB).
- So the honest verdict is **not** "the heuristic beats the principled method." At
  matched conditions they are close (conditionally equivalent); the heuristic keeps
  only a small 0.5-dB edge
  whose mechanism (requirement b) is unresolved.

**σ-configuration correction + the σ axis (branch `practical_sigma`).** The EP
`[5]×20` numbers above used **syndrome-adaptive** denoiser σ (the handcrafted
default lookup), so "EP best" implicitly credited the adaptive-σ machinery. A
dedicated σ-strategy study (3200 cw, Wilson CI) revises this: at **0.5 dB fixed
σ=0.3 scores 0.0066 [.0043,.0100]**, *better* than the handcrafted adaptive
(0.0106–0.0112) — the adaptive σ was not helping. So EP's own best σ is **fixed**.
This does **not** change the legacy-vs-EP verdict: legacy (0.0009 [.0003,.0028])
still separates from EP-fixed (0.0066 [.0043,.0100]) at 0.5 dB with disjoint CIs;
both floor at 0.6/0.7. Neither open-loop annealing nor a principled two-stage
syndrome→SNR→σ* LUT beats fixed σ with CI separation — **σ is not a BLER lever**,
and closed-loop adaptive ≈ open-loop annealing (σ-trajectory corr 0.86–1.0,
because the syndrome decays predictably so the state carries no extra info).

Critically, a **large initial σ** — the ICDM "smear the modes, then sharpen"
device ported to the chunk axis — is **catastrophic and monotone** (anneal σ_s
0.6→0.022, 1.2→0.105, 2.4→0.320 BLER): a single-shot Tweedie call at large σ
yields a **grey over-smoothed mean** (denoised-vs-true PSNR 15.9→6.2 dB as σ
0.15→2.4), *not* a wide-but-coherent unimodal, and the damage does not reverse
when σ later drops (per-round BER plateaus). This **extends the exclusion chain to
the σ axis**: ICDM-style mode devices need the diffusion *trajectory*; our
single-shot projection cannot separate modes by σ alone → they require
**sampler-level intervention**. (`SIGMA_REPORT.md`, `sigma_denoise_viz.png`.)

### Revision history (honest record of interpretation updates)

Each update was forced by a specific experiment — this trail is part of the result.
- ~~"turbo α≈0.1 *is* damped EP (equivalent)"~~ (old §F) → refuted: at `[2]×15`
  legacy ≫ EP (`practical_turbo_vs_ep.py`), then re-refined: the gap is a schedule
  artefact, at `[5]×20` legacy ≈ EP (final table).
- ~~"incomplete cavity = protective self-**damping** (magnitude)"~~ → refuted:
  |src_ext| magnitude is equal (H1/H2 discriminate).
- ~~"the impurity implements **mode-locking** (higher direction consistency)"~~ →
  refuted: consecutive cosine is not higher for `bp_post` (direction diag). The
  D<C fact stands; its mechanism is **open**.
- ~~"EP `[5]×20` best uses adaptive σ (final table 0.0106 @0.5, 0.0006 @0.6)"~~ →
  corrected: the adaptive σ (handcrafted lookup) was **not** an improvement;
  **fixed σ=0.3 is EP's best** (0.0066 < 0.0106 at 0.5 dB). σ is not a BLER lever
  (`practical_sigma`; see §F σ-axis note). Legacy-vs-EP verdict unchanged.

## G. Future work

- **Correlated error needs a richer per-bit interface.** The exclusion chain
  (§C) shows the diagonal-Gaussian EP site cannot represent the off-diagonal
  correlated component. A **mixture / list** source site (carry a few candidate
  garments; let the code factor select) is the principled candidate — and it is
  also the natural explicit form of **requirement (b)** in §F, whose implicit
  mechanism in the heuristic remains unresolved. Making mode-consistency explicit
  (rather than purifying the cavity) is the right direction for a principled gain.
  The σ-axis result (`practical_sigma`) sharpens this: a single-shot Tweedie
  projection **cannot** separate modes by raising σ (it smears to the grey mean),
  so mode-consistency must be made **explicit** (mixture/list) rather than coaxed
  from one projection — and any diffusion-style mode device needs the sampler
  *trajectory*, not a single call.
- **Same-resource comparison vs neural-compression-then-transmit.** Our decoder
  *exploits* the payload's redundancy (a learned image prior) at decode time;
  the orthogonal design *removes* redundancy first (compress) then transmits.
  A matched-rate comparison would quantify redundancy-exploit vs
  redundancy-removal. (A `compression_baseline/` scaffold exists for this.)

---

## Experiment index

| section | script | result file(s) | branch |
|---|---|---|---|
| B | `practical_promptB_fullep_diag.py` | `results/promptB_fullep_diag.*` | `pure-EP_practical` |
| C-1 | `bp_reset_each_chunk` (decoder flag) | (Part C notes) | both |
| C-2 | `denoiser_training_probe.py` | `results/denoiser_training_probe.*` | `pure-EP_practical` |
| C-3b | `practical_promptC_eps_bitplane.py`, `practical_promptC_sweep.py` | `results/promptC_*` | `pure-EP_practical` |
| C-3a/4 | `practical_promptD_tweedie_fullep.py`, `practical_promptD_damped.py`, `practical_promptD_varlog.py` | `results/promptD_*` | `pure-EP_tweedie_practical` |
| C-2 (Trouser) | `train_denoiser_trouser.py` → `checkpoints/denoiser_trouser.pt` | `results/denoiser_trouser.*` | `pure-EP_practical` |
| C-5, D | `practical_promptOpt1_alphasched.py`, `practical_promptOpt2_incalpha.py` | `results/promptOpt{1,2}_*` | `pure-EP_practical` |
| C-6 | `practical_promptOpt3_hybrid.py` | `results/promptOpt3_hybrid.*` | `pure-EP_practical` |
| F (a) | `practical_true_turbo_verify.py`, `practical_true_turbo_search.py` | `results/true_turbo_search.*` | `pure-EP_practical` |
| F (b) | `practical_source_purity.py`, `practical_purity_discriminate.py`, `practical_direction_diag.py` | `results/source_purity.*`, `results/purity_discriminate.*`, `results/direction_diag.*` | `pure-EP_practical` |
| F (turbo↔EP) | `practical_turbo_vs_ep.py` | `results/turbo_vs_ep.*` | `pure-EP_practical` |
| F (table) | `practical_ep100_search.py`, `practical_final_table.py`, `practical_lowsnr_baseline.py` | `results/ep100_search.*`, `results/final_table.*`, `results/promptOpt0_baseline.*` | `pure-EP_practical` |
| F (σ axis) | `sigma_experiment.py`, `calibrate_sigma_2stage.py`, `sigma_compare.py`, `sigma_denoise_viz.py`; `AnnealingSigmaScheduler` | `results/sigma_compare.*`, `sigma_2stage*.*`, `sigma_denoise_viz.png`, `SIGMA_REPORT.md` | `practical_sigma` |

Companion docs: `EP_THEORY.md`, `EP_MIGRATION_PLAN.md`, `EP_SYSTEM_BLOCK.md`,
`EP_DIAGNOSTICS.md`, `EP_APPENDIX.md`, `EP_SCHEDULING_EXPERIMENT.md`,
`COMPUTE_LESSONS.md`.
