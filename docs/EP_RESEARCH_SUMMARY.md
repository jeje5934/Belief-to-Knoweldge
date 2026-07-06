# Learned-prior iterative decoding as Expectation Propagation — research summary

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
constant damping α ≈ 0.02 on the `[2]×15` schedule keeps the accumulated source
site at ~20–30 % of the belief, so the over-confident source can *guide* BP
without *dictating* a wrong codeword. **"If you cannot fix the correlated clump,
don't create it."** Damped EP is thus not a heuristic patch but the necessary
consequence of the exclusion chain.

## F. Performance — damped EP beats the BP ceiling at low SNR

Where BP struggles (0.5–0.7 dB), damped EP (α=0.02, `[2]×15`, `sigma_post=3.0`,
adaptive σ) is measured against BP-30 (the *same* 30-iteration BP budget) and
BP-100 (the BP-limit ceiling). `practical_lowsnr_baseline.py`, 3200 cw, Wilson CI:

| Eb/N0 | BP-30 (same budget) | **damped EP** | BP-100 (ceiling) | EP vs ceiling |
|---|---|---|---|---|
| 0.5 | 0.999 | **0.348** [.331,.364] | 0.574 | below (0.61×) |
| 0.6 | 0.983 | **0.115** [.105,.127] | 0.234 | below (**0.49×**) |
| 0.7 | 0.862 | **0.036** [.030,.043] | 0.051 | below (0.71×) |

At **every** low SNR damped EP decodes below the BP-100 ceiling: source knowledge
reaches block-error rates that BP cannot at *any* iteration count. The gain over
the same-budget BP-30 is large (e.g. 0.6 dB: 0.983 → 0.115). The legacy turbo
`α≈0.1` is, in hindsight, exactly this damped EP (a non-accumulating equivalent) —
so EP *explains* the turbo heuristic.

## G. Future work

- **Correlated error needs a richer per-bit interface.** The exclusion chain
  *proves* the limitation is the diagonal-Gaussian EP site: it cannot represent
  the off-diagonal correlated component. Catching it requires extending the
  interface beyond independent per-bit marginals — e.g. a **mixture / list**
  source site (carry a few candidate garments and let the code factor select),
  which the block-metric view of §D motivates.
- **Same-resource comparison vs neural-compression-then-transmit.** Our decoder
  *exploits* the payload's redundancy (a learned image prior) at decode time;
  the orthogonal design *removes* redundancy first (compress) then transmits.
  A matched-rate comparison would quantify redundancy-exploit vs
  redundancy-removal.
- **Turbo ↔ EP equivalence check** (optional, below).

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
| F | `practical_lowsnr_baseline.py` | `results/promptOpt0_baseline.*` | `pure-EP_practical` |

Companion docs: `EP_THEORY.md`, `EP_MIGRATION_PLAN.md`, `EP_SYSTEM_BLOCK.md`,
`EP_DIAGNOSTICS.md`, `EP_APPENDIX.md`, `EP_SCHEDULING_EXPERIMENT.md`,
`COMPUTE_LESSONS.md`.
