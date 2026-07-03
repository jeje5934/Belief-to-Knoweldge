# Belief-to-Knowledge — LDPC + score-denoiser decoding as Expectation Propagation

5G-LDPC decoding of **Fashion-MNIST images** over an AWGN channel, with a
learned EDM **score denoiser** supplying an image-domain prior. This branch
(`pure-EP_ada-sigma`) reformulates the decoder as **Expectation Propagation
(EP)** on a three-factor graph and documents, honestly, where pure EP works and
where it does not.

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

## 5. Documentation (`docs/`)

| file | role |
|---|---|
| `EP_THEORY.md` | factor graph, per-factor EP cycle, LLR algebra, damped EP (vs true power EP), scheduling |
| `EP_MIGRATION_PLAN.md` | the double-count fix (turbo → EP), line-level |
| `EP_SYSTEM_BLOCK.md` | corrected block diagram + before/after |
| `EP_DIAGNOSTICS.md` | convergence/evidence metrics (Z_i, Δsite), syndrome↔EP mapping |
| `EP_SCHEDULING_EXPERIMENT.md` | schedule sweep + **decode-quality** experiments (§5: full EP fails, damping needed) |
| `EP_APPENDIX.md` | full probabilistic derivation (A.1–A.6), all approximations named |

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
