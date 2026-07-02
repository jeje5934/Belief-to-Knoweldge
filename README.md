# Belief-to-Knowledge — LDPC + score-denoiser decoding as Expectation Propagation

5G-LDPC decoding of **Fashion-MNIST images** over an AWGN channel, with a
learned EDM **score denoiser** supplying an image-domain prior. This branch
(`pure-EP_tweedie-2nd-diagonal-precision`) reformulates the decoder as
**Expectation Propagation (EP)** on a three-factor graph and adds the principled
**diagonal Tweedie 2nd-order (per-pixel) precision** for the source site — then
shows, honestly, that it is *not enough* to make pure EP decode.

> ### Status — read this first
> **EP-aligned decoder. `full_ep` (α=1, the source site reflected in full) is the
> EP-fidelity *reference point* — but it does NOT decode (BLER 1.0).** The
> denoiser is an over-confident, un-calibrated factor, and trusting its site
> fully corrupts the belief every round. The **working configuration is
> fractional EP** — a damped source site (`ep_source_power ≈ 0.02` on the
> `[2]×15` schedule → BLER ≈ 0.004 at 0.8 dB). The legacy turbo `α≈0.1` is, in
> hindsight, exactly this fractional EP (an implicit precision discount on the
> denoiser). Do not read "EP" here as "it just works": the honest result is that
> *pure* EP is a reference point and *fractional* EP is the practical decoder.
>
> **This branch adds diagonal Tweedie precision — and it does not save pure EP.**
> Filling the source site's per-pixel precision with the real diagonal Tweedie
> 2nd moment (`σ²·diag ∂D/∂x̃`) still gives BLER 1.0 under `full_ep`: the
> denoiser's error is *inter-pixel correlated*, so a *diagonal* precision cannot
> catch it, and the full `d×d` covariance (`d=784`→`d²≈6.1·10⁵`) is intractable
> and unrepresentable by the Gaussian EP site. So the site cannot be correctly
> *shaped*, only *down-weighted* → fractional EP. (Sibling branch
> `pure-EP_ada-sigma` uses a fixed `sigma_post` instead.)
> See `docs/EP_SCHEDULING_EXPERIMENT.md` §5 and `docs/EP_APPENDIX.md` §A.6.

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
| **fractional EP, α≈0.02 `[2]×15`** | **0.004** | working decoder |
| turbo α=0.1 β=0.1 `[10]×3` (legacy) | 0.016 | = fractional EP, non-accumulating |

Mechanism: a partly-converged BP cavity fed to the denoiser yields an
over-confident *wrong* image; `full_ep` trusts that site fully and the belief is
poisoned every round (proven by per-round BER and cleanup-BP tests,
`docs/EP_SCHEDULING_EXPERIMENT.md` §5.1). The damping weight is an **implicit
precision discount** on the miscalibrated denoiser factor. This is the sharpest
form of the project's **EP-fidelity vs performance** tension.

## 4. This branch: diagonal Tweedie 2nd-order precision (and why it is not enough)

The EP source projection formally needs the source site's **precision** (2nd
moment), not just its mean. This branch **computes it per pixel** from the
denoiser's own Jacobian via Tweedie's formula:

```
Var[s_j | x̃] = σ² · ∂D_j/∂x̃_j          (Tweedie diagonal 2nd moment)
```

estimated by **Hutchinson finite differences** (Rademacher probes,
**no backprop**; `source_prior.py::_tweedie_pixel_std`) and fed as the per-pixel
`sigma_post` into the pixel→bit read-out. Toggle with
`denoiser.tweedie_precision = True`; **default off** falls back to the fixed
`sigma_post` (the `pure-EP_ada-sigma` behaviour). The `σ` itself still comes from
the syndrome-ratio adaptive scheduler (a global per-image proxy for the cavity
variance; `syndrome_sigma_schedule.py`).

**Honest result — it does not rescue pure EP.** With the diagonal Tweedie
precision, `full_ep` still decodes at **BLER 1.0**. The denoiser fails by
hallucinating the *wrong garment* — an **inter-pixel correlated** error. A
**diagonal** precision only measures per-pixel local sensitivity and cannot see
it; catching it needs the full projected covariance `σ²·∂D/∂x̃`, a dense `d×d`
matrix (`d = n_pix = 784` → `d² ≈ 6.1·10⁵` entries) that is intractable to
estimate/propagate and, crucially, **not representable** by the diagonal Gaussian
EP site. So *exact* EP for this factor is out of reach: the site cannot be
correctly **shaped**, only **down-weighted** — which is the fundamental
justification for **fractional EP** (§3). Details:
`docs/EP_SCHEDULING_EXPERIMENT.md` §5.2, `docs/EP_APPENDIX.md` §A.6.4. The sibling
branch `pure-EP_ada-sigma` omits this computation and uses a fixed `sigma_post`.

## 5. Documentation (`docs/`)

| file | role |
|---|---|
| `EP_THEORY.md` | factor graph, per-factor EP cycle, LLR algebra, fractional/damped EP, scheduling |
| `EP_MIGRATION_PLAN.md` | the double-count fix (turbo → EP), line-level |
| `EP_SYSTEM_BLOCK.md` | corrected block diagram + before/after |
| `EP_DIAGNOSTICS.md` | convergence/evidence metrics (Z_i, Δsite), syndrome↔EP mapping |
| `EP_SCHEDULING_EXPERIMENT.md` | schedule sweep + **decode-quality** experiments (§5: full EP fails, fractional needed) |
| `EP_APPENDIX.md` | full probabilistic derivation (A.1–A.6), all approximations named |

## 6. Running

Smoke-sized runs (the host has had GPU lockups on large sweeps — prefer CPU or
cap GPU memory; see `safe_sweep.sh`):

```bash
# α sweep (turbo / fractional-EP weighting) at β=0
CUDA_VISIBLE_DEVICES=0 python3 experiment.py --alpha 0.1 --ebno 0.8 --batch 16 --rounds 1

# baseline vs adaptive-σ comparison
python3 plot_comparison.py --alpha 0.1 --beta 0.1 --sigma 0.3 --ebno 0.8 \
    --batch 16 --rounds 1 --compare-adaptive --no-save-plot

# Pure-EP vs baseline-BP Eb/N0 sweep (set the EP curve to a WORKING config —
# fractional_ep, small ep_source_power — not full_ep, which is BLER 1.0)
python3 ep_snr_sweep.py --batch 32 --rounds 24
```

The EP decoder (`decoder.py::LDPC5GDecoder_soft`) is driven by kwargs:
`ep_mode=True`, `ep_update="full_ep" | "fractional_ep"`, `ep_source_power`,
`bp_schedule=[2]*15`, `adaptive_sigma=True`. `ep_mode=False` selects the legacy
turbo path.

**Dependencies**: Python 3.8+, TensorFlow 2.x, PyTorch, Sionna (LDPC/mapper/AWGN),
NumPy, Matplotlib, torchvision. The comms chain runs in TensorFlow; the denoiser
runs in PyTorch; `denoiser.py` bridges via NumPy.

## 7. Legacy turbo path (preserved)

The original turbo update `new_input = channel + β·bp_ext + α·src_ext` is kept
as `ep_mode=False` (knobs `alpha`, `beta`). It is the practical decoder's
non-accumulating equivalent of fractional EP (best known: `α=β=0.1`, `σ=0.3`).
Design notes: `adaptive_sigma_review.md`; context handoff: `HANDOFF.md`.
