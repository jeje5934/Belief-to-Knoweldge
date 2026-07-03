# EP_DIAGNOSTICS — convergence & evidence diagnostics for the EP decoder

> **Status**: implementation + theory note (Task 6). Documents the EP
> convergence/evidence quantities computed by `decoder.py` in `ep_mode=True`,
> and the correspondence between the existing syndrome-ratio σ scheduling and
> genuine EP indicators.
>
> **Authority**: builds on [`docs/EP_THEORY.md`](EP_THEORY.md) (§3.3-style
> energy / fixed-point characterization, §4.1/§4.4 cycle). Section numbers
> "§N" without a file refer to `EP_THEORY.md`.
>
> **Global rule (unchanged):** these are **recorded only** — no diagnostic here
> alters the decode. The σ schedule is deliberately left untouched (requirement
> 3); EP indicators are logged *alongside* it. Performance is not a goal.

---

## 1. Background: what EP gives us at each projection

EP characterizes fixed points through two objects (Minka 2001, *Expectation
Propagation for Approximate Bayesian Inference*, §3):

1. **Per-factor normalizer `Z_i`.** Each projection forms a tilted distribution
   `p̂_i(x) = q_{\i}(x)·f_i(x)` with normalizer
   `Z_i = ∫ q_{\i}(x) f_i(x) dx = E_{q_{\i}}[ f_i ]`. The product of the `Z_i`
   (with the site-partition corrections) approximates the model evidence
   `p(D)`. `Z_i` is the "evidence contribution" of factor `i` given its cavity.

2. **Energy / stationarity.** An EP fixed point is a stationary point of the EP
   (Bethe-like) energy function (Minka 2001 §3.3). Operationally, a fixed point
   is reached exactly when **every site stops changing**: `t̃_i^{new} = t̃_i`
   for all `i`. The site-change magnitude is therefore the primary, exact
   convergence indicator; it needs no approximation.

> **Caveat (stated up front):** EP is **not** a descent method — its energy is
> not guaranteed to decrease monotonically. So we track energy/normalizer for
> *inspection* and use the (rigorous) **site-change** and **site-magnitude**
> quantities for the convergence/divergence *decisions*.

---

## 2. Quantities computed (`decoder.py:_ep_diagnostics`)

For the **source factor** each round (chunk), with cavity LLR `cavity_source`
(§4.1 step 1), full-EP projection message `src_full = src_post − cavity`
(§4.1 step 4), and updated site `src_site` (§4.4):

| key | definition | EP meaning |
|---|---|---|
| `src_site_delta_l2` | `mean_b ‖src_site − src_site_prev‖₂` | **stationarity** — → 0 at a fixed point (Minka §3.3). Primary convergence measure. |
| `src_site_l2` | `mean_b ‖src_site‖₂` | site magnitude (how much the source factor is currently asserting). |
| `src_site_max_abs` | `max_bit |src_site|` | **divergence guard** — blow-up ⇒ full_ep diverging. |
| `source_logZ` | `mean_b Σ_bit log Z_bit` | **source normalizer `Z_src`** (log-domain) — the source factor's per-round evidence contribution. |
| `source_free_energy` | `− source_logZ` | source free-energy proxy (an energy term to watch for stabilization). |

### The source normalizer `Z_src`

`f_src` couples the payload bits jointly (through the image), so its exact
`Z_src = E_{q_{\src}}[f_src]` is intractable with a black-box denoiser. We use
the **Bernoulli tilted normalizer of folding the projected source message into
the cavity**, per bit:

```
Z_bit = P_cav(bit=0) + P_cav(bit=1)·exp(src_full)      (src_full = source site message, LLR)
      = E_{cavity}[ source factor message ]            (per bit)
log Z_bit = logaddexp( log P_cav(0),  log P_cav(1) + src_full )     (stable log-sum-exp)
source_logZ = mean over batch of  Σ_bit log Z_bit
```

with `P_cav(bit=1) = sigmoid(cavity_source)` in the denoiser's sign convention
(`sigmoid(llr)=P(1)`, see `source_prior.py`). This is the exact EP normalizer
**if the source factor acted as the per-bit projected Bernoulli message** — i.e.
it is the projection-consistent, factorized proxy of `Z_src`.

> **Approx F (factorized normalizer).** `source_logZ` treats the source message
> as independent per-bit Bernoulli likelihoods (`∏_bit Z_bit`); the true
> `Z_src` includes the joint pixel/image coupling. This is the same
> mean-field/factorized spirit as Approx A (§4.1). It is a consistent *trend*
> indicator, not a calibrated evidence value.

### What to read from them

- **Converging**: `src_site_delta_l2` shrinks across chunks; `source_logZ`
  settles. Both are the signatures of approaching an EP fixed point.
- **Diverging / oscillating**: `src_site_max_abs` blows up, or
  `src_site_delta_l2` **grows** round-over-round (see §4).

---

## 3. Syndrome ratio ↔ EP indicators

The σ scheduler (`syndrome_sigma_schedule.py`) picks the denoiser σ from the
**syndrome ratio** `= (# unsatisfied parity checks)/(# checks)` of the hard
decision. Its EP interpretation:

| syndrome-ratio picture | EP picture |
|---|---|
| fraction of parity checks violated by `sign(x̂)` | a cheap, hard-decision proxy for how far the current posterior mass sits **off the code manifold** — i.e. inversely related to the **code factor's normalizer `Z_code`** (§4.3). Low syndrome ⇔ high `Z_code` (posterior consistent with `1{Hx=0}`). |
| high syndrome ⇒ pick a different σ | high syndrome ⇒ low code-evidence ⇒ BP posterior less reliable ⇒ the scheduler compensates by changing denoiser strength. |

So **syndrome ratio is a hard-decision proxy for the code factor's evidence**
`Z_code`. The EP-principled replacements, none of which are wired in (σ
untouched by design):

1. **σ from the cavity variance** (Approx D, §4.1 step 3): the denoiser noise
   level should be the cavity std, not a syndrome lookup. This is the honest
   EP source of σ.
2. **A soft code normalizer** in place of the hard syndrome count: the tilted
   mass of the parity factors (a soft `Z_code`) rather than `sign(x̂)`-syndrome.
3. **Schedule / stop on EP indicators**: use `src_site_delta_l2` (stationarity)
   or `source_logZ` (evidence) to decide when to stop or how hard to denoise,
   instead of the syndrome ratio.

These are recorded now (so a future task can compare syndrome-ratio decisions
against the EP indicators on the same runs) but the scheduling logic is
unchanged. `source_logZ`/`src_site_delta_l2` are logged **in parallel** with
`mean_syndrome_ratio` in `last_chunk_diagnostics` and in the aggregated
`ep_metrics` block of `summarize_chunk_diagnostics`.

---

## 4. Divergence / non-convergence detection

`decoder.py` flags two non-convergence modes each round and logs the first
occurrence (requirements 2, 4):

| condition | meaning | action |
|---|---|---|
| `src_site_max_abs > ep_divergence_llr` (default 1e4) | site LLR blow-up | `ep_diverged=True`; if `ep_update=="full_ep"` also `ep_recommend_fractional=True` and a `[EP] WARNING …` line recommending `damped_ep` with a smaller `ep_source_power`. |
| `src_site_delta_l2 > 1.5·(prev round)` and `> 1.0` | site-change **growing** ⇒ oscillation / non-convergence | same flags + warning. |

This connects the divergence detection directly to the **full_ep → damped_ep
switch criterion** of §4.4: `full_ep` (pure replacement) can oscillate/diverge
because the denoiser is non-linear; when it does, the diagnostics say so and
point at the damped `damped_ep` fix. Because EP energy is not monotone
(§1 caveat), we do **not** flag on "energy increased" alone — the robust signals
are site blow-up and site-change growth.

### Recorded fields

Per chunk in `last_chunk_diagnostics` (EP mode only): `src_site_l2`,
`src_site_delta_l2`, `src_site_max_abs`, `source_logZ`, `source_free_energy`,
`ep_diverged`, `ep_delta_growing`, `ep_recommend_fractional`, `ep_update`.
Aggregated per chunk in `summarize_chunk_diagnostics` under `ep_metrics`
(`mean_*` of the floats + `ep_diverged` = any). Legacy mode records none of
these (`ep_update=None`), so existing outputs are unchanged.

---

## 5. Where they show up

- **`decoder.last_chunk_diagnostics`** — raw per-chunk dicts with the EP fields.
- **`decoder.last_chunk_summary`** (via `summarize_chunk_diagnostics`) — an
  `ep_metrics` sub-dict per chunk, so `experiment.py` / `plot_comparison.py`
  JSON already carries them when `ep_mode=True`.
- **stdout** — one `[EP] update=…` config line per decoder instance, and one
  `[EP] WARNING …` line on the first divergence/oscillation event.
