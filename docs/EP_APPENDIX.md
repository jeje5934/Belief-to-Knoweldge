# EP_APPENDIX — probabilistic derivation of the EP-aligned decoder

> **Status**: final deliverable (Task 7), paper-appendix form. This is the
> complete probabilistic derivation of the decoder as **actually built**
> (`ep_mode=True`, Tasks 3–6). It refines the project's original turbo framing
> into Expectation Propagation and states every approximation explicitly, for
> information-theoretic honesty.
>
> **Cross-references**: derivation sketch [`EP_THEORY.md`](EP_THEORY.md); the
> concrete fix [`EP_MIGRATION_PLAN.md`](EP_MIGRATION_PLAN.md); diagnostics
> [`EP_DIAGNOSTICS.md`](EP_DIAGNOSTICS.md); the block diagram
> [`EP_SYSTEM_BLOCK.md`](EP_SYSTEM_BLOCK.md). Primary reference: T. Minka,
> *Expectation Propagation for Approximate Bayesian Inference*, UAI 2001
> (§3 = EP; §4 = loopy BP as the fully-factorized special case of EP).
>
> **Code map**: `decoder.py::LDPC5GDecoder_soft.call` (EP loop),
> `decoder.py::LDPC5GDecoder_soft._ep_diagnostics`,
> `source_prior.py::SourcePriorDenoiser.{forward,llr_to_soft_field,
> soft_field_to_posterior_logits,compute_source_extrinsic,
> projected_pixel_precision}`, `denoiser.py::SoftDenoiser.call`,
> `score_denoiser/networks.py::EDMPrecond.forward`.

---

## A.1 Posterior factorization (three factors)

We transmit a Fashion-MNIST image `s ∈ {0,…,255}^{n_pix}` as the payload bits
`x_p` of a 5G-LDPC codeword `x ∈ {0,1}^n`, over an AWGN channel with BPSK, and
receive channel LLRs `y`. The payload bits are a deterministic big-endian
bit-plane encoding of the pixels, `s = P(x_p)` with
`s_j = Σ_{i=0}^{bpp-1} 2^{bpp-1-i} x_{p,(j,i)}` (`source_prior.py`: `bit_weights`,
`bit_masks`, `pixel_values`).

The posterior over the codeword factorizes into three factors:

```
p(x | y) ∝ p(y | x) · 1{Hx = 0 (mod 2)} · p_src( P(x_p) )
          └─ f_ch ─┘   └───── f_code ─────┘   └──── f_src ────┘
```

* **`f_ch(x) = p(y|x) = ∏_i p(y_i|x_i)`** — channel factor. AWGN+BPSK ⇒ it
  factorizes per bit, each term log-linear in `x_i`.
* **`f_code(x) = 1{Hx = 0} = ∏_c 1{⊕_{i∈N(c)} x_i = 0}`** — the LDPC parity
  constraints; the only factor that *couples* bits.
* **`f_src(x_p) = p_src(P(x_p))`** — the learned image prior over the payload
  bits, realized through a score/EDM denoiser.

This is the object EP approximates. (`EP_THEORY.md` §1.)

---

## A.2 Modeling assumptions

**Exponential-family choices (the EP approximating family).**

* *Bit domain* — **Bernoulli**, natural parameter the log-likelihood ratio
  `λ = log P(x=0)/P(x=1)` (Sionna convention; README §6). A per-bit site is one
  scalar LLR. Information combination is LLR **addition**; the cavity (division)
  is LLR **subtraction** (`EP_THEORY.md` §3).
* *Pixel domain* — **Gaussian**, moments `(mean μ, variance v)` (equivalently
  natural parameters `(μ/v, −1/2v)`). The source projection is naturally a
  Gaussian moment match in pixel space; sites are pushed to/from the bit domain
  through the bit↔pixel map.

The global approximation is the product of all sites; in LLR coordinates,

```
q(x) = ∏_i Bernoulli(x_i; λ_i),   λ_i = λ^ch_i + Σ_c m_{c→i} + λ^src_i.   (A.2.1)
       └ channel_site ┘└ code_site (BP) ┘└ src_site ┘
```

**Approximation list** (each named and used throughout; single source of truth
in `EP_SYSTEM_BLOCK.md` §5):

| tag | statement |
|---|---|
| **A** (mean-field cavity) | The bit→pixel map (`llr_to_soft_field`) propagates only the cavity's **first** moment `E[pixel]`; the cavity variance is not derived from the per-bit LLRs but supplied as the denoiser noise `σ`. |
| **C** (2nd-moment calibration) | Projected posterior **variance**: default fixed `sigma_post` (over-confident), or the real diagonal Tweedie `σ²·diag(∂D/∂x̃)` (`_tweedie_pixel_std`, Hutchinson finite-diff, no backprop). The diagonal Tweedie is principled but **insufficient** — it misses the denoiser's globally-correlated over-confidence (§A.6.4, `EP_SCHEDULING_EXPERIMENT.md` §5.2). |
| **D** (σ not from cavity variance) | The denoiser `σ` is scheduler-chosen (syndrome ratio = cheap proxy for the cavity's global per-image uncertainty), not the cavity std. Diagonal Tweedie (Approx C) is the drop-in alternative; trade-off: syndrome drops per-pixel detail, diagonal Tweedie drops inter-pixel correlation (§A.6.3). |
| **E** (fractional code evidence) | In `fractional_ep`, the code factor's fresh evidence enters the source cavity at power `β_ep` (`cavity_source = channel_site + β_ep·code_added`). |
| **F** (factorized normalizer) | The recorded source normalizer `source_logZ` factorizes over bits (`∏_bit Z_bit`) instead of the joint `Z_src`. |
| **damped-EP** | The `fractional_ep` update is **damped EP** (shares full-EP fixed points), not the tempered-factor power EP of Minka 2004; the denoiser is not tempered. |

*(Not listed: the inner-BP iteration count is a refinement **schedule**, not an
approximation — each BP step is the EP refinement of the individual check factors
(§A.4, EP_THEORY.md §4.5–4.6). An earlier draft's "Approx G" was a misconception
and has been removed.)*

**Honest bottom line.** With `ep_update="full_ep"` and were A/C/D exact, this
would be textbook EP with the code factor handled by exact BP. As built, it is
**EP-structured**: the message *topology*, cavity, and site algebra are exact;
the residual gaps are the amortization approximations A/C/D and, in
`fractional_ep`, the deliberate damping (E, damped-EP). The channel factor
(A.3.1) and the code factor (A.4) are exact.

---

## A.3 EP applied uniformly to the factor graph

EP replaces each factor `f_k` by a site `t̃_k` in the chosen family and iterates
a four-step cycle per factor (Minka 2001 §3). In natural parameters, with global
`θ = Σ_k θ_k`:

1. **Cavity** (remove the factor's own site — "leave-one-out"):
   `θ_{\k} = θ − θ_k`. In LLR: subtraction (A.2.1).
2. **Tilted** (reintroduce the *true* factor):
   `p̂_k(x) ∝ q_{\k}(x) · f_k(x)`, with normalizer `Z_k = ∫ q_{\k} f_k`.
3. **Projection** (moment match back into the family):
   `q^{new} = proj[p̂_k] = argmin_{q∈fam} KL(p̂_k ‖ q)` — i.e. match the
   sufficient statistics (Bernoulli: the bit marginal; Gaussian: mean+variance).
4. **Site update** (replacement): `θ_k ← θ^{proj} − θ_{\k}`, so the new posterior
   is `θ^{proj}`.

A fixed point is where no site changes, `θ_k = θ^{proj} − θ_{\k}` for all `k`
(equivalently, the tilted and the approximation share moments) — a stationary
point of the EP energy (Minka 2001 §3.3). The recorded `src_site_delta_l2` and
`source_logZ` (`_ep_diagnostics`) track exactly this (`EP_DIAGNOSTICS.md`).

**A.3.1 Channel factor is exact.** `f_ch` is a product of per-bit log-linear
terms, already in the Bernoulli family. Its projection loses nothing and its
site never changes: `θ^{proj}_ch = θ_ch = ` channel LLR. Hence
`channel_site = payload0` is computed once and **frozen**
(`decoder.py::call`, `channel_site = x1_sys[:, :k_payload]`). Freezing it is not
a heuristic — it is the exact EP treatment of an exact factor (§4.2).

The remaining two factors give the two nontrivial messages, A.4 and A.5.

---

## A.4 Code-to-source message = standard belief propagation

Consider EP on the parity factor `f_code = ∏_c 1{⊕_{i∈N(c)} x_i = 0}` with the
**fully-factorized** Bernoulli family (one independent site per bit). Minka
(2001) §4 shows this specializes to loopy sum-product BP; we restate it for the
per-edge sites.

Keep one site per check–bit edge, `t̃_{c→i}(x_i) ∝ exp(m_{c→i}·[x_i=0])`. For a
single check `c`:

* **Cavity for edge `(c,i)`**: divide out that edge's site,
  `λ^{\c}_i = λ_i − m_{c→i}` — this is the **variable-to-check message**
  `m_{i→c}` (the bit's belief minus what this check told it).
* **Tilted + projection**: match the marginal of `x_i` under
  `1{parity_c} · ∏_{i'∈N(c)} Bernoulli(λ^{\c}_{i'})`. Summing the parity
  indicator over the other bits yields the **boxplus / tanh** rule, and the
  moment-matched site is the **check-to-variable message**
  ```
  m_{c→i} = 2·atanh( ∏_{i'∈N(c)\i} tanh(m_{i'→c}/2) ).            (A.4.1)
  ```
* **Site update**: `t̃_{c→i} ← m_{c→i}`; the code contribution to bit `i` is
  `code_site_i = Σ_{c∈N(i)} m_{c→i}`.

Thus **EP-on-`f_code` = sum-product BP**, and the code-to-source message that
enters the source cavity is precisely the BP posterior contribution. We do not
reimplement (A.4.1): Sionna's `LDPCBPDecoder.call` runs it, fed the prior
`channel_site + src_site`, and the persisted check-to-variable state is
`msg_v2c` (warm-started across chunks). In code (`decoder.py::call`):

```
A_bp     = channel_site + src_site                     # BP prior (§3 site product)
BP_post, msg_v2c = LDPCBPDecoder.call(A_bp, msg_v2c)    # runs (A.4.1) internally
code_added = BP_post − A_bp                             # = Σ_c m_{c→i}, this round's code evidence
```

Because BP already carries `t̃_code` in `msg_v2c`, the outer loop must **not**
re-add a code term (doing so double-counts the code factor — the β·bp_ext term
of the legacy path; `EP_MIGRATION_PLAN.md` §1.4). Pure EP reconstructs the BP
prior as `channel_site + src_site` only.

**Iteration count is a schedule, not an approximation.** (A.4.1) is not a
monolithic "projection" to be run to convergence — each parity check is its own
factor and each BP step is the EP refinement of those check-factor sites. So the
number of BP steps per chunk (`bp_schedule`, `bp_convergence`) is a **refinement
schedule**, not an approximation: `iters=10` and a run-to-settle differ as two
points along the same refinement, both legitimate intermediate EP states, not
exact-vs-truncated. Because `msg_v2c` is the complete BP state, stepping one
iteration at a time is bit-identical to the fixed call, so `bp_convergence` /
`bp_track_delta` change only the iteration count and logging, never the result.
Schedule choice instead governs the outer loop's **convergence dynamics**
(EP_THEORY.md §4.5–4.6): over-refining the check subgraph while the source site is
frozen can drive the check messages into a limit cycle, which alternating in
smaller steps may avoid. (An earlier draft mislabelled the fixed count as
"Approx G"; removed.)

---

## A.5 Source-to-code message = full EP projection

Now the source factor `f_src(x_p) = p_src(P(x_p))`. Its EP cycle is the crux of
the alignment ([어긋남 1]); we give it in full.

Let `λ` be the current payload posterior (A.2.1) and `λ^src` the stored source
site (`decoder.py::call`, `src_site`, initialized to `0`).

**(1) Cavity — remove the source self-message.**
```
λ^{\src} = λ − λ^src.                                            (A.5.1)
```
Since BP was fed `channel_site + src_site` and returns
`BP_post = channel_site + code_site + src_site`, this cavity is the exactly
computable subtraction
```
cavity_src = BP_post − src_site = channel_site + code_site,       (A.5.2)
```
i.e. "everything the channel and code believe about the payload, with the
source's own previous message removed." **This is the input the denoiser must
see** — not the full `BP_post`. Feeding `BP_post` (which already contains
`src_site`) is the source double-count the migration removed. In code
(`decoder.py::call`), with the `fractional_ep` code-temper `β_ep` (Approx E):
```
cavity_source = channel_site + b_ep * code_added         # = BP_post − src_site  when b_ep = 1
```
so `full_ep` (β_ep=1) realizes (A.5.2) exactly.

**(2) Tilted — reintroduce the source prior.**
```
p̂_src(x_p) ∝ q^{\src}(x_p) · f_src(P(x_p)).                     (A.5.3)
```
Read the cavity as a **Gaussian observation of the clean image**: the cavity's
per-pixel mean is `μ_cav = E_{q^{\src}}[pixel]`
(`source_prior.py::llr_to_soft_field`, Approx A), and its variance is taken as
`σ²` with `σ` the denoiser noise level (Approx D). Then (A.5.3) is, per pixel,
`p̂(s_j) ∝ N(s_j; μ_cav,j, σ²) · p_src(s_j)`.

**(3) Projection — moment matching (mean *and* variance).**
```
q^{proj}: μ^{proj}_j = E_{p̂}[s_j],   v^{proj}_j = Var_{p̂}[s_j].  (A.5.4)
```
The first moment is exactly what a score denoiser provides (A.6). The second
moment `v^{proj}` is available from Tweedie (A.6.3) but is **approximated by a
fixed pixel std** `sigma_post` in the bit read-out
(`source_prior.py::soft_field_to_posterior_logits`, Approx C). Converting the
projected pixel distribution back to bit marginals gives the projected bit-LLRs
`λ^{proj}` (same routine).

**(4) Site update — replacement (`full_ep`) or damped (`fractional_ep`).**
The full-EP source site is
```
src_full = λ^{proj} − λ^{\src} = src_post − cavity_src.          (A.5.5)
```
Crucially, the denoiser wrapper returns `src_post − input`
(`source_prior.py::compute_source_extrinsic`), so **evaluating it at the cavity
yields `src_full` directly** — no separate subtraction. The site is then applied
(`decoder.py::call`, `EP_THEORY.md` §4.4):
```
full_ep      (α_ep=1):  src_site ← src_full                       (pure replacement)
fractional_ep(α_ep<1):  src_site ← (1−α_ep)·src_site + α_ep·src_full   (damped EP)
```
and the posterior is reassembled as `payload_intr = channel_site + src_site`
(cavity ⊗ new site), which becomes the next BP prior. `full_ep` is the exact EP
replacement ([어긋남 2] fixed) and the alignment target; `fractional_ep` is
damped EP (A.2 damped-EP tag), sharing the full-EP fixed points and used only to
stabilize the non-linear projection.

**Evidence.** The normalizer of (A.5.3), `Z_src = ∫ q^{\src} f_src`, is the
source factor's evidence contribution. With a black-box denoiser it is
intractable jointly, so we record the factorized proxy (Approx F)
`log Z_src ≈ Σ_bit log(P_cav(0) + P_cav(1)·e^{src_full})`
(`decoder.py::_ep_diagnostics`, `source_logZ`), whose stabilization across
rounds signals convergence (`EP_DIAGNOSTICS.md` §2).

---

## A.6 Amortizing the projection with a score-based denoiser

The projection (A.5.4) requires the posterior mean of the clean image under a
Gaussian observation — exactly what Tweedie's formula gives for a score model.

**A.6.1 Tweedie's formula.** For a clean signal `s ∼ p_src` observed as
`x̃ = s + N(0, σ²I)`, with noised marginal `p_σ(x̃) = ∫ p_src(s) N(x̃; s, σ²) ds`,
the posterior mean is
```
E[s | x̃] = x̃ + σ² ∇_{x̃} log p_σ(x̃).                            (A.6.1)
```
The right-hand side is the **denoiser** `D(x̃; σ)`. The EDM preconditioned network
computes exactly this (`score_denoiser/networks.py::EDMPrecond.forward`):
```
D(x; σ) = c_skip·x + c_out·F_θ(c_in·x; c_noise),                  (A.6.2)
c_skip = σ_data²/(σ²+σ_data²),  c_out = σ·σ_data/√(σ²+σ_data²), …
```
Identifying `x̃ = μ_cav` (cavity pixel mean) and the observation noise with the
cavity variance `σ²`, the denoiser output **is** the projected posterior mean of
(A.5.4):
```
μ^{proj} = D(μ_cav; σ) = E[s | cavity].                          (first moment, EXACT given the score)
```
This is the `mu_proj = self.net(mu_cavity, sigma)` line in
`source_prior.py::forward`. So **denoiser = source-factor projection, first
moment** — the amortization that makes EP tractable here.

**A.6.2 Second moment (variance).** Differentiating (A.6.1),
```
Var[s | x̃] = σ²( I + σ² ∇² log p_σ(x̃) ) = σ² · ∂D/∂x̃.           (A.6.3)
```
So the *exact* projected variance is `v^{proj} = σ²·∂D/∂x̃` (the denoiser
Jacobian). The real diagonal 2nd moment **is now implemented** —
`SourcePriorDenoiser._tweedie_pixel_std` estimates `diag(∂D/∂x̃)` by Hutchinson
finite differences (no backprop) and feeds `√(σ²·diag)` as the per-pixel std into
the pixel→bit read-out (`tweedie_precision=True`; the fixed `sigma_post` path
remains the default). **It does not rescue full EP** (see below and
`EP_SCHEDULING_EXPERIMENT.md` §5.2).

**A.6.3 Which approximations entered.** Summarizing this section against A.2:

* **Exact**: the first-moment projection (A.6.1)/(A.6.2) — the denoiser is the
  Bayes posterior-mean estimator for the assumed Gaussian observation.
* **Approx A**: the cavity is summarized by its pixel *mean* only (`llr_to_soft_field`).
* **Approx C**: the projected 2nd moment. Two paths: a fixed `sigma_post`
  (default, over-confident), or the diagonal Tweedie `σ²·diag(∂D/∂x̃)`
  (`_tweedie_pixel_std`). The diagonal Tweedie is the principled form but is
  **empirically insufficient** — it captures per-pixel *local* sensitivity, while
  the denoiser's error is a *globally correlated* over-confidence (a wrong garment
  hallucinated from noise) that only the full off-diagonal covariance would see.
* **Approx D**: the denoiser `σ` is chosen by the **syndrome-ratio scheduler**,
  not set to the cavity std. Framed honestly: exact EP wants the cavity variance;
  the syndrome ratio (fraction of unsatisfied parity checks) is a cheap proxy for
  the cavity's **global, per-image** uncertainty (distance from the code
  manifold). The pixel-wise diagonal Tweedie 2nd moment (Approx C) is the
  drop-in alternative — the precision slot already has the right shape. **Trade-off:**
  the syndrome ratio discards per-pixel differences (one global scalar); diagonal
  Tweedie discards inter-pixel correlations — and the decisive error lives in
  those correlations (§5.2), so neither is a complete calibration.
* **Approx E / damped-EP**: in `fractional_ep`, code evidence enters the cavity
  at power `β_ep` and the site step is damped by `α_ep` — damped EP (Minka 2004
  power EP is *not* implemented; the denoiser is not tempered).

**A.6.4 Full EP does not decode; fractional EP is required.** `full_ep`
(α_ep=β_ep=1) is the *aligned* configuration — exact EP site replacement (A.5.5).
But it **decodes at BLER 1.0**: the learned denoiser is an over-confident,
un-calibrated factor, and full site trust corrupts the belief from the first
round (`EP_SCHEDULING_EXPERIMENT.md` §5.1). The principled fix (Approx C diagonal
Tweedie precision, **implemented on this branch**) does **not** rescue it (§5.2) —
the miscalibration is correlated, not diagonal.

**Why a *diagonal* precision cannot suffice — and why pure EP is out of reach.**
The denoiser's error is a *joint* one: from a noisy input it commits to the
wrong garment, so the wrong pixels are strongly **correlated**. Capturing that
would require the **full projected covariance** `Cov[s | x̃] = σ²·∂D/∂x̃`, a dense
`d×d` matrix with `d = n_pix = 784` — i.e. `d² ≈ 6.1·10⁵` entries per image
(and its inverse/product for the EP update), which is neither exposed by a
score/EDM denoiser nor tractable to estimate/propagate per chunk. The Gaussian
EP source site can only carry a diagonal (mean + per-pixel precision); the very
statistic that would calibrate this factor is the off-diagonal covariance that
the approximating family **cannot represent**. So *exact* EP for this
learned-denoiser factor is not merely unimplemented — it is **computationally and
representationally out of reach**. That is the fundamental justification for
**fractional EP**: since the site cannot be correctly *shaped* (no tractable
covariance), it must instead be *down-weighted*. What works is a damped source
site (`ep_source_power` small enough that the accumulated site stays ≈ 20–30 % of
the denoiser's full belief → BLER ≈ 0.004, matching the best turbo); the damping
is an **implicit precision discount** on the miscalibrated factor — the same
thing the legacy turbo α≈0.1 does non-accumulatively. This is the concrete form
of the project's *EP-fidelity vs performance* tension, stated openly rather than
hidden. (The sibling branch `pure-EP_ada-sigma` omits the Tweedie computation and
uses a fixed `sigma_post` with the syndrome-ratio σ; it reaches the same
conclusion without the diagonal-Tweedie machinery.)

---

### References

* T. P. Minka, *Expectation Propagation for Approximate Bayesian Inference*,
  UAI 2001. §3: the EP cavity/tilted/projection/site cycle and energy
  (our A.3, A.5). §4: loopy belief propagation as EP with a fully-factorized
  approximating family (our A.4).
* T. P. Minka, *Power EP*, MSR-TR-2004-149, 2004; *Divergence measures and
  message passing*, MSR-TR-2005-173, 2005 — power/fractional EP and the
  α-divergence view (our A.2 damped-EP tag, `EP_THEORY.md` §4.4).
* B. Efron, *Tweedie's formula and selection bias*, JASA 2011 — posterior mean
  and variance via the score (our A.6.1, A.6.3).
* T. Karras, M. Aittala, T. Aila, S. Laine, *Elucidating the Design Space of
  Diffusion-Based Generative Models* (EDM), NeurIPS 2022 — the preconditioned
  denoiser (A.6.2), `score_denoiser/networks.py`.
