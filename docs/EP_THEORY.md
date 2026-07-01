# EP_THEORY — Expectation-Propagation reformulation of the LDPC + score-denoiser decoder

> **Status**: theory / spec document (Task 1). No code is changed by this file.
> This document is the *reference specification* for every subsequent code change
> in the Pure-EP alignment effort. Where the current code and this theory
> disagree, **this theory is authoritative** and the code is what must move.
>
> **Scope reminder**: the goal is *algorithmic fidelity to Expectation
> Propagation*, not BLER/BER. Wherever EP fidelity and performance conflict, EP
> fidelity wins. Every approximation is named explicitly so it can be tracked.

---

## 0. Notation and domains

We decode one 5G-LDPC codeword carrying a Fashion-MNIST image.

| Symbol | Meaning | Domain |
|---|---|---|
| `x ∈ {0,1}^n` | full LDPC codeword bits (graph domain, `n = n_ldpc` after 5G expansion) | bit |
| `x_p ⊂ x` | the `K_payload` payload bits that encode the image (`k_payload = n_pixels · bpp`) | bit |
| `s ∈ [0,255]^{n_pix}` (or `[0,1]` normalized) | the grayscale image pixels | pixel (continuous) |
| `y` | AWGN channel observations (received symbols → channel LLRs) | real |
| `H` | LDPC parity-check matrix; code constraint is `Hx = 0 (mod 2)` | — |

**Bit ↔ pixel map.** The payload bits are a *deterministic* function of the
image: each pixel `s_j ∈ {0,…,255}` is the big-endian `bpp`-bit integer of its
bit block, `s_j = Σ_i 2^{bpp−1−i} · x_{p,(j,i)}`. Write this map `s = P(x_p)`
and its (many-to-one only through quantization) inverse view `x_p = B(s)`.
The source prior lives on `s`; the code and channel factors live on `x`.

**Exponential families used as the EP approximating family.**

* Bit domain: **Bernoulli**, parameterized by its natural parameter, the
  **log-likelihood ratio**
  `λ_i = log P(x_i = 0) / P(x_i = 1)`  (Sionna's convention; see README §6).
  A per-bit Bernoulli site is fully described by one scalar LLR.
* Pixel domain: **Gaussian**, natural parameters `(η_1, η_2) = (μ/v, −1/(2v))`,
  or equivalently the moment pair **(mean `μ`, variance `v`)** — i.e. site
  carries *both* a location and a precision.

The global EP approximation to the posterior is the fully-factorized product

```
q(x) = ∏_i Bernoulli(x_i ; λ_i)              (bit-domain view)
```

whose per-bit natural parameter `λ_i` is the **sum of all site LLRs touching
bit i** (see §3).

---

## 1. Factor graph

The exact posterior we want to approximate is

```
p(x | y) ∝ p(y | x) · 1{Hx = 0} · p_src( P(x_p) )
          └── f_ch ──┘  └ f_code ┘  └──── f_src ────┘
```

with three factors:

### f_ch — channel factor  (acts on **all** bits, bit domain)
```
f_ch(x) = p(y | x) = ∏_{i=1}^{n} p(y_i | x_i)
```
AWGN + BPSK ⇒ this factorizes per bit and each term is log-linear in `x_i`.
Its exact contribution to every bit is a fixed Bernoulli natural parameter: the
**channel LLR**. In the code this is `payload0` for payload bits, plus the
parity/filler/punctured LLRs assembled in `decoder.py:call`.

### f_code — code factor  (acts on **all** bits, bit domain)
```
f_code(x) = 1{ Hx = 0 (mod 2) } = ∏_{c} 1{ ⊕_{i∈N(c)} x_i = 0 }
```
The LDPC parity constraints. Each check `c` couples the bits in its
neighborhood `N(c)`. This is the only factor that *couples* bits.

### f_src — source factor  (acts on **payload** bits, via the pixel domain)
```
f_src(x_p) = p_src( P(x_p) )
```
The learned image prior, expressed through the score/EDM denoiser. It couples
the `bpp` bits of each pixel (and, through the CNN, pixels to each other). It is
most naturally handled in the **pixel domain** (continuous Gaussian sites) and
then pushed back to bit-domain Bernoulli sites through the bit↔pixel map.

Variable sets, summarized:

| Factor | Variables it touches | Natural site domain |
|---|---|---|
| `f_ch` | all bits `x` | Bernoulli (LLR), **exact & frozen** |
| `f_code` | all bits `x` | Bernoulli (LLR), one message per (check, bit) edge |
| `f_src` | payload bits `x_p` (through pixels `s`) | Gaussian in pixel domain → Bernoulli in bit domain |

---

## 2. Site approximations `t̃_i`

EP replaces each *intractable* factor by a product of simple site terms in the
chosen exponential family, and keeps those sites as explicit state.

### Channel sites `t̃_ch`
`f_ch` is already a product of per-bit log-linear terms, so its EP site is
**exact**:
```
t̃_ch,i(x_i) ∝ exp( λ^ch_i · [x_i = 0] ) ,   λ^ch_i = channel LLR of bit i.
```
No projection, no iteration — `t̃_ch` is fixed for the whole decode. In the
code this is `payload0` (payload) and the assembled parity/filler LLRs. **This
is EP-correct as-is.**

### Code sites `t̃_code`
For each check–bit edge `(c, i)` EP keeps a Bernoulli site
`t̃_code,c→i(x_i) ∝ exp( m_{c→i} · [x_i=0] )`, natural parameter `m_{c→i}`
(a check-to-variable LLR message). The product over checks of these sites is the
code factor's EP approximation. **These sites are exactly the sum-product BP
check-to-variable messages** — see §4.3.

### Source sites `t̃_src`
For each payload bit `i` (equivalently, in the pixel view, for each pixel `j`)
EP keeps a source site. Two equivalent representations:

* **Pixel-domain (natural):** a Gaussian site per pixel
  `t̃_src,j(s_j) ∝ exp( η_{1,j} s_j + η_{2,j} s_j^2 )`, i.e. a
  `(mean μ_j, variance v_j)` pair. This is the *honest* EP site because the
  source factor's projection is a Gaussian moment match in pixel space.
* **Bit-domain (as consumed by BP):** a per-bit Bernoulli site
  `t̃_src,i(x_i) ∝ exp( λ^src_i · [x_i=0] )`, obtained by pushing the pixel
  Gaussian through the bit↔pixel observation model.

> **KEY STATE REQUIREMENT.** EP *requires* `t̃_src` to be stored between
> iterations, because the source cavity is *the posterior with the previous
> source site divided out* (§4.1). The current code stores **no** `t̃_src`
> (only `payload_intr`), which is the root cause of the double-counting bug
> ([어긋남 1]/[어긋남 3] in §5).

---

## 3. LLR-domain algebra (Bernoulli EP arithmetic)

For Bernoulli distributions in natural (LLR) coordinates, the EP operations are
simple addition/subtraction. Let `λ = log P(x=0)/P(x=1)`.

**Product of Bernoulli terms = sum of LLRs.** Combining independent pieces of
information about a bit multiplies their Bernoulli likelihoods:
```
Bernoulli(λ_a) · Bernoulli(λ_b)  ≡  Bernoulli(λ_a + λ_b).
```
Hence the global posterior natural parameter is the **sum of all sites**:
```
λ_i  =  λ^ch_i  +  Σ_c m_{c→i}  +  λ^src_i .        (posterior LLR of bit i)
        └ t̃_ch ┘  └── t̃_code ──┘  └ t̃_src ┘
```

**Cavity = division = LLR subtraction.** Removing one factor's site from the
posterior to form its *cavity* (a.k.a. the "extrinsic"/"leave-one-out"
distribution that factor should not see its own message) is division:
```
q^{\f}(x_i) = q(x_i) / t̃_f,i(x_i)   ⟺   λ^{\f}_i = λ_i − λ^f_i .
```

So, concretely:

| EP operation | Bernoulli/LLR form |
|---|---|
| combine information (posterior) | `λ = Σ_f λ^f` (add all sites) |
| cavity for factor `f` | `λ^{\f} = λ − λ^f` (subtract that site) |
| new site for factor `f` | `λ^f_new = λ^{proj} − λ^{\f}` (project − cavity) |
| posterior update | `λ_new = λ^{\f} + λ^f_new` (cavity + new site) |

This table is the entire discipline the code must follow. The last two rows say
**a site update is a replacement, not a damped addition onto the channel LLR.**

---

## 4. Per-factor EP cycle

Each EP factor update is the four-step loop: **cavity → tilted → projection →
site update**. We instantiate it for each factor.

### 4.1 Source factor (the factor that must change most)

Let `λ` be the current global posterior LLR over payload bits and `λ^src` the
*stored* previous source site.

**(1) Cavity.** Divide the source site out of the posterior:
```
λ^{\src} = λ − λ^src_prev .
```
Interpretation: `λ^{\src}` is *everything the rest of the graph (channel + code)
believes about the payload bits, with the source's own previous message
removed.* **This is the input the denoiser must see** — not the full posterior.

> **This is the crux.** In the code, the denoiser is fed `BP_post`
> (= `post_payload`, the full BP posterior), and the "extrinsic" is
> `src_post − BP_post`. That would be EP-correct *only if* `BP_post` were the
> cavity `λ^{\src}`. But `BP_post` already contains the previous source
> message (it was folded into `payload_intr` and re-fed to BP), so feeding it
> to the denoiser double-counts the source. **The correct cavity is
> `BP_post − λ^src_prev`.**

**(2) Tilted distribution.** Multiply the cavity by the true source factor:
```
p̃(x_p) ∝ q^{\src}(x_p) · f_src( P(x_p) ) .
```
In the pixel domain this reads: take a Gaussian pixel prior `N(μ_cav, v_cav)`
induced by the cavity, and multiply by the image prior `p_src(s)`. This tilted
distribution is the object whose moments we want.

**(3) Projection (moment matching).** Project the tilted distribution back onto
the approximating family by matching moments:
```
q^{proj} = proj[ p̃ ] = argmin_{q∈family} KL( p̃ ‖ q ).
```
For a Gaussian family this means computing the **posterior mean and variance of
the pixels under the tilted distribution**:
```
μ^{proj}_j = E_{p̃}[ s_j ] ,     v^{proj}_j = Var_{p̃}[ s_j ] .
```

> **This is exactly what a score/EDM denoiser amortizes.** With a Gaussian
> cavity `s ≈ N(μ_cav, σ²)`, Tweedie's formula gives the posterior mean of the
> clean image as
> `E[s | μ_cav] = μ_cav + σ² ∇ log p_σ(μ_cav) = D(μ_cav ; σ)`,
> and the EDM preconditioned network computes exactly
> `D(x;σ) = c_skip·x + c_out·F_θ(c_in·x; c_noise)` (see
> `score_denoiser/networks.py:441`). So **the denoiser = the source-factor
> projection, evaluated at the cavity mean, with σ = the cavity standard
> deviation.** The posterior *variance* is likewise available from Tweedie
> (`v^{proj} = σ²(1 + σ² ∇² log p_σ)`, i.e. `σ²·∂D/∂x`), but the current code
> discards it.

**(4) Site update (replacement).** The new source site is projection ÷ cavity:
```
λ^src_new = λ^{proj} − λ^{\src} ,
```
and the posterior is updated by *replacing* the old site:
```
λ_new = λ^{\src} + λ^src_new  (= λ^{proj}) .
```
Store `λ^src_new` for the next iteration's cavity. Optional damping is applied
*to the site* (`λ^src ← (1−ρ)λ^src_prev + ρ λ^src_new`), never as an additive
correction onto the channel LLR.

**Bit ↔ pixel bridging (the amortization approximations).** Two conversions in
`source_prior.py` implement the pixel↔bit boundary of steps (1)–(3):

* `llr_to_soft_field` : cavity bit-LLRs → soft pixel mean `μ_cav`
  = `Σ_i 2^{bpp−1−i} σ(λ_i)`. **Approximation A (mean-field):** it uses the
  *expected* pixel value under independent per-bit Bernoullis, i.e. a
  point/mean summary of the cavity, and supplies **no** cavity variance.
* `soft_field_to_posterior_logits` : denoised pixel `μ^{proj}` → bit LLRs
  through a Gaussian observation model with fixed width `sigma_post`.
  **Approximation B (fixed projection width):** the pixel→bit marginalization
  uses a *constant* `sigma_post = 3.0` instead of the true projected variance
  `v^{proj}`.

### 4.2 Channel factor

`f_ch` is exact in the Bernoulli family, so its EP cycle is degenerate:

* Cavity: `λ^{\ch} = λ − λ^ch`.
* Tilted = cavity × exact per-bit likelihood.
* Projection: no information loss (the family already contains it).
* Site: `λ^ch_new = λ^ch` — unchanged.

Therefore the channel site is computed once and **frozen**, which is what
`payload0` does. **EP-correct as-is.** (This is why "channel frozen" in the
code is not a bug — it is the correct EP treatment of an exact factor.)

### 4.3 Code factor → standard sum-product BP

For the parity factor with a *fully-factorized* (per-bit Bernoulli)
approximating family, the EP cycle reduces **exactly** to loopy sum-product
belief propagation:

* The cavity for edge `(c,i)` is the product of all sites on bit `i` *except*
  `t̃_code,c→i` — i.e. the **variable-to-check message** `m_{i→c}` (bit's
  posterior LLR minus the incoming check message: `λ_i − m_{c→i}`).
* The tilted-then-projected update of `t̃_code,c→i` is the **check-to-variable
  message** `m_{c→i}` given by the parity-check (tanh / boxplus) rule.
* Matching the marginal of the single bit `i` under
  `1{parity} · ∏_{i'∈N(c)} cavity_{i'}` *is* the sum-product horizontal step.

This equivalence is Minka's result: **EP with a fully-factorized Gaussian/
discrete approximation on each factor recovers loopy belief propagation**
(T. Minka, *Expectation Propagation for Approximate Bayesian Inference*, UAI
2001, §4 "Belief propagation as a special case"; see also §5 there). The
practical consequence for us: **the existing Sionna `LDPCBPDecoder` inner loop
already IS the EP treatment of the code factor.** No change is needed to the BP
core; the `msg_v2c` state it warm-starts across chunks *is* the stored code-site
state `t̃_code`. What must change is the *outer* source-factor loop.

### 4.4 Fractional / Power EP and damped site updates

Full EP (§4.1) replaces a site outright: `λ_new = λ^{\f} + (λ^{proj} − λ^{\f}) =
λ^{proj}`. With a **non-linear** source factor (the denoiser), that full
replacement can oscillate or diverge across rounds. Two standard relaxations
weaken the step; both are used here and it is worth stating them precisely
because they are *different algorithms* with *different fixed points*.

Let `θ` be the current posterior natural parameters (sum of all sites), `θ_f`
the site being refined, and `θ_{\f} = θ − θ_f` the (full) cavity.

**(A) Classical power / fractional EP** (Minka 2004, *Power EP*; Seeger 2005),
fraction `η ∈ (0,1]`:
```
cavity :  θ_{\f} = θ − η·θ_f            (remove a FRACTION η of the site)
tilted :  p̂ ∝ q_{\f} · f^η             (raise the true factor to power η)
project:  θ_proj = proj[ p̂ ]
site   :  θ_f ← θ_f + (1/η)·(θ_proj − θ)
post   :  θ ← θ + (1/η)·(θ_proj − θ)
```
This minimizes the **α-divergence with α = η** in each local projection
(Minka 2005, *Divergence measures and message passing*); `η = 1` is EP (KL),
`η → 0` tends to the variational/mean-field update. Because the *tilted*
distribution uses `f^η`, the fixed points depend on `η` (they coincide with EP
only when the family represents the tilted exactly). Applying this to the
source factor requires **tempering the denoiser** to the power-`η` prior; since
`∇log p^η = η·∇log p`, a first-order tempered denoiser is
`D_η(x;σ) ≈ x + η·(D(x;σ) − x)`. (This exact-power variant is documented as a
future option; the code does **not** temper the denoiser today.)

**(B) Damped EP** (the update this codebase implements), damping `ρ ∈ (0,1]`:
```
cavity :  θ_{\f} = θ − θ_f              (remove the FULL site, fraction 1)
tilted :  p̂ ∝ q_{\f} · f               (the FULL factor, power 1)
project:  θ_proj = proj[ p̂ ]           (the full-EP projection)
full   :  θ_f^{full} = θ_proj − θ_{\f}
site   :  θ_f ← (1−ρ)·θ_f + ρ·θ_f^{full}
post   :  θ ← (1−ρ)·θ + ρ·θ_proj
```
The cavity and tilted are the *full-EP* ones; only the site step is a convex
combination. Consequently **damped EP has exactly the full-EP fixed points**
(`θ_f^{full} = θ_f ⇔ θ_proj = θ`), and `ρ` affects only the convergence path /
stability. This is the sense in which "fractional" here shares EP's solutions.

> **What the code calls `full_ep` / `fractional_ep`.** Both are the damped-EP
> form (B), with a per-factor split of the damping:
> - **source power `α_ep`** damps the source site step (`ρ = α_ep`);
> - **code power `β_ep`** tempers how much of BP's freshly-added code evidence
>   enters the *source cavity* this round (a fractional treatment of the code
>   factor's evidence — **Approx E**), so that the denoiser sees
>   `cavity = channel_site + β_ep·code_added` instead of the full
>   `BP_post − src_site`.
>
> `full_ep` fixes `α_ep = β_ep = 1` ⇒ pure replacement and the exact §4.1 cavity
> `BP_post − src_site`; it is the **alignment target** (performance-agnostic).
> `fractional_ep` allows `α_ep, β_ep < 1` purely to stabilize the non-linear
> denoiser. The exact code equations are:
> ```
> channel_site = payload0                               (frozen, §4.2)
> A_bp         = channel_site + src_site                 (BP prior; §3 site product)
> BP_post      = BP(A_bp)                                (= channel + code + src)
> code_added   = BP_post − A_bp                          (code evidence this round)
> cavity_src   = channel_site + β_ep · code_added        (= BP_post − src_site when β_ep=1)
> src_full     = denoiser(cavity_src; σ)                 (= src_post − cavity_src)
> src_site     ← (1 − α_ep)·src_site + α_ep · src_full   (damped source site)
> ```
> The code implements these verbatim (`decoder.py`, `ep_mode=True`); it is
> damped EP (B), **not** the tempered-factor power EP (A). Divergence of the
> `full_ep` path is detected (site LLR magnitude blow-up) and logged with a
> recommendation to switch to `fractional_ep`.

### 4.5 BP iterations are a refinement schedule, not a projection to converge

A tempting but **incorrect** framing is that the code factor is a single
"projection" which ideal EP should obtain by running loopy BP to convergence, so
that a fixed iteration count would be a "truncated projection". That is not the
structure of this factor graph. As §4.3 (Minka 2001 §4) makes precise, **each
parity check is its own factor**, and each sum-product BP iteration *is* an EP
refinement step of those check-factor sites (`msg_v2c`). There is no monolithic
"code projection" — there is a set of check-factor sites being refined.

Consequently the number of BP iterations per chunk is a **refinement schedule**,
not an approximation. `iters=10` vs `iters=52` is no more "an approximation of" a
converged run than running EP for 3 sweeps is "an approximation of" 5 — they are
different points along the same refinement, both legitimate intermediate EP
states. (This corrects an earlier draft that listed a fixed count as "Approx G";
there is no such approximation, and it has been removed from the approximation
tables.)

The `decoder.py::_run_bp_chunk` modes are therefore all *schedule choices*, not
an exact-vs-approximate pair:

* **Fixed** (`bp_convergence=False`, default): run the chunk's `iters`
  check-factor refinement steps.
* **Run-to-settle** (`bp_convergence=True`): keep refining the check factors until
  their messages settle (`max|Δ msg_v2c| < bp_conv_tol`, cap `bp_max_iter`).
  Because `msg_v2c` is the complete BP state, one-step-at-a-time is
  **bit-identical** to a single `num_iter` call (verified, `max|Δ|=0`) — this
  changes only *how far the check factors are refined within a chunk*, i.e. the
  schedule.
* **Fixed + measure** (`bp_track_delta=True`): run `iters` steps but record the
  final message settledness (diagnostic only).

Per-chunk stats are in `decoder.last_bp_stats` (`bp_iters_used`, `bp_final_delta`,
`bp_converged`), optionally logged (`bp_conv_verbose`).

### 4.6 Factor refinement scheduling and convergence dynamics

The check factors and the source factor sit at the **same level** of the factor
graph (§1), so a decode is just a *schedule* over how much to refine each before
re-reading the others; **schedule choice governs the outer loop's convergence
dynamics**, with no schedule being an approximation of another. Empirically,
over-refining the check subgraph while the source site is held fixed can drive
the check messages into a **limit cycle**, which refining in smaller alternating
steps avoids; the confirmed default `bp_schedule=[2]*15` is the measured sweet
spot. Full setup, trajectories, the oscillation reframing, and the SNR sweep are
in [`EP_SCHEDULING_EXPERIMENT.md`](EP_SCHEDULING_EXPERIMENT.md).

> **Pure EP's natural terminus (Minka 2001 §3.3).** With `[2]*15` the site does
> not reach a true fixed point (`Δsite → 0`) but a **small-amplitude stable
> orbit**. This is expected of *pure* EP with a non-exponential-family factor:
> the denoiser lies outside the Bernoulli/Gaussian approximating family, so the
> moment-matching stationarity conditions cannot be met exactly and the site
> settles into a bounded orbit rather than a point. We report this honestly and
> do **not** damp it — damping would trade EP fidelity for a cosmetically
> stationary site.

---

## 5. The four misalignments, mapped to this theory

The current update (from `decoder.py:call`, README §1):

```
bp_ext    = BP_post  − payload_intr          # post − a-priori-input
src_ext   = src_post − BP_post               # denoiser posterior − denoiser input
new_input = channel  + β·bp_ext + α·src_ext  # payload_intr for next chunk
```

with `src_post = denoiser(BP_post, σ)` and `payload_intr = payload0 + β·bp_ext +
α·src_ext`.

### [어긋남 1] Cavity double-counts the source (self-message feedback)
The denoiser is fed the **full** BP posterior `BP_post`, and `src_ext =
src_post − BP_post`. EP requires the source factor to be fed its **cavity**
`λ^{\src} = BP_post − λ^src_prev` (§4.1 step 1). Because `BP_post` already
contains the previous source message (it re-entered BP via `payload_intr`), the
denoiser sees its own prior output → the source information is counted twice.
This violates the BP "a node never sees its own message back" rule. **Fix
(Task 3+):** store `λ^src_prev`, form the true cavity, feed *that* to the
denoiser.

### [어긋남 2] Update is damped addition, not site replacement
EP says `posterior = cavity × new_site`, i.e.
`λ_new = λ^{\f} + λ^f_new` (§3, §4.1 step 4). The code instead builds
`new_input = channel + β·bp_ext + α·src_ext`, a *heuristic linear
recombination* onto the frozen channel LLR with turbo weights `α, β`. This is
turbo-decoder damping, not an EP site replacement. In EP, damping (if used) is
applied to the *site* (`λ^src ← (1−ρ)λ^src_prev + ρ·λ^src_new`), and the
posterior is reassembled as *cavity + site*, not *channel + weighted
extrinsics*. **Fix (Task 4+):** replace the additive recombination with a site
replacement; keep the turbo path under `legacy_turbo` for comparison.

### [어긋남 3] No explicit site storage
Only `payload_intr` is carried between chunks. EP needs each factor's site
(`t̃_ch`, `t̃_code`, `t̃_src`) held separately (§2). `t̃_ch` exists implicitly
(`payload0`) and `t̃_code` exists implicitly (`msg_v2c`), but **`t̃_src` is not
stored at all** — which is *why* [어긋남 1] happens (you cannot form the source
cavity without `λ^src_prev`). **Fix (Task 3):** add explicit `λ^src` state.

### [어긋남 4] Sites carry mean only, not (mean, precision)
The honest source site is a **Gaussian in pixel space** with a variance
(§2, §4.1 step 3). The denoiser returns only a posterior *mean* → bit LLRs; the
projected **variance `v^{proj}` is discarded**, and the pixel→bit conversion
uses a **fixed `sigma_post`** (Approximation B) instead. Relatedly, the cavity
fed to the denoiser has no propagated variance: `σ` is chosen by a
syndrome-ratio heuristic scheduler, whereas EP dictates **σ = the cavity
standard deviation** (Approximation A supplies no cavity variance at all). So
first- and second-order information are decoupled from EP. **Fix (Task 5+):**
propagate a precision with each site; drive the denoiser `σ` from the cavity
variance; recover `v^{proj}` from Tweedie and use it in the pixel→bit
projection. Where a variance is still not recoverable, record it as a named
approximation.

---

## 6. Summary: the target EP source-factor cycle (spec for later tasks)

Pseudocode the code must converge to (payload sub-block; channel + code handled
by the BP core, which is already EP-correct):

```
# stored state across chunks:
#   λ^src  — source site (bit-domain LLR), init 0
#   optionally v_src — source site precision (pixel domain)

BP_post = BP( channel + λ^src )            # code+channel posterior (BP = EP on f_code)

# (1) cavity: remove previous source site
λ_cav   = BP_post − λ^src                  # EP division  (NOT BP_post itself)

# (2)-(3) tilted + projection via the denoiser (Tweedie = source projection)
μ_cav, σ_cav = llr_to_pixels(λ_cav)        # cavity mean (+ variance when available)
μ_proj       = D(μ_cav ; σ = σ_cav)        # posterior mean  (EDM denoiser)
v_proj       = σ_cav² · ∂D/∂x              # posterior variance (Tweedie); else named approx
λ_proj       = pixels_to_llr(μ_proj, v_proj)

# (4) site update = projection − cavity  (with optional site-space damping ρ)
λ^src_new = λ_proj − λ_cav
λ^src     ← (1−ρ)·λ^src + ρ·λ^src_new      # ρ=1 ⇒ pure EP

# next posterior is reassembled as cavity + new site inside BP( channel + λ^src )
```

Contrast the two update laws side by side:

```
legacy_turbo (current):   new_input = channel + β·(BP_post − payload_intr) + α·(src_post − BP_post)
pure_ep      (target):    λ^src    ← λ_proj − (BP_post − λ^src) ;  BP fed channel + λ^src
```

Every later task cites the section it implements (order as actually executed):

* **Task 3** — introduce explicit `λ^src` state + the true cavity
  `BP_post − λ^src`, gate legacy path under `ep_mode` ([어긋남 1], [어긋남 3];
  §4.1 step 1, §5).
* **Task 4** — realize the denoiser as the source-factor projection; carry a
  precision *slot* with the source site (fixed 2nd moment today; Tweedie
  `v^{proj}` later) ([어긋남 4]; §4.1 step 3).
* **Task 5** — replace damped *addition* with an EP site *update*: `full_ep`
  (pure replacement) vs `fractional_ep` (damped EP, §4.4); fold `channel_site`
  into the site product ([어긋남 2]; §3, §4.1 step 4, §4.4).
* **Task 6** — end-to-end CLI wiring / diagnostics.
* **Task 7** — documentation finalization (`docs/EP_SYSTEM_BLOCK.md`,
  `docs/EP_APPENDIX.md`).
* **Tasks 1–2** — this document and the migration/diagnostic mapping.

---

### References

* T. P. Minka, *Expectation Propagation for Approximate Bayesian Inference*,
  UAI 2001 — §4 shows loopy belief propagation is the special case of EP with a
  fully-factorized approximating family (our §4.3).
* T. P. Minka, *A family of algorithms for approximate Bayesian inference*, PhD
  thesis, MIT, 2001 — cavity/tilted/projection/site cycle (our §4).
* Y. Efron, *Tweedie's formula and selection bias*, JASA 2011 — posterior mean
  (and variance) of a Gaussian-corrupted signal via the score (our §4.1 step 3).
* T. Karras et al., *Elucidating the Design Space of Diffusion-Based Generative
  Models* (EDM), NeurIPS 2022 — the `D(x;σ) = c_skip·x + c_out·F_θ`
  preconditioning used by `score_denoiser/networks.py` as the amortized
  source projection.
```
