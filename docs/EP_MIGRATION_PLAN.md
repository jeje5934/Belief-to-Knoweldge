# EP_MIGRATION_PLAN — fixing [어긋남 1] cavity double-counting

> **Status**: design / spec document (Task 2). **No code is changed by this
> file.** It specifies exactly how Tasks 3–4 will remove the source-factor
> double-counting and introduce explicit EP site state.
>
> **Authority**: [`docs/EP_THEORY.md`](EP_THEORY.md) is the theory of record;
> this file is the concrete migration derived from it. Section numbers like
> "§4.1" refer to `EP_THEORY.md`.
>
> **Global rule (unchanged):** EP fidelity beats BLER/BER. Every approximation
> is named. The legacy turbo path is preserved (`legacy_turbo`), not deleted.

---

## 1. Line-level trace: why the current code double-counts the source

All line numbers are `decoder.py` at its current revision unless noted.

### 1.1 The state that is carried between chunks

```
decoder.py:216   payload0 = x1_sys[:, :k_payload]     # channel site  t̃_ch  (frozen)  ✓
decoder.py:219   payload_intr = payload0              # a-priori LLR fed to BP
decoder.py:220   curr_msg_v2c = msg_v2c               # code site  t̃_code  (warm-start) ✓
```

Note what is **and is not** carried:

* `payload0` — the channel site `t̃_ch`. Frozen. **EP-correct** (§4.2).
* `curr_msg_v2c` — the code site `t̃_code`, warm-started across chunks.
  **EP-correct** (§4.3: BP messages *are* the code sites).
* `payload_intr` — a *single blended a-priori vector*. It is **not** a site.
  There is **no** `src_site` variable anywhere. This is [어긋남 3], and it is
  the enabling cause of the double-count below.

### 1.2 The feedback loop that double-counts

Trace one full chunk `idx ≥ 1` (the bug is absent only at `idx = 0`):

```
decoder.py:237   x1_stage = concat([payload_intr, crc_and_rest])   # payload_intr already
decoder.py:238   llr_bp   = concat([x1_stage, z_short, x2_par])    #   holds α·src_ext_{idx-1}
decoder.py:240   x_hat, curr_msg_v2c = LDPCBPDecoder.call(llr_bp, …)  # BP propagates it
decoder.py:258   post_payload = x_hat[:, :k_payload]               # BP_post  CONTAINS the
                                                                   #   previous source msg
decoder.py:261   src_ext = self._denoiser(post_payload, …)         # denoiser INPUT = BP_post
   └─ source_prior.py:118  soft_img  = llr_to_soft_field(llr=post_payload)
   └─ source_prior.py:126  denoised  = net(soft_img, sigma)
   └─ source_prior.py:129  posterior = soft_field_to_posterior_logits(denoised)   # = src_post
   └─ source_prior.py:130  src_ext   = posterior − llr = src_post − post_payload   # = src_post − BP_post
decoder.py:284   payload_intr = payload0 + b·bp_ext + a·src_ext    # re-inject source AGAIN
```

The defect, stated precisely:

1. `payload_intr` at the top of chunk `idx` already contains the source
   contribution from chunk `idx−1` (added at `decoder.py:284` in the previous
   iteration, via the `a·src_ext` term).
2. BP at `decoder.py:240` propagates that a-priori LLR, so **`BP_post`
   (`post_payload`) already includes the previous source message**.
3. The denoiser is fed `post_payload` at `decoder.py:261`/`:269`, i.e. it sees
   an input that **already contains its own prior output**. In EP terms the
   denoiser (source factor) is being shown its own previous site instead of the
   *cavity* with that site removed (§4.1 step 1 requires
   `cavity = BP_post − src_site_prev`).
4. `src_post = D(BP_post)` therefore re-derives source belief on top of source
   belief, and `decoder.py:284` adds it back in. **The source information is
   counted twice** — the LDPC-BP "never send a node its own message" rule,
   applied to the source factor, is violated.

### 1.3 Why `idx = 0` is accidentally correct

At `idx = 0`, `payload_intr == payload0` (`decoder.py:219`), i.e. the source
site is implicitly zero. So `BP_post` contains **no** source message and the
denoiser input `BP_post` *equals* the true cavity `BP_post − 0`. The
double-count is a strictly-`idx ≥ 1` phenomenon. (This matches the README note
"In the FIRST chunk, `payload_intr == channel`.") EP formalizes that accident:
**the site starts at 0**, so round 0 is the one round where "feed BP_post" and
"feed the cavity" coincide.

### 1.4 A note on `b·bp_ext` — a *second*, latent double-count

`bp_ext = post_payload − payload_intr` (`decoder.py:272`) is the code factor's
own extrinsic, and `decoder.py:284` re-injects `b·bp_ext` into the next BP
input. But BP fed with a warm-started `msg_v2c` **already** reproduces the code
posterior internally; adding `b·bp_ext` externally re-injects the code factor's
message on top of itself. This is the code-factor analogue of the source
double-count. Pure EP needs neither term added by hand: BP + warm-start carries
`t̃_code`, and the source site carries `t̃_src`. → in pure EP, `b·bp_ext`
disappears (equivalently `β = 0`), and `a·src_ext` is replaced by a proper site
(see §4). This is called out here because Task 4 removes it; it is a
consequence of [어긋남 2], not a separate misalignment.

---

## 2. Explicit site state design

Introduce the three EP sites as first-class state. Only `src_site` is new; the
other two already exist under different names and stay as-is.

| Site | Variable (proposed) | Domain / units | Shape | Init | Refined by | Persisted across chunks? |
|---|---|---|---|---|---|---|
| `t̃_ch` channel | `payload0` (exists) | bit-LLR | `[B, k_payload]` | channel LLR | never (exact) | yes, frozen |
| `t̃_code` code | `curr_msg_v2c` (exists) | check→var LLR msgs | `[B, num_edges]` (Sionna internal) | `None`/0 (cold) or warm-start | BP inner loop | yes, warm-started |
| `t̃_src` source | **`src_site` (NEW)** | bit-LLR (pixel-Gaussian is the honest form; Task 5) | `[B, k_payload]` | **0** (Bernoulli site = uniform ⇔ LLR 0 ⇔ EP "site = 1") | denoiser projection | **yes (this is the fix)** |

Design points:

* **`src_site` init = 0.** EP initializes every site to the identity of the
  multiplicative family (`t̃ = 1`), which in Bernoulli/LLR coordinates is
  `LLR = 0`. This makes round 0 reduce to plain BP, matching §1.3.
* **`src_site` lives on the payload sub-block only** (`[B, k_payload]`), because
  `f_src` touches only payload bits (§1 of theory). Parity/filler/punctured
  bits get no source site.
* **Posterior identity (payload sub-block).** Feeding BP the variable prior
  `payload0 + src_site` and reading its marginals yields, in LLR:
  ```
  BP_post = payload0 + code_contribution + src_site
          =  t̃_ch    +     t̃_code       +  t̃_src
  ```
  Hence the source cavity is *exactly* the subtraction
  `cavity_source = BP_post − src_site = payload0 + code_contribution`, i.e.
  "channel + code belief, source removed" (§4.1 step 1). This is the whole
  reason the fix is local: **`BP_post − src_site` is the honest cavity, no BP
  re-run needed.**
* **`src_site` is bit-domain LLR for now.** The pixel-domain Gaussian
  `(mean, precision)` form (§2, §4.1 step 3) is deferred to Task 5
  ([어긋남 4]); this plan carries only the first-order LLR. That deferral is
  **Approximation C (mean-only site)** and must be annotated in code + docs.

---

## 3. Corrected EP cycle (pseudocode)

### 3.1 Source factor cycle (the fix)

```python
# ---- persistent state (per decode) ----
payload0  = channel_LLR_payload            # t̃_ch  (frozen)
src_site  = zeros_like(payload0)           # t̃_src (NEW, init 0)
msg_v2c   = warm_start_or_None             # t̃_code

for idx, iters in enumerate(schedule):
    # ---- code-factor refinement: BP fed channel + source site ----
    bp_input_payload = payload0 + src_site          #  t̃_ch · t̃_src  (LLR add)
    llr_bp   = assemble_graph(bp_input_payload, crc_and_rest, z_short, x2_par)
    x_hat, msg_v2c = LDPCBPDecoder.call(llr_bp, num_iter=iters, msg_v2c=msg_v2c)
    BP_post  = x_hat[:, :k_payload]                 #  = payload0 + code + src_site

    if idx == last:            # final chunk: decode only, no source update
        break

    # ---- source-factor EP update (§4.1) ----
    cavity_source = BP_post - src_site               # (1) cavity  = remove prev src site
    #   feed the CAVITY (not BP_post) to the denoiser:
    src_site_new  = denoiser(cavity_source, sigma)   # (2)+(3)+(4) fused, see note ▼
    #   ▼ denoiser internally returns  src_post − input  (source_prior.py:130);
    #     with input = cavity_source this is exactly  src_post − cavity = the NEW SITE.

    # ---- site replacement, with optional site-space damping ρ (ρ=1 ⇒ pure EP) ----
    src_site = (1 - rho) * src_site + rho * src_site_new
    #   next chunk's BP_input = payload0 + src_site  (reconstruction is implicit;
    #   posterior = cavity + new_site is what BP recomputes from channel+src_site).
```

**Key realization.** Because the denoiser already computes `src_post − input`
(`source_prior.py:104-106,130`), **changing its input from `BP_post` to
`cavity_source = BP_post − src_site` makes its return value equal the EP site
update `src_post − cavity` with no other change to `source_prior.py`.** The
cavity subtraction and the site formula collapse into one edit at the call
site. (Task 3 does the subtraction + storage; Task 4 does the reconstruction /
damping and gates the legacy path.)

**σ note (forward-ref to Task 5).** `sigma` here should be the cavity standard
deviation (§4.1 step 3). Task 2/3 keep the existing scheduler-chosen `σ` and
annotate it as **Approximation D (σ not tied to cavity variance)**; Task 5
replaces it.

### 3.2 Code factor cycle (already EP-correct — for completeness)

The code factor's EP cycle *is* what Sionna's BP does internally; we do not
reimplement it, we only feed it the right prior:

```
for each check c and incident bit i:
    cavity_{i→c} = posterior_i − m_{c→i}          # variable-to-check message (leave-one-out)
    m_{c→i}      = boxplus_{i'∈N(c)\i} cavity_{i'→c}   # projected check-to-var message (new site)
    posterior_i  = (payload0_i + src_site_i) + Σ_c m_{c→i}    # reassemble = cavity + site
```

The *only* migration action for the code factor is: **feed BP the prior
`payload0 + src_site`** (§3.1) instead of the blended `payload_intr`, and **stop
adding `b·bp_ext` externally** (§1.4). `msg_v2c` warm-start already persists
`t̃_code` correctly.

---

## 4. Numerical difference vs the current turbo equations

Let `A_idx` = the a-priori payload LLR fed to BP at chunk `idx`
(`payload_intr`), `P_idx = BP_post` at chunk `idx`, `D(·)` the denoiser
posterior map, and `S_idx = src_site` after chunk `idx`.

| | **current turbo** | **pure EP (this plan)** |
|---|---|---|
| BP prior | `A_idx = payload0 + β·bp_ext_{idx-1} + α·src_ext_{idx-1}` | `payload0 + S_{idx-1}` |
| denoiser input | `P_idx` (full BP_post) | `P_idx − S_{idx-1}` (cavity) |
| denoiser return used | `src_ext = src_post − P_idx` | `src_site_new = src_post − (P_idx − S_{idx-1})` |
| state update | `A_{idx+1} = payload0 + β·bp_ext_idx + α·src_ext_idx` | `S_idx = (1−ρ)S_{idx-1} + ρ·src_site_new` |

### 4.1 When they coincide

They are **identical** iff all of:

1. `idx = 0` (so `S_{-1} = 0` ⇒ cavity `= P_0` ⇒ same denoiser input), **and**
2. `β = 0` (no external `b·bp_ext` term — pure EP has none, §1.4), **and**
3. `α = 1` and `ρ = 1` (site fully applied, undamped).

Under (1)–(3): turbo builds `A_1 = payload0 + src_ext_0 = payload0 + (src_post −
P_0)`, and EP builds `payload0 + S_0 = payload0 + (src_post − P_0)`. Same
vector. **Round 0 of pure EP is exactly a special case of the turbo update.**
This is the anchor that makes the migration safe to validate: with `β=0, α=1,
ρ=1` and a 2-chunk schedule, the *first* denoiser call must be numerically
identical between `legacy_turbo` and `pure_ep`.

### 4.2 When and how they diverge

* **`idx ≥ 1` (the double-count):** turbo feeds the denoiser `P_idx` while EP
  feeds `P_idx − S_{idx-1}`. The inputs differ by exactly the previous source
  site `S_{idx-1}`. Since `D` is nonlinear, in the small-signal regime
  `src_post_turbo ≈ src_post_EP + J_D · S_{idx-1}` (with `J_D = ∂D/∂x` the
  denoiser Jacobian). The turbo path thus **over-includes the previous source
  belief**, scaled by the denoiser gain — the quantitative signature of the
  double-count.

* **`β ≠ 0`:** turbo adds `β·bp_ext` (code extrinsic) to the BP prior every
  round; EP never does (§1.4). This makes turbo's BP prior drift from
  `payload0 + S` by `β·bp_ext`, an extra code-factor self-feedback absent in EP.

* **`α ≠ 1` / `ρ ≠ 1`:** both damp, but *differently*. Turbo scales the raw
  extrinsic `α·(src_post − P_idx)` and re-adds it to `payload0`. EP damps the
  **site** `S ← (1−ρ)S + ρ·(src_post − cavity)`, retaining the previous site's
  `(1−ρ)` share. Turbo has no `(1−ρ)S_{idx-1}` memory term — each round rebuilds
  from `payload0`.

### 4.3 Why the EP-wrong turbo still works empirically (important)

The best-known setting is `α = β = 0.1` (README §10) — i.e. **heavy
attenuation**. Because `α = 0.1`, the source contribution folded into
`payload_intr`, and hence the `S_{idx-1}` that leaks into `BP_post`, is small
(≈ `α·src_ext`, further reshaped by BP). The double-counted excess `J_D·S_{idx-1}`
is therefore an order of magnitude down. **Small `α` acts as an implicit,
crude site damping that masks the double-count** — it does not fix it. Pure EP
with `ρ < 1` achieves the same stabilizing effect *correctly* (damping the
stored site, with cavity subtraction intact). This is the predicted reason
BLER may *not* improve — and possibly regress at `ρ = 1` — when we switch to
pure EP, which is acceptable per the global rule (fidelity > performance).

---

## 5. Scheduling: code factor (BP) → source factor

EP fixed points are order-independent, but a finite-iteration implementation
must pick an order and it affects the transient. **Proposed order per outer
round: refine `t̃_code` (BP inner loop) first, then refine `t̃_src` (one
denoiser projection).** This keeps the existing chunk structure (BP chunk, then
denoiser) and changes only *what* is fed where. Rationale:

1. **Cheap/robust factor first.** BP is inexpensive and well-conditioned; the
   denoiser is an expensive amortized projection. Converging the code
   constraints first gives the source factor a code-consistent cavity to
   project, which is the standard turbo-on-codes and EP-on-codes ordering.
2. **The cavity must be meaningful.** The denoiser turns a cavity *image* into a
   denoised image (§4.1). If the source factor ran first (on a raw
   channel-only image with parity not yet enforced), it would project from a
   much noisier, code-inconsistent field. BP-first yields `cavity_source =
   payload0 + code_contribution`, the best available "everything-but-source"
   belief.
3. **Minimal migration surface.** The current loop is already BP→denoiser
   (`decoder.py:240` then `:261`); keeping the order means Tasks 3–4 touch only
   the cavity input (`:258/:261`), the state update (`:284`), and add
   `src_site` — no reordering, easy `legacy_turbo` vs `pure_ep` A/B.
4. **Channel factor: refined never.** `t̃_ch` is exact (§4.2), computed once
   (`payload0`), never re-projected — consistent with "channel frozen".
5. **Final chunk: code only.** The last schedule entry runs BP with no source
   update (`decoder.py:246-247`), producing the decode. Pure EP keeps this: the
   last action is a code-factor refinement using the final `src_site`.

Per outer round, in one line:
```
round:  BP(payload0 + src_site) → cavity = BP_post − src_site → src_site ← project(cavity)
final:  BP(payload0 + src_site) → decode
```

---

## 6. Migration checklist (what Tasks 3–4 will implement; no code here)

* **Task 3** — add `src_site` state (init 0, `[B, k_payload]`); feed the
  denoiser `cavity_source = BP_post − src_site` instead of `BP_post`; store the
  returned value as `src_site_new`. Annotate Approximations C (mean-only) and D
  (σ not from cavity variance). Keep `legacy_turbo` path selectable.
* **Task 4** — feed BP the prior `payload0 + src_site` (drop the
  `+ β·bp_ext + α·src_ext` reconstruction); apply site-space damping `ρ`; gate
  legacy vs pure-EP behind a mode flag so both run for comparison.
* **Validation anchor** — with `β=0, α=1, ρ=1` and any schedule, the **first**
  denoiser call and its effect must match `legacy_turbo` bit-for-bit (§4.1);
  divergence must appear only from chunk `idx ≥ 1` and be attributable to the
  removed `S_{idx-1}` double-count (§4.2).

---

## Completion conditions (Task 2)

- ✅ `docs/EP_MIGRATION_PLAN.md` created.
- ✅ Double-counting path pinned to specific lines
  (`decoder.py:219, 237-240, 258-269, 272, 284` and
  `source_prior.py:118-130`), with the `idx≥1`-only feedback route spelled out
  (§1).
- ✅ Explicit `src_site` state introduced (shape `[B,k_payload]`, bit-LLR,
  init 0), alongside the existing `payload0`/`msg_v2c` sites, with the
  `posterior = channel + code + src_site` LLR identity (§2).
- ✅ Corrected source-factor EP pseudocode with `cavity = posterior −
  src_site_prev`, `src_site_new = src_post − cavity`, reconstruction `posterior
  = cavity + src_site_new`; code-factor cycle stated too (§3).
- ✅ Numerical difference vs the turbo equations analyzed: exact
  coincidence conditions (`idx=0, β=0, α=1, ρ=1`), divergence modes, and why
  small `α` masks the bug (§4).
- ✅ Factor refinement order (BP → source) proposed with rationale (§5).
