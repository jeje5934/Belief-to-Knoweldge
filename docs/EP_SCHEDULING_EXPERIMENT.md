# EP_SCHEDULING_EXPERIMENT — factor refinement scheduling & SNR sweep

> **Status**: experiment log. Supports [`EP_THEORY.md`](EP_THEORY.md) §4.5–4.6
> (BP iterations are a *schedule*, not an approximation) and records the
> scheduling + **decode-quality** experiments.
>
> **⚠ Corrected conclusion (see §3, §5).** An earlier draft "confirmed"
> `bp_schedule=[2]*15` with `ep_update="full_ep"` (α_ep=β_ep=1). That was based on
> convergence *dynamics* (Δsite/logZ) only. A later decode-quality test
> (§5) shows **`full_ep` decodes at BLER 1.0** — its "stable orbit" is a stably
> *wrong* image. The working configuration is **`[2]*15` with damped source
> sites** (`damped_ep`, small `ep_source_power`, or the legacy turbo path),
> which reaches BLER ≈ 0.004 at 0.8 dB. `[2]*15` remains the right *schedule*;
> only the *update weight* had to change.

---

## 1. Why schedule at all (framing)

Each parity check is its own factor and each BP iteration *is* an EP refinement
step of those check factors (Minka 2001 §4; `EP_THEORY.md` §4.5). So a decode is
a **schedule** over which factors to refine and how much before re-reading the
others — there is no "code projection to run to convergence", and the number of
BP iterations per chunk is not an approximation. Different schedules induce
different **convergence dynamics** for the outer EP loop; this experiment finds
which one behaves best.

Setup (unless noted): `full_ep`, fixed BP, Eb/N0 = 0.8 dB, batch 8,
Fashion-MNIST test images, denoiser σ = 0.3, checkpoint loaded, identical channel
realization and denoiser weights across schedules.

---

## 2. Scheduling experiment

Four schedules spanning "many BP, source rarely" → "few BP, source often". Per
source update we log `src_site_delta_l2` (outer stationarity; → 0 at a fixed
point) and `source_logZ` (source evidence, `EP_DIAGNOSTICS.md`). We also run an
**endpoint BP-oscillation test**: freeze each schedule's final `src_site` and run
the check-factor BP to settle (cap 250, tol 1e-3) — does it reach a stable point
or a limit cycle?

### 2.1 Summary

| schedule | #source updates | Δsite: first→last (min) | logZ: first→last | mean ‖site‖ | endpoint BP |
|---|---|---|---|---|---|
| `[10]×5` | 4 | 619 → 405 (405) | −5546 → 1663 | 507.8 | settles, 47 it |
| `[50]×3` | 2 | **2403 → 2160** | −4707 → −4433 | 786.7 | **limit cycle** (cap 250, Δmsg 13.1) |
| `[2]×15` | 14 | 498 → **242** (238.7) | 1789 → **~2100** | 539.6 | settles, **25 it** |
| `[1]×20` | 19 | 499 → **386** (drift↑) | 2718 → **1518** (drift↓) | 537.7 | settles, 36 it |

### 2.2 Trajectories

```
Δsite_l2 per source update:
  [10]×5 : 619 612 484 405
  [50]×3 : 2403 2160
  [2]×15 : 498 291 282 273 273 273 269 267 258 242 239 254 252 242      ← plateaus ~250
  [1]×20 : 499 161 262 258 308 301 324 326 332 339 342 350 355 365 …386  ← drifts UP

source_logZ per source update:
  [10]×5 : -5546 540 1037 1663                                          ← still rising
  [50]×3 : -4707 -4433                                                  ← stuck negative
  [2]×15 : 1789 2152 2118 2127 2142 2307 2011 2289 … 2063               ← plateaus ~2100
  [1]×20 : 2718 1789 2241 … 1747 1718 1528 1558 1672 1518               ← drifts DOWN
```

Final `src_site` differs across schedules (pairwise `max|Δ|`: `[10]×5`vs`[50]×3`
= 64.7, `[2]×15`vs`[1]×20` = 46.7, `[10]×5`vs`[2]×15` = 46.1) — they do **not**
reach a common state.

### 2.3 The oscillation, reframed (not "code non-convergence")

A companion run with `bp_convergence=True` on `[10]×5` shows the check-factor BP
**settles when the source site is source-free** (chunk 0: 52 it; chunk 4: 62 it)
but enters a **limit cycle** on the middle chunks where a strong source site is
held fixed (chunks 1–3: capped at 200 it, `max|Δ msg_v2c|` up to 44). This is a
property of the **joint schedule** — over-refining the check subgraph while the
source is frozen — not a defect of "the code factor" and not a non-existent
"projection failing to converge" (`EP_THEORY.md` §4.6). The endpoint test above
confirms it at the schedule level: only the extreme `[50]×3` lands on a `src_site`
whose check subgraph oscillates.

(For reference: at an identical chunk-0 input, a fixed 10-iter code state differs
from a 52-iter settled one by mean 24.8 / max 38.7 LLR — a large *refinement*
difference, but **not** an approximation error: both are legitimate intermediate
EP states, §4.5.)

### 2.4 Judgment — hypothesis partially supported, with a sweet spot

Hypothesis: "few BP + source often" avoids oscillation and reaches the outer
fixed point better. Verdict: **the core is supported, both extremes fail:**

* **Extreme many-BP (`[50]×3`)** — worst: huge site swings, endpoint BP limit
  cycle. Over-refining the checks while the source is frozen drives the messages
  into oscillation.
* **Moderate alternation (`[2]×15`)** — **best**: Δsite falls and **plateaus at a
  ~250 band**, logZ **plateaus at ~2100**, endpoint settles fastest (25 it).
* **Extreme alternation (`[1]×20`)** — also bad: with 1 BP iter the cavity is
  under-refined, so the source chases a noisy cavity and `src_site` **drifts up**
  (no plateau).

So the good regime is a **sweet spot** — enough BP to give a meaningful cavity,
frequent enough source updates to let the two factors co-adapt. Here that is
**`[2]×15`**, adopted as the default (§3).

---

## 3. Confirmed configuration (decode-quality corrected)

The §2 experiment fixed the **schedule** (`[2]×15`) by convergence dynamics. A
separate **decode-quality** test (§5) then fixed the **update weight**:

* **Schedule `[2]×15`** — the sweet spot of §2.4. Unchanged.
* **Source weight — NOT `full_ep`.** `full_ep` (α_ep=1) decodes at **BLER 1.0**
  (§5): the denoiser is an over-confident, un-calibrated factor and full site
  trust corrupts the belief from the first round. The working weight is a
  **damped source site**: `damped_ep` with a small `ep_source_power` chosen
  so the *accumulated* site stays ≈ 20–30 % of the denoiser's full belief
  (e.g. `[2]×15` → `ep_source_power ≈ 0.02` → BLER ≈ 0.004 at 0.8 dB), or
  equivalently the legacy turbo path (`ep_mode=False`, α≈0.1).
* **Adaptive syndrome-ratio σ** — retained; cheap proxy for the cavity's global
  uncertainty (`EP_APPENDIX.md` §A.6, Approx D).
* **On the "stable orbit".** `full_ep`'s `Δsite`/`logZ` plateau (§2.2) is a
  *stably wrong* image, not a good decode — a caution that convergence-dynamics
  metrics do **not** imply decode quality. Damping is not a cosmetic choice here;
  it is required for the decoder to work, and is the honest statement of the
  EP-fidelity vs performance tension (§5, `EP_APPENDIX.md` §A.6).

---

## 4. SNR sweep (Pure EP vs Baseline BP)

*(Harness: `ep_snr_sweep.py`; outputs in `results/ep_snr_sweep.{csv,png,json}`.
The EP curve must use a **working** config — `damped_ep`, small
`ep_source_power` — not `full_ep`, which is BLER 1.0 (§5). Re-run pending.)*

Curves: **bp30** (baseline BP, 30 iters, no prior — same BP budget as `[2]×15`),
**bp100** (baseline BP, 100 iters — BP-limit reference), **ep** (`[2]×15`,
damped source, adaptive σ). Two subplots (BER, BLER; log-y) vs Eb/N0 over the
waterfall 0.4–1.2 dB.

<!-- SNR_SWEEP_RESULTS -->

---

## 5. Decode quality: full EP fails; damping is required

Convergence dynamics (§2) do not measure decoding. Measuring BLER/BER at 0.8 dB
(CRC-checked; harness validated by turbo α=0.1 reproducing the README) exposes a
sharp result.

### 5.1 `full_ep` decodes catastrophically — diagnosis (contamination, not landing)

| config | BLER | BER |
|---|---|---|
| baseline BP 30 it (no prior) | 0.50 | 3.5e-3 |
| baseline BP 100 it | 0.008 | 6.6e-4 |
| turbo α=.1 β=.1 `[10]×3` (README repro) | 0.016 | 3.7e-6 |
| turbo α=.1 β=.1 `[2]×15` | **0.004** | 0.0 |
| **`full_ep` `[2]×15`** | **1.000** | **0.45** |
| **`full_ep` `[10]×3`** | 1.000 | 0.45 |

Two decisive follow-ups pinned the mechanism to **per-round contamination**
(not "insufficient final BP"):

* **Per-round BER** (denoiser ON vs OFF, one batch): ON jumps `0.13 → 0.46` at
  the *first* source update and stays there; OFF (pure BP) decays `0.13 → 0.006`.
  The denoiser corrupts the belief immediately, every round.
* **Final-cleanup BP**: appending BP-100 *with the source kept* recovers nothing
  (BLER 1.0); *dropping the source* + BP-100 recovers full baseline (BLER 0.008).
  So the channel/code state is intact — the **source site is the corruption**.

Mechanism: a 2-iter BP cavity (BER 0.13) fed to the denoiser yields an
over-confident *wrong* image; `full_ep` (α_ep=1) trusts that site fully and the
belief is corrupted. `full_ep` is EP-*aligned* (`full_ep == turbo α=1,β=0`,
verified) but α=1 is empirically catastrophic — turbo α=1 also gives BLER 1.0.

### 5.2 A principled per-pixel precision correction — candidate

The over-confidence is a *2nd-moment* (precision) miscalibration: this branch
uses a **fixed** `sigma_post`, which is uniformly over-confident. The principled
fix is to fill the source site's precision with the real per-pixel projected
variance — the diagonal Tweedie 2nd moment `v_proj = σ²·∂D/∂x̃` — for which the
precision slot already has the right shape (`source_prior.py`:
`projected_pixel_precision`, `forward(return_precision=)`). **This branch does not
implement it** (see the sibling branch `pure-EP_tweedie-2nd-diagonal-precision`,
which does — and finds the *diagonal* insufficient because the denoiser's error
is inter-pixel *correlated*, not diagonal; see that branch's `EP_APPENDIX.md`
§A.6 / §5.2). The trade-off is named in `EP_APPENDIX.md` §A.6.

### 5.3 Fractional EP (damped site) is necessary **and** sufficient

Damping the source site so its *accumulated* magnitude stays small decodes well:

| config | BLER |
|---|---|
| `damped_ep` α=.1 `[10]×3` | 0.027 |
| `damped_ep` α=.05 `[2]×15` | 0.48 (accumulates to ~0.5 of full) |
| **`damped_ep` α=.02 `[2]×15`** | **0.004** |
| `damped_ep` α=.01 `[2]×15` | 0.016 |

`ep_source_power` must scale with the number of source updates so the EMA-
accumulated site stays ≈ 20–30 % of the full denoiser belief (`[2]×15` → ≈0.02;
`[10]×3` → ≈0.1). This down-weighting is an **implicit precision discount** on
the over-confident denoiser factor — exactly what turbo's α≈0.1 does
non-accumulatively (`EP_APPENDIX.md` §A.6).

### 5.4 Verdict

**Honest failure of *pure* EP; damping required.** Full EP integration of this
learned denoiser fails because the denoiser is fundamentally miscalibrated
(globally over-confident); a per-pixel precision correction is the natural
candidate (§5.2, implemented on the sibling branch and found insufficient because
the error is correlated, not diagonal). **Fractional EP — a damped source site —
is practically necessary and, with the accumulated site kept small, matches the
best turbo (BLER ≈ 0.004).** This is the sharpest statement of the project's
EP-fidelity-vs-performance tension.
_Results table, waterfall-shift (dB), and the low-SNR schedule-adequacy diagnostic
are inserted here once the sweep completes._
