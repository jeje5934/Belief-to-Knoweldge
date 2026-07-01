# EP_SCHEDULING_EXPERIMENT — factor refinement scheduling & SNR sweep

> **Status**: experiment log + confirmed configuration. Supports
> [`EP_THEORY.md`](EP_THEORY.md) §4.5–4.6 (BP iterations are a *schedule*, not an
> approximation) and records the two experiments that fixed the decoder's default
> configuration.
>
> **Confirmed decision (see §3):** the aligned decoder uses **`bp_schedule=[2]*15`,
> `ep_mode=True`, `ep_update="full_ep"` (α_ep=β_ep=1), adaptive syndrome-ratio σ**.
> Legacy turbo α/β are removed from the aligned path (kept only as the
> `ep_mode=False` comparison baseline).

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

## 3. Confirmed configuration

* **Schedule `[2]×15`** — the sweet spot of §2.4.
* **`full_ep`, α_ep = β_ep = 1** — pure EP site replacement (`EP_THEORY.md` §4.4);
  **legacy turbo α/β are removed** from the aligned decoder (the `ep_mode=False`
  turbo path with α/β is retained *only* as a comparison baseline).
* **Adaptive syndrome-ratio σ** — retained; framed honestly as a cheap proxy for
  the cavity's global uncertainty (`EP_APPENDIX.md` §A.6, Approx D).
* **No damping.** The `[2]×15` terminus is **not** a true fixed point
  (`Δsite → 0`) but a **small-amplitude stable orbit** (~250 band). This is the
  natural terminus of *pure* EP with a non-linear factor: the denoiser is not in
  the Bernoulli/Gaussian approximating family, so the moment-matching fixed-point
  conditions (Minka 2001 §3.3) cannot be met exactly and the site settles into a
  bounded orbit rather than a point. We report this honestly rather than forcing
  `Δsite → 0` with damping — damping would trade EP fidelity for a cosmetically
  stationary site.

---

## 4. SNR sweep (Pure EP vs Baseline BP)

*(Populated by `ep_snr_sweep.py`; outputs in `results/ep_snr_sweep.{csv,png,json}`.)*

Curves: **bp30** (baseline BP, 30 iters, no prior — same BP budget as `[2]×15`),
**bp100** (baseline BP, 100 iters — BP-limit reference), **ep** (`[2]×15`,
`full_ep`, adaptive σ). Two subplots (BER, BLER; log-y) vs Eb/N0 over the
waterfall 0.4–1.2 dB.

<!-- SNR_SWEEP_RESULTS -->
_Results table, waterfall-shift (dB), and the low-SNR schedule-adequacy diagnostic
are inserted here once the sweep completes._
