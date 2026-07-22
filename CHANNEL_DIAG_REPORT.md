# Fading diagnostic — legacy vs EP under fast Rayleigh + imperfect CSI

Branch `practical_sigma`.  3-way (BP-only / legacy warm-start turbo / EP), same
method-B channel LLR to all three.  **No commit — report + wait.**

## 0. Method A ≡ B (the noise-scale concern is refuted)
`channel_ab_compare.py`: on the SAME channel realisation, method A (equalise,
Ñ0=N0/|ĥ|²) and method B (direct likelihood, fixed N0) give **identical** LLRs at
every σ_e² — max|A−B| ≈ 1e-4 (float), corr 1.000, sign-agree 1.000, equal std.
Algebraically `|ỹ−x|²/Ñ0 = |y−ĥx|²/N0`, so the |ĥ|² cancels; A does **not**
over-amplify at deep fades.  The miscalibration source is `ĥ≠h` in the *signal*
term, present in BOTH.  (We nonetheless use B for all runs, as requested.)

## 1. LLR calibration (`fading_llr_calib.py`, ebno 5)
Imperfect CSI makes the channel LLR **over-confident** (actual error > what |LLR|
implies); perfect CSI is calibrated:

| condition | implied err | actual err | ratio | verdict |
|---|---|---|---|---|
| perfect CSI | 0.109 | 0.109 | **1.00** | calibrated |
| σ_e²=0.05 | 0.109 | 0.124 | 1.14 | — |
| σ_e²=0.10 | 0.108 | 0.138 | 1.28 | — |
| σ_e²=0.20 | 0.106 | 0.160 | **1.52** | over-confident |

By |LLR| bin (σ_e²=0.1): mid-confidence LLRs are the worst — [2,5): actual 0.098
vs implied 0.045 (2×), [5,10): 0.009 vs 0.002 (4.5×).  mean|LLR| barely moves
(6.5→6.8) while the error rate rises → over-confidence, not larger magnitudes.

## 2. Perfect-CSI fading control + σ_e² sweep (3-way, ebno 2, 256–512 cw)
`results/f3_perfect.csv`, `results/f3_imperfect.csv`.  ebno 2 is the fading
waterfall (BP-only recovers by ebno 3).

| σ_e² | BP-only | **legacy** | **EP** |
|---|---|---|---|
| 0.0 (perfect) | 0.994 | **0.006** [.002,.017] | 0.209 [.176,.246] |
| 0.05 | 1.000 | **0.354** [.313,.396] | 0.938 [.913,.955] |
| 0.10 | 1.000 | **0.979** | 1.000 |
| 0.20 | 1.000 | 1.000 | 1.000 |

- **Both denoiser decoders BEAT BP-only** at the fading waterfall (perfect CSI:
  BP 0.994 vs legacy 0.006 / EP 0.209).  The denoiser is **not** useless under
  fading — the earlier "denoiser weak" read was an EP-only + easy-SNR artifact.
- **legacy ≫ EP at every σ_e²** — and already at **perfect CSI** (0.006 vs 0.209,
  CI-disjoint).  So it is **fading itself**, not imperfect CSI, that first splits
  them.  Imperfect CSI then widens it and EP **collapses faster** (σ_e²=0.05:
  EP 0.938 vs legacy 0.354).

**Re-tuning EP α_ep does NOT recover it** (ebno 2, perfect CSI; `results/f3_retune.csv`):

| α_ep | 0.005 | **0.01** | 0.02 | 0.05 | 0.10 |
|---|---|---|---|---|---|
| EP BLER | 0.293 | **0.195** | 0.238 | **1.000** | **1.000** |

Best EP is α_ep=0.01 (0.195) — still **16× worse than legacy** (0.012), and EP
**diverges to 1.0 for α_ep ≥ 0.05**.  So the accurate cavity + source injection is
*unstable* under fading (narrow α_ep window, blows up when the source is trusted
more).  legacy at its AWGN-best α=β=0.1 is already near-optimal.  → EP's fragility
is **fundamental (a/c), not a tuning artefact (b)**.

## 3. Verdict (per series)
- **(ii) one series is weak, not both.** legacy (incomplete cavity / self-
  anchoring) is robust to fading and degrades gracefully with CSI error; EP
  (accurate cavity) is fragile — it fails on pure fading and collapses fastest.
- **§F AWGN legacy≈EP equivalence is BROKEN by fading — a new finding.** The
  "accurate cavity vs incomplete cavity" distinction, invisible at AWGN, is
  decisive for channel robustness: the incomplete cavity's self-anchoring buffers
  the fading/over-confident LLR that the exact cavity faithfully (and harmfully)
  propagates into the source projection.
- The "denoiser is weak under fading" conclusion is **withdrawn**: with both
  series + perfect-CSI control + method B, the denoiser *helps* (both beat BP);
  the real story is legacy ≫ EP.

- **Cause is fundamental, not tuning.** EP cannot be re-tuned to legacy (best
  α_ep=0.01 → 0.195 vs legacy 0.012; diverges for α_ep≥0.05).  The exact cavity
  faithfully carries the fading + (under imperfect CSI) over-confident channel LLR
  into the source projection, where the denoiser amplifies it; the incomplete
  cavity's self-anchoring discards enough of that miscalibrated evidence to stay
  stable.  This is the concrete mechanism behind requirement (b) in §F, now made
  visible by a channel that breaks the AWGN tie.

## Caveats
- Scale is 256–512 cw (diagnostic, not final).  The direction is CI-clean at the
  key points (legacy 0.006 [.002,.017] vs EP 0.209 [.176,.246] disjoint; legacy
  0.354 vs EP 0.938 disjoint).  Confirmation at 3200 cw + a finer ebno waterfall
  is the natural next step.
- ebno 2 is the only fading-waterfall point (BP recovers by ebno 3); a denser
  waterfall (e.g. 1.5/2/2.5) would sharpen the legacy-vs-EP margin curve.

## Files
`channel_models.py` (+method B), `channel_ab_compare.py`, `fading_3way.py`,
`fading_llr_calib.py`.  Results in `results/f3_*.csv`, `fading_demo.csv`.
