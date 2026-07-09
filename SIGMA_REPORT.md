# practical_sigma — denoiser-σ strategies: annealing vs syndrome-adaptive

Branch `practical_sigma` (base `089fd8a`). EP `[5]×20`, α_ep=0.01, damped_ep fixed;
**only the denoiser σ strategy varies**.  Target = the denoiser noise argument σ
(`SourcePriorDenoiser.forward(llr, sigma)` → EDM call); the pixel→bit readout
`sigma_post` stays **3.0** (separate variable, no entanglement).  Existing decoder
code untouched; the only additive change is `AnnealingSigmaScheduler` in
`syndrome_sigma_schedule.py`.

## Strategies
- **fixed σ=0.3** — current baseline anchor.
- **handcrafted adaptive** — the final-table "EP best" config (DEFAULT lookup:
  ratio (0.03,0.10,0.20) → σ (0.15,0.25,0.35,0.45)).
- **2-stage LUT adaptive** (`[B]`, new) — LUT1 (pure-BP steady-state syndrome
  w̄(SNR)) ∘ LUT2 (σ*(SNR) by end-to-end NACK); composed σ(w) ∈ [0.15,0.35].
- **annealing** (open-loop, chunk-index) — σ_s→σ_e, linear/geom.
- **combo** — min(LUT-adaptive, annealing).

## Results @ 0.5 dB, 3200 cw, Wilson 95% CI (0.6 dB EP floors ~6e-4, not resolvable)

| strategy | BLER | 95% CI | nack/3200 |
|---|---|---|---|
| anneal geom 0.35→0.15 | 0.0066 | [.0043,.0100] | 21 |
| **fixed σ=0.30 (anchor)** | 0.0066 | [.0043,.0100] | 21 |
| 2-stage LUT adaptive | 0.0072 | [.0048,.0108] | 23 |
| combo (LUT ∧ anneal1.2) | 0.0078 | [.0053,.0115] | 25 |
| handcrafted adaptive | 0.0112 | [.0081,.0155] | 36 |
| *(screen 640cw)* anneal geom 0.6 | 0.0219 | [.0131,.0364] | 14/640 |
| *(screen)* anneal geom 1.2 | 0.1047 | [.083,.131] | 67/640 |
| *(screen)* anneal linear 1.2 | 0.2250 | [.194,.259] | 144/640 |
| *(screen)* anneal geom 2.4 | 0.3203 | [.285,.357] | 205/640 |

### Extension to 0.6 / 0.7 dB (3200 cw) — the tie holds

| strategy | 0.5 dB | 0.6 dB | 0.7 dB |
|---|---|---|---|
| fixed σ=0.30 | 0.0066 [.0043,.0100] | 0.0019 [.0009,.0041] (6) | 0.0003 [.0001,.0018] (1) |
| 2-stage LUT adaptive | 0.0072 [.0048,.0108] | 0.0003 [.0001,.0018] (1) | 0.0003 [.0001,.0018] (1) |

At 0.6 dB the LUT adaptive is point-lower (1 vs 6 nack/3200) but the CIs overlap
([.0001,.0018] vs [.0009,.0041]) — not a separation; at 0.7 dB both floor at 1/3200.
So fixed and the LUT adaptive are statistically **tied across 0.5–0.7 dB**, with at
most a non-significant hint that the LUT adaptive is not worse.

## Verdict

**(a) No σ strategy beats fixed σ=0.3 with CI separation.** fixed_0.30, the new
2-stage LUT adaptive, the small-σ anneal, and combo are all ~0.007 with fully
overlapping CIs.  The *handcrafted* adaptive (the final-table "EP best") is
point-**worse** (0.0112) — so the adaptive used in the final table was **not** an
improvement over fixed σ at the resolvable SNR; the new LUT adaptive (0.0072)
repairs that regression but still only **ties** fixed.  σ is not a BLER lever
here (echoing the earlier "schedule/α is not a BLER lever" §F finding).

**(b) A large early σ (the ICDM-style "mode-smearing" hypothesis) is decisively
HARMFUL, not helpful.** Annealing degrades monotonically with σ_s: 0.35→0.0066,
0.6→0.022, 1.2→0.105, 2.4→0.320 (geom; linear is worse still — it dwells longer
at high σ).  The mode-smearing diagnostic (`results/sigma_denoise_viz.png`) shows
why: a single Tweedie call at large σ does **not** yield a broad-but-coherent
garment — it collapses to a **gray, over-smoothed mean** (denoised-vs-true PSNR
15.9→6.2 dB as σ 0.15→2.4).  The per-round BER trajectory
(`results/sigma_compare.png`) confirms the damage is **irreversible on the chunk
axis**: large-σ_s runs plateau at BER 0.01–0.04 and do **not** recover when σ
falls in later chunks.  → *Single-shot Tweedie cannot split modes by raising σ;
it just smears to the mean.  Mode exploration would need **sampler-level
(iterative) intervention**, not a one-shot σ schedule.*  This extends the
exclusion chain from the earlier increasing-α failure.

**(c) Closed-loop (syndrome-adaptive) does NOT beat open-loop (annealing) at the
same σ range.** LUT adaptive (0.0072) ≈ anneal geom 0.35→0.15 (0.0066).  Their σ
trajectories are nearly identical (corr 0.86, RMSΔσ 0.055; combo/handcrafted corr
≥0.99).  Reason: the syndrome decays predictably across chunks, so the
state-driven σ path ≈ a deterministic geometric schedule — the syndrome carries
**no actionable information beyond "how many chunks in."**  Closed-loop control is
redundant here.

**(d) Optimal σ trajectory: flat and small.** σ≈0.3 constant is as good as any
schedule; a gentle 0.35→0.15 descent ties it; fixed σ=0.15 is slightly worse
(0.0141); anything venturing above ~0.45 hurts, catastrophically past ~1.0.  The
useful σ band is the narrow [0.15,0.35] around the trained median 0.30.

## Files
`sigma_experiment.py` (runner), `calibrate_sigma_2stage.py` (LUT1∘LUT2→JSON),
`sigma_compare.py` (+`_plot.py`), `sigma_denoise_viz.py`;
`AnnealingSigmaScheduler` added to `syndrome_sigma_schedule.py`.  Results:
`sigma_2stage_lookup.json`, `sigma_compare.json`, `sigma_compare.png`,
`sigma_2stage.png`, `sigma_denoise_viz.png`.
