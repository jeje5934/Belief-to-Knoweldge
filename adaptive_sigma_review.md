# Adaptive Sigma — Review Notes

> **CRC correction:** sigma_actual/RMSE observations remain usable, but all
> CRC-conditioned groups and CRC-BLER claims below are historical soft-CRC
> results. The hard-CRC sigma canonical replacement is documented in
> `HARD_CRC_REMEASUREMENT_REPORT_KO.md`.

This document explains the adaptive denoiser-σ scheduling scheme used in
this branch (`onlyextrinsic_ada_sigma`), how it relates to DDECC, the
exact syndrome computation, sign-convention conventions, and the
status of the current evaluation.

The core turbo update equation is **unchanged**:

```
bp_ext    = BP_post  − payload_intr
src_ext   = src_post − BP_post
new_input = channel + β · bp_ext + α · src_ext
```

Only how σ is chosen for the EDM denoiser between BP chunks changes.

---

## 1. Why adaptive σ?

The fixed denoiser σ = 0.3 baseline already gives a strong waterfall
shift on Fashion-MNIST + 5G LDPC.  The motivation for adaptive σ is to
match DDECC's intuition:

* When BP has nearly converged (few unsatisfied checks → low syndrome
  ratio), the residual error is small.  A small σ asks the denoiser to
  do a *gentle* correction.
* When BP is far from a codeword (many unsatisfied checks → high
  syndrome ratio), trust BP's posterior less and run the denoiser at a
  larger σ to make a more aggressive correction.

This is the same idea behind DDECC: use the syndrome as a per-iteration
signal of how unreliable the current decision is.

---

## 2. DDECC analogy

| DDECC                                 | This branch                          |
|---------------------------------------|--------------------------------------|
| Per-iteration syndrome → reliability  | Per-chunk syndrome ratio → σ        |
| Learned reliability mask              | Static or calibrated piecewise σ    |
| Joint training                        | Pretrained EDM denoiser, no joint   |

We deliberately do **not** retrain the denoiser, and the source-extrinsic
formula is unchanged.

---

## 3. Exact syndrome computation

Given the BP marginal logits `x_hat` (full LDPC graph domain, length
`num_vns`), the hard decision is `c_hat = (x_hat > 0)` (default sign
convention; see §4).  The syndrome is

```
s = H · c_hat   mod 2          ∈ {0, 1}^{num_cns}
syndrome_weight = sum(s)
syndrome_ratio  = syndrome_weight / num_cns
```

`H` is the parity-check matrix of the *full* LDPC code, so the syndrome
is taken on the graph-domain bits — not on the payload, systematic, or
rate-matched slice.  The runtime assertion

```
H.shape[1] == hard_bits.shape[-1]
```

inside `compute_syndrome` enforces this and raises a clear error if a
narrower bit slice is ever passed in by mistake.

---

## 4. Why `xhat_gt0` is the default

The formal Sionna LLR convention is `log P(b=0)/P(b=1)`, which would
suggest `x_hat < 0 → bit=1`.  In practice, end-to-end syndrome checks at
high Eb/N0 (1.2 dB) on this LDPC graph configuration consistently show:

| Convention   | Mean syndrome ratio at 1.2 dB | Interpretation                  |
|--------------|------------------------------:|---------------------------------|
| `xhat_gt0`   | ≈ 0.018                       | Near-zero: codeword consistent. |
| `xhat_lt0`   | ≈ 0.573                       | Large: inverted bit mapping.    |

Therefore `xhat_gt0` is the **default**.  `xhat_lt0` remains available
only as `--syndrome-hard-decision xhat_lt0` for debugging.  When
`--syndrome-sign-debug` is set, the runner reports both conventions'
mean syndrome ratios and which one is better, **without** storing
heavy per-sample arrays unless explicitly requested.

If the underlying Sionna version or LDPC config ever changes, re-run the
check with `--syndrome-sign-debug` before flipping the default.

---

## 5. Schedulers

All schedulers live in `syndrome_sigma_schedule.py` and expose a single
method `select_sigma(chunk_idx, ratios) -> tf.Tensor`.

### 5.1 Fixed σ (baseline)

`FixedSigmaScheduler(σ)` — broadcasts a scalar σ across the batch.  The
canonical fixed baseline is **σ = 0.3** with α = 0.1, β = 0.1.  This
baseline must remain available; it is the strongest reference point.

### 5.2 Hand-crafted lookup

`HandcraftedLookupScheduler(thresholds, sigma_levels)` with default
thresholds `(0.03, 0.10, 0.20)` and σ levels `(0.15, 0.25, 0.35, 0.45)`.
This gives a monotonic, intuition-driven mapping from syndrome ratio to
σ.

### 5.3 Calibrated lookup (JSON)

`CalibratedLookupScheduler` is loaded from JSON via
`load_sigma_lookup_json` / `build_calibrated_scheduler_from_json`.  It
supports two modes:

* **Global lookup**  — single `(thresholds, sigma_levels)` for all chunks.
* **Per-chunk lookup** — separate `(thresholds, sigma_levels)` per chunk
  index.  Recommended by default because the syndrome-ratio
  distribution differs across BP chunks (early chunks see large
  ratios, later chunks see small ratios).

If a chunk index is missing from a per-chunk lookup, the scheduler falls
back to the largest available chunk index ≤ idx (and ultimately the
smallest available chunk).

---

## 6. Monotonic σ option

`--monotonic-sigma` post-processes the active lookup so that

```
sigma_levels[i] = max(sigma_levels[i], sigma_levels[i-1])
```

i.e. larger syndrome-ratio bins cannot map to a *smaller* σ.  This
matches the physical intuition: more violations ⇒ at least as much
denoiser smoothing.

The flag is plumbed through `cli_common.add_adaptive_sigma_args`, so
every script that consumes adaptive σ (`plot_comparison.py`,
`experiment.py`, `syndrome_diagnostics.py`, `visualize_progression.py`,
`calibrate_sigma_lookup.py`) accepts it.  Unconstrained lookups remain
available for ablation by simply omitting the flag.

For per-chunk calibrated lookups, the repair is applied independently
per chunk.

---

## 7. Calibration objectives

`calibrate_sigma_lookup.py` supports three objectives:

| Objective              | Meaning                                                                                                   | Status         |
|------------------------|-----------------------------------------------------------------------------------------------------------|----------------|
| `source_posterior_ber` | Cheap proxy: BER on payload bits after one denoiser step on `BP_post`, before the remaining BP chunks.    | **Proxy.**     |
| `tail_final_ber`       | Apply candidate σ at the calibration chunk, run remaining BP with `--trace-tail-sigma`, score final BER.   | Surrogate.     |
| `tail_final_nack`      | Same tail trace but score CRC NACK (block error rate) on the final decoder output.                        | **Recommended.** |

Use `tail_final_nack` for any calibration meant to influence the final
decoder.  `source_posterior_ber` is kept for cheap, illustrative sweeps
only and **must not** be used as the main calibrated lookup unless
explicitly requested.

---

## 8. Sigma range

For the main experiments, σ is kept near the strong fixed baseline:

```
candidate_sigmas = 0.20  0.25  0.30  0.35  0.40
sigma_min        = 0.20
sigma_max        = 0.40
```

Wider ranges (e.g. 0.05–0.50) are allowed for diagnostics only — they
inflate the search space without buying anything in the main regime.
The defaults in `calibrate_sigma_lookup.py` and `safe_sweep.sh` already
match this range.

---

## 9. Calibrated-lookup JSON schema (v2)

`calibrate_sigma_lookup.py` writes a JSON file with the following keys:

```
schema_version          (= 2)
method                  ("offline_sigma_lookup_calibration")
objective               ("source_posterior_ber" | "tail_final_ber" | "tail_final_nack")
alpha, beta             decoder weights used during calibration
schedule                BP chunk lengths
ebno_list               Eb/N0 values used
batch, rounds           sample counts
syndrome_hard_decision  ("xhat_gt0" | "xhat_lt0")
collection_sigma        σ used to drive the trajectory while calibrating
trace_tail_sigma        σ used on later chunks during tail-objective scoring
candidate_sigmas        list of evaluated σ values
sigma_min, sigma_max    optional clamps applied at inference time
binning_type            ("quantile" | "manual")
monotonic_sigma         bool — whether monotonic repair was applied
per_chunk               bool — per-chunk lookup vs single global lookup
samples_per_bin         (global lookup only)
thresholds, sigma_levels (global lookup only)
per_chunk_lookup        (per-chunk lookup only)
                            { "<chunk_idx>": {
                                "thresholds": [...],
                                "sigma_levels": [...],
                                "num_bins": int,
                                "per_bin": [
                                    { "bin_idx", "bin_edges_open_closed",
                                      "num_samples",
                                      "candidate_objective_values",
                                      "selected_sigma",
                                      "selected_objective_value" }, ... ]
                            }}
objective_notes         documentation strings for each objective
num_samples_total       int
```

The loader (`load_sigma_lookup_json`) is backward compatible with v1
files (no `schema_version` key, top-level `thresholds`/`sigma_levels`
only).

---

## 10. Current empirical status

* The **fixed σ = 0.3, α = β = 0.1** setting remains the strongest
  reference (best known BLER ≈ 0.030 at Eb/N0 = 0.8 dB).
* The **hand-crafted adaptive σ** lookup is functional and visibly
  changes σ with chunk index, but is **not yet proven to beat fixed σ
  = 0.3** on the test bitbank.
* The **calibrated adaptive σ** lookups (both `source_posterior_ber`
  proxy and `tail_final_nack` surrogate) are functional and produce
  per-chunk schedules; preliminary smoke runs do not show a
  statistically clear win over fixed σ = 0.3 yet.
* `--monotonic-sigma` is a valid robustness option that prevents
  pathological calibration outputs but does not by itself improve BLER.

In short: adaptive σ is **architecturally clean** (per the refactor in
this commit), but its *practical advantage over fixed σ = 0.3 is not
yet established* on this benchmark.  Larger sweeps are deliberately
not run here — the user's working principle is host-stability-first.

---

## 11. GPU stability

For routine work, this branch only runs **small smoke tests**:

* sign-convention sanity (`visualize_progression.py
  --syndrome-sign-debug`)
* short syndrome diagnostics
  (`syndrome_diagnostics.py --batch 64 --rounds 2`)
* short fixed vs adaptive comparison
  (`plot_comparison.py --compare-adaptive --batch 64 --rounds 2 \
   --no-save-plot --append-csv …`)

`safe_sweep.sh` enforces:
* sequential execution only,
* CPU thread caps,
* GPU memory cap (`--gpu-memory-mb`),
* `RUN_TIMEOUT` per step,
* GPU temperature cooldown,
* CUDA cleanup between steps via `cuda_cleanup.py`,
* explicit `ALPHA`, `BETA`, `SIGMA`, `EBNO_LIST` env vars passed as
  CLI args to every command.

Do **not** run large sweeps or launch concurrent GPU jobs here.

---

## 12. Things explicitly preserved

* The pretrained EDM denoiser is **not** retrained.  Its checkpoint
  (`checkpoints/denoiser.pt`) is unchanged.
* The turbo extrinsic equations are unchanged.
* The `xhat_gt0` default is preserved.
* The fixed σ = 0.3 baseline is preserved.
* The hand-crafted adaptive σ baseline is preserved.
* All previous command-line invocations of `plot_comparison.py`,
  `experiment.py`, `syndrome_diagnostics.py`, `visualize_progression.py`
  continue to work.
