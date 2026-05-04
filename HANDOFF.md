# HANDOFF — onlyextrinsic_ada_sigma

Read-this-first context for any new chat / agent that picks up this
codebase.  Pair this file with `README.md` (project overview) and
`adaptive_sigma_review.md` (deep-dive on the adaptive σ design).

> **TL;DR.**  5G LDPC + EDM source-denoiser turbo decoder for
> Fashion-MNIST.  The decoder core is stable; the recent work refactored
> adaptive σ scheduling into a clean module and added a calibration
> pipeline.  **Adaptive σ is *not* yet proven to beat fixed σ = 0.3.**
> Do small smoke runs only — large sweeps risk host instability.

---

## 1. Hard invariants — DO NOT change without explicit user approval

| # | Invariant | Why |
|---|-----------|-----|
| 1 | Turbo extrinsic equations: `bp_ext = BP_post − payload_intr`, `src_ext = src_post − BP_post`, `new_input = channel + β·bp_ext + α·src_ext` | Core algorithm; downstream results depend on this exact form. |
| 2 | EDM denoiser checkpoint `checkpoints/denoiser.pt` (~27 MB) | Pretrained; user does not want it retrained. |
| 3 | Default sign convention: `--syndrome-hard-decision xhat_gt0` | Empirically verified; `xhat_lt0` looks formally correct but produces ~0.57 syndrome ratio at high SNR vs ~0.018 for `xhat_gt0`. |
| 4 | Fixed σ = 0.3 baseline must remain available | Strongest reference point (BLER ≈ 0.030 at 0.8 dB, α=β=0.1). |
| 5 | Hand-crafted adaptive σ baseline must remain available | Used for ablation. |
| 6 | Channel LLR is **frozen** across BP chunks (`payload0` is reused, never modified) | Prevents self-feedback. |
| 7 | `msg_v2c` warm-start across chunks is preserved | BP doesn't fully restart between chunks. |

If a user request appears to violate one of these, ask before editing.

---

## 2. Repo map (focused)

```
onlyextrinsic_ada_sigma/
├── README.md                       project overview + commands
├── HANDOFF.md                      ← this file
├── adaptive_sigma_review.md        adaptive σ design, JSON schema, status
│
├── checkpoints/denoiser.pt         pretrained EDM denoiser (do not retrain)
├── score_denoiser/networks.py      EDM SongUNet (NVlabs/edm)
├── source_prior.py                 PyTorch SourcePriorDenoiser (LLR↔image)
├── denoiser.py                     TF SoftDenoiser wrapper (TF↔PyTorch)
├── decoder.py                      LDPC5GDecoder_soft — uses sigma_scheduler
│
├── syndrome_sigma_schedule.py      ★ canonical adaptive-σ module
│   ├── compute_syndrome / compute_syndrome_weight_and_ratio
│   ├── FixedSigmaScheduler / HandcraftedLookupScheduler
│   ├── CalibratedLookupScheduler   (per-chunk + monotonic)
│   ├── enforce_monotonic_sigma
│   ├── load_sigma_lookup_json      (v1 + v2)
│   └── summarize_chunk_diagnostics
│
├── cli_common.py                   shared adaptive-σ CLI + scheduler builder
├── calibrate_sigma_lookup.py       offline σ-lookup calibration (writes JSON v2)
│
├── plot_comparison.py              Eb/N0 sweep, baseline/fixed/adaptive/calibrated
├── experiment.py                   α sweep at β=0
├── syndrome_diagnostics.py         per-chunk syndrome stats across modes
├── visualize_progression.py        per-image step-by-step plots
├── sigma_sweep.py                  σ sweep at fixed (α, β)         [legacy]
├── train_denoiser.py               denoiser training (do not run)
│
├── gpu_limits.py                   TF GPU memory cap helper
├── cuda_cleanup.py                 between-step CUDA/GC cleanup
├── safe_sweep.sh                   stability-first sequential runner
│
└── results/                        outputs: PNG, JSON, CSV, logs (cursorignored)
```

---

## 3. Adaptive σ — short reference

### 3.1 Sigma schedulers (in `syndrome_sigma_schedule.py`)

All expose `select_sigma(chunk_idx, ratios) -> tf.Tensor` and an
`is_adaptive` boolean.

```python
FixedSigmaScheduler(sigma=0.3)
HandcraftedLookupScheduler(thresholds, sigma_levels,
                           sigma_min=None, sigma_max=None,
                           monotonic=False)
CalibratedLookupScheduler(thresholds=..., sigma_levels=...,    # global
                          per_chunk={ci: {...}},               # per-chunk
                          sigma_min=None, sigma_max=None,
                          monotonic=False)
```

Per-chunk lookup is the **default** in calibration, because the syndrome
ratio distribution differs by BP chunk (early chunks see ~0.16; late
chunks see ~0.005).

### 3.2 Decoder integration (`decoder.py`)

The decoder accepts either a scheduler **or** legacy lookup args:

```python
LDPC5GDecoder_soft(
    ldpc_enc, bp_schedule=[10,10,10],
    alpha=0.1, beta=0.1, k_payload=K_PAYLOAD, ...,
    # New API:
    sigma_scheduler=<scheduler>,
    # Legacy API (still supported):
    adaptive_sigma=True,
    syndrome_thresholds=(0.03, 0.10, 0.20),
    sigma_levels=(0.15, 0.25, 0.35, 0.45),
    sigma_min=None, sigma_max=None,
    monotonic_sigma=False,
    syndrome_hard_decision="xhat_gt0",
    syndrome_sign_debug=False,
)
```

The decoder contains **no** JSON parsing or threshold logic itself;
everything lives in `syndrome_sigma_schedule.py`.

### 3.3 CLI helpers (`cli_common.py`)

Use these in any new script:

```python
from cli_common import (
    add_adaptive_sigma_args,   # --sigma --sigma-min --sigma-max
                               # --syndrome-thresholds --adaptive-sigmas
                               # --sigma-lookup-json --monotonic-sigma
                               # --syndrome-hard-decision --syndrome-sign-debug
                               # --adaptive-sigma
    add_runtime_args,          # --batch --rounds --gpu-memory-mb
    decoder_sigma_kwargs,      # → (kwargs, info) for LDPC5GDecoder_soft
    early_gpu_mb_argv,         # parse --gpu-memory-mb before TF import
)
```

### 3.4 Calibrated lookup JSON v2 schema

Top-level keys (non-exhaustive — see `calibrate_sigma_lookup.py`):

```
schema_version (=2)         per_chunk (bool)
method                      objective (source_posterior_ber | tail_final_ber | tail_final_nack)
alpha, beta                 syndrome_hard_decision
schedule, ebno_list         collection_sigma, trace_tail_sigma
batch, rounds               candidate_sigmas
sigma_min, sigma_max        binning_type (quantile | manual)
monotonic_sigma             objective_notes
num_samples_total

# global lookup:
thresholds, sigma_levels, samples_per_bin, per_bin

# per-chunk lookup:
per_chunk_lookup = { "<chunk_idx>": {
    thresholds, sigma_levels, num_bins,
    per_bin: [ {bin_idx, bin_edges_open_closed, num_samples,
                candidate_objective_values, selected_sigma,
                selected_objective_value}, ... ]
}}
```

Loader (`load_sigma_lookup_json`) accepts both v1 (no `schema_version`,
top-level `thresholds` only) and v2.

---

## 4. Calibration objectives

| Objective | Use as | Notes |
|-----------|--------|-------|
| `source_posterior_ber` | proxy / diagnostics | Cheap, scores after a single denoiser step on `BP_post`.  **Do not use as the main calibrated lookup unless explicitly requested.** |
| `tail_final_ber` | surrogate | Runs the remaining BP chunks with `--trace-tail-sigma`, scores final payload BER. |
| `tail_final_nack` | **recommended** | Same tail trace, scores CRC NACK on final decoder output. |

`safe_sweep.sh` exposes both:

* `RUN_PROXY_CALIBRATE=1`  →  `source_posterior_ber` (also accepts `RUN_CALIBRATE=1` for backward compat)
* `RUN_TAIL_CALIBRATE=1`   →  `tail_final_nack`     (also accepts `RUN_RECALIBRATE=1` for backward compat)

---

## 5. Sigma range policy

For main experiments:

```
candidate_sigmas = 0.20  0.25  0.30  0.35  0.40
sigma_min        = 0.20
sigma_max        = 0.40
```

Wider ranges (e.g. 0.05–0.50) are **diagnostic-only** — `safe_sweep.sh`
defaults already enforce the narrow range.

---

## 6. Empirical status (as of this commit)

| Mode | Status |
|------|--------|
| Baseline BP (no denoiser) | reference; BLER ≈ 0.535 at 0.8 dB |
| Fixed σ = 0.3 (α=β=0.1) | **strongest**; BLER ≈ 0.030 at 0.8 dB |
| Hand-crafted adaptive σ | works, but **no clear win over fixed σ = 0.3** |
| Calibrated adaptive σ (`source_posterior_ber`, proxy) | works, no clear win |
| Calibrated adaptive σ (`tail_final_nack`, surrogate) | works, no clear win in smoke runs; large sweep not run |
| Monotonic σ repair | functional robustness option, not a BLER win by itself |

Treat adaptive σ as **architecturally clean but empirically unproven**
on this benchmark.

---

## 7. GPU stability rules

The host has experienced kernel hard-lockups during large GPU sweeps,
so:

* **Sequential only.**  Never launch concurrent GPU jobs.
* **Cap GPU memory.**  Always pass `--gpu-memory-mb <N>` (typical: 2048).
* **Use `safe_sweep.sh`** for any multi-step run; it handles timeouts,
  cooldowns, CUDA cleanup, thermal throttling, and process-collision
  detection.
* **No long sweeps in chat sessions.**  Do small smokes only:
  `--batch 16 --rounds 1` for sanity checks; `--batch 64 --rounds 2`
  for short comparisons.
* Use `--no-save-plot` and `--append-csv` to keep memory usage low.
* `cuda_cleanup.py` runs torch synchronize/empty_cache/ipc_collect
  between steps.
* Killing a hung run: send SIGTERM to the `python3` PID, then run
  `python3 cuda_cleanup.py`.

---

## 8. Smoke commands (≤ 1 min each, GPU-friendly)

These are the same checks used to verify the refactor.

### Sign convention (verifies `xhat_gt0` is correct)

```bash
CUDA_VISIBLE_DEVICES=0 python3 syndrome_diagnostics.py \
    --gpu-memory-mb 2048 \
    --alpha 0.1 --beta 0.1 --sigma 0.3 \
    --ebno 0.8 --batch 16 --rounds 1 \
    --syndrome-sign-debug \
    --output-prefix results/smoke_sign_debug
```
Expected: `gt0✓` markers (primary_better=True), gt0 ratio ≪ lt0 ratio.

### Fixed vs adaptive comparison

```bash
CUDA_VISIBLE_DEVICES=0 python3 plot_comparison.py \
    --gpu-memory-mb 2048 \
    --alpha 0.1 --beta 0.1 --sigma 0.3 \
    --ebno 0.8 --batch 16 --rounds 1 \
    --compare-adaptive --no-save-plot \
    --append-csv results/smoke_compare.csv \
    --sweep-tag smoke_compare_a0.1_b0.1
```

### Calibration (writes JSON v2 per-chunk lookup)

```bash
CUDA_VISIBLE_DEVICES=0 python3 calibrate_sigma_lookup.py \
    --gpu-memory-mb 2048 \
    --alpha 0.1 --beta 0.1 \
    --calibration-objective tail_final_nack \
    --collection-sigma 0.3 --trace-tail-sigma 0.3 \
    --candidate-sigmas 0.20 0.25 0.30 0.35 0.40 \
    --sigma-min 0.20 --sigma-max 0.40 \
    --batch 16 --rounds 1 --ebno 0.8 \
    --num-bins 3 --monotonic-sigma \
    --output-json results/smoke_tail.json
```

### Four-mode comparison (loads calibrated JSON)

```bash
CUDA_VISIBLE_DEVICES=0 python3 plot_comparison.py \
    --gpu-memory-mb 2048 \
    --alpha 0.1 --beta 0.1 --sigma 0.3 \
    --ebno 0.8 --batch 16 --rounds 1 \
    --compare-four-modes \
    --sigma-lookup-json results/smoke_tail.json \
    --monotonic-sigma --no-save-plot \
    --append-csv results/smoke_four.csv \
    --sweep-tag smoke_four
```

---

## 9. Common pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Computing syndrome on payload-only bits | Mean syndrome ratio ≈ 0.5 even at high SNR | Pass full graph-domain `x_hat` to `compute_syndrome` (the runtime assertion will catch this). |
| Flipping the sign default to `xhat_lt0` based on Sionna docs | Adaptive σ becomes useless (ratio stuck at ~0.57) | Keep `xhat_gt0` default; `xhat_lt0` is debug-only. |
| Using `source_posterior_ber` calibration as production | Calibrated σ may pick extremes (0.05 or 0.5) | Recalibrate with `tail_final_nack` and a narrow candidate range. |
| Running large sweeps from chat | Host kernel lockup | Use `safe_sweep.sh` from a tmux session, not from chat. |
| Forgetting `--gpu-memory-mb` | TF grabs all VRAM, instability | Always pass it (or set `FMNIST_GPU_MEM_MB` env var). |
| Editing `decoder.py` to parse JSON | Coupling drift | Parsing belongs in `syndrome_sigma_schedule.py` and `cli_common.py` only. |

---

## 10. Where to make changes

| If you want to … | Edit |
|------------------|------|
| Add a new scheduler shape | `syndrome_sigma_schedule.py` (subclass `_SchedulerBase`). |
| Tweak the calibration objective | `calibrate_sigma_lookup.py` (`_select_sigma_per_bin`). |
| Add a CLI flag used by multiple scripts | `cli_common.py`. |
| Change the BP / extrinsic flow | **Probably do not** — see invariants in §1. |
| Add a new run mode to plots | `plot_comparison.py` `decoders_named` dict. |
| Add a new safe-sweep step | `safe_sweep.sh` `run_one …` block. |

---

## 11. Recommended first actions for a new agent

1. Read `README.md` for the high-level pipeline.
2. Read `adaptive_sigma_review.md` for the adaptive-σ design and known status.
3. Skim `syndrome_sigma_schedule.py` and `cli_common.py` (small, central).
4. Run the sign-convention smoke (§8.1) to confirm the environment works.
5. Only then plan any algorithmic change.
