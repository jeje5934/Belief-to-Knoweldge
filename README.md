# onlyextrinsic_ada_sigma — Turbo-style BP + Source-Extrinsic Denoiser with Adaptive σ

5G LDPC decoding of **Fashion-MNIST images** over an AWGN channel,
augmented with a learned EDM source-denoiser.  The denoiser supplies
**source extrinsic information** (image-domain prior knowledge) that is
fed back into BP in a turbo-style loop.

This branch (`onlyextrinsic_ada_sigma`) extends the original
`onlyextrinsic` decoder with a clean **adaptive σ scheduling** module
inspired by DDECC.  After each BP chunk the syndrome ratio of the hard
decision is computed and used to choose the EDM denoiser σ via a
configurable lookup (fixed / hand-crafted / calibrated-from-JSON,
optionally per-chunk and/or monotonic).

> Quick orientation:
> * `README.md` (this file) — what the project is, how to run it.
> * `HANDOFF.md` — context handoff for a new chat / agent.
> * `adaptive_sigma_review.md` — design notes for adaptive σ + JSON schema.

---

## 1. Decoding equations (unchanged)

At each comma in the BP schedule (except after the last chunk):

```
bp_ext    = BP_post  − payload_intr        # BP extrinsic       (posterior − a priori)
src_ext   = src_post − BP_post             # source extrinsic   (posterior − denoiser input)
new_input = channel + β · bp_ext + α · src_ext
```

Where:
- `channel` = `payload0` = channel intrinsic LLR (frozen, never modified)
- `payload_intr` = the a priori LLR that was actually input to BP for this chunk
- `BP_post` = BP posterior LLR after the chunk's iterations
- `src_post` = denoiser posterior logits (denoiser takes `BP_post` as input,
  with σ chosen by the active scheduler)

Special cases:

| α | β | Effect |
|---|---|--------|
| 0 | 0 | `new_input = channel` (baseline BP, no denoiser) |
| 0.1 | 0 | `new_input = channel + 0.1·src_ext` (denoiser-only correction) |
| 0 | 1 | `new_input = channel + bp_ext` (turbo BP feedback, no denoiser) |
| **0.1** | **0.1** | **Best known setting, fixed σ = 0.3** |

The turbo equations are **invariant** in this branch — only how σ is
chosen changes.

---

## 2. Adaptive σ (DDECC-style)

After each BP chunk (except the last):

1. Hard-decide on the **full LDPC graph-domain** `x_hat`:
   `c_hat = (x_hat > 0)` (default — see §6 on sign convention).
2. Compute the syndrome `s = H · c_hat mod 2` and the
   `syndrome_ratio = sum(s) / num_cns`.
3. Map `syndrome_ratio` to σ via a **scheduler**:
   - `FixedSigmaScheduler` — scalar σ for all chunks.
   - `HandcraftedLookupScheduler` — piecewise lookup
     (`thresholds=(0.03, 0.10, 0.20)`, `sigma_levels=(0.15, 0.25, 0.35, 0.45)`).
   - `CalibratedLookupScheduler` — loaded from a JSON file produced by
     `calibrate_sigma_lookup.py`.  Supports **per-chunk** and **monotonic**
     options.
4. Run the EDM denoiser with that σ to get `src_post`.
5. Update `payload_intr` per the turbo equation above.

Read `adaptive_sigma_review.md` for a deeper discussion (DDECC analogy,
why the `xhat_gt0` default, calibration objectives, monotonic option,
JSON schema, current empirical status).

---

## 3. Folder / file overview

```
onlyextrinsic_ada_sigma/
├── README.md                       ← this file
├── HANDOFF.md                      cross-chat context handoff
├── adaptive_sigma_review.md        adaptive-σ design + JSON v2 schema
│
├── checkpoints/denoiser.pt         pretrained EDM denoiser (~27 MB)
├── score_denoiser/networks.py      EDM-preconditioned SongUNet (NVlabs/edm)
├── source_prior.py                 PyTorch SourcePriorDenoiser
├── denoiser.py                     TF SoftDenoiser (TF↔PyTorch bridge)
├── decoder.py                      LDPC5GDecoder_soft  (uses sigma_scheduler)
│
├── syndrome_sigma_schedule.py      canonical adaptive-σ module
├── cli_common.py                   shared CLI flags + scheduler builder
├── calibrate_sigma_lookup.py       offline σ-lookup calibration → JSON v2
│
├── plot_comparison.py              Eb/N0 sweep, baseline/fixed/adaptive/calibrated
├── experiment.py                   α sweep at β=0
├── syndrome_diagnostics.py         per-chunk syndrome stats across modes
├── visualize_progression.py        per-image step-by-step plots
├── sigma_sweep.py                  σ sweep at fixed (α, β)        [legacy]
├── train_denoiser.py               trains EDM denoiser   (do NOT run by default)
│
├── gpu_limits.py                   TF GPU memory cap helper
├── cuda_cleanup.py                 between-step CUDA / GC cleanup
├── safe_sweep.sh                   stability-first sequential runner
└── results/                        outputs (cursorignored)
    ├── *.png   *.json   *.csv   *.log
```

### File roles

| File | Role |
|------|------|
| `decoder.py` | Core algorithm: subclasses Sionna's `LDPC5GDecoder`, runs the multi-chunk BP schedule, calls `sigma_scheduler.select_sigma(idx, ratios)`, feeds the denoiser, updates `payload_intr`.  No JSON or threshold parsing. |
| `denoiser.py` | TF↔PyTorch bridge.  `SoftDenoiser` (Keras Layer) wraps `SourcePriorDenoiser` (PyTorch) and converts via NumPy.  Accepts a per-batch σ tensor or a scalar σ. |
| `source_prior.py` | PyTorch module: LLR → soft image → EDM denoise → posterior logits → extrinsic. |
| `score_denoiser/networks.py` | EDM-preconditioned SongUNet (~7 M params, model_channels=64). |
| **`syndrome_sigma_schedule.py`** | **Canonical adaptive-σ module.**  Hard-decision helpers, syndrome computation, scheduler classes (`FixedSigmaScheduler`, `HandcraftedLookupScheduler`, `CalibratedLookupScheduler`), monotonic-σ repair, JSON v1+v2 loader, diagnostics summarization. |
| **`cli_common.py`** | **Shared CLI helpers** for adaptive-σ scripts: `add_adaptive_sigma_args`, `add_runtime_args`, `decoder_sigma_kwargs`, `early_gpu_mb_argv`. |
| **`calibrate_sigma_lookup.py`** | Offline calibration of σ-lookup tables.  Default objective `tail_final_nack`, default candidate σ `0.20…0.40`, per-chunk lookup, optional `--monotonic-sigma`.  Writes JSON v2. |
| `plot_comparison.py` | Eb/N0 sweep.  Modes: default (baseline + fixed σ), `--adaptive-sigma`, `--compare-adaptive`, `--compare-four-modes`.  Optional `--no-save-plot` + `--append-csv` for stability. |
| `experiment.py` | α sweep at β = 0; supports the same fixed/adaptive/calibrated branches. |
| `syndrome_diagnostics.py` | Per-chunk syndrome stats (mean / std / range / σ distribution / sign-debug) for baseline, fixed, hand-crafted, and calibrated modes. |
| `visualize_progression.py` | Per-image, step-by-step decoding plots; useful for inspecting failures. |
| `safe_sweep.sh` | Stability-first sequential runner.  Caps CPU threads, GPU memory, run timeout, GPU temp, runs `cuda_cleanup.py` between steps.  Accepts `ALPHA`, `BETA`, `SIGMA`, `EBNO_LIST`, `MONOTONIC_SIGMA`, plus `RUN_PROXY_CALIBRATE` / `RUN_TAIL_CALIBRATE` (with legacy `RUN_CALIBRATE` / `RUN_RECALIBRATE` aliases). |

---

## 4. Quick start (smoke runs only)

The host has experienced kernel lockups during large GPU sweeps.  All
ad-hoc commands below are **smoke-sized** (BATCH=16, ROUNDS=1).  Use
`safe_sweep.sh` for anything larger.

### Sign-convention sanity check

```bash
CUDA_VISIBLE_DEVICES=0 python3 syndrome_diagnostics.py \
    --gpu-memory-mb 2048 \
    --alpha 0.1 --beta 0.1 --sigma 0.3 \
    --ebno 0.8 --batch 16 --rounds 1 \
    --syndrome-sign-debug \
    --output-prefix results/smoke_sign_debug
```

Expect `gt0✓` markers and `gt0` syndrome ratio ≪ `lt0` ratio at 0.8 dB.

### Fixed vs adaptive σ comparison

```bash
CUDA_VISIBLE_DEVICES=0 python3 plot_comparison.py \
    --gpu-memory-mb 2048 \
    --alpha 0.1 --beta 0.1 --sigma 0.3 \
    --ebno 0.8 --batch 16 --rounds 1 \
    --compare-adaptive --no-save-plot \
    --append-csv results/smoke.csv --sweep-tag smoke_a0.1_b0.1
```

### Calibrate a per-chunk σ lookup (recommended objective)

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

### Four-mode plot using calibrated JSON

```bash
CUDA_VISIBLE_DEVICES=0 python3 plot_comparison.py \
    --gpu-memory-mb 2048 \
    --alpha 0.1 --beta 0.1 --sigma 0.3 \
    --ebno 0.8 --batch 16 --rounds 1 \
    --compare-four-modes \
    --sigma-lookup-json results/smoke_tail.json \
    --monotonic-sigma --no-save-plot \
    --append-csv results/smoke.csv --sweep-tag smoke_four
```

### Per-image visualization

```bash
CUDA_VISIBLE_DEVICES=0 python3 visualize_progression.py \
    --gpu-memory-mb 2048 \
    --img_idx 3 --ebno 0.6 --alpha 0.1 --beta 0.1
```

---

## 5. Multi-step stability runner: `safe_sweep.sh`

For anything beyond a single smoke, drive runs through `safe_sweep.sh`
in a `tmux` session.  Override knobs via env vars:

```bash
ALPHA=0.1 BETA=0.1 SIGMA=0.3 \
EBNO_LIST="0.6 0.7 0.8 0.9 1.0" \
BATCH=64 ROUNDS=2 GPU_MB=2048 \
RUN_TAIL_CALIBRATE=1 \
MONOTONIC_SIGMA=1 \
./safe_sweep.sh
```

What it does sequentially (cooldown + `cuda_cleanup.py` between each):

1. (optional) `RUN_PROXY_CALIBRATE=1` → calibrate a `source_posterior_ber` lookup → `results/sigma_lookup_calibrated.json`.
2. (optional) `RUN_TAIL_CALIBRATE=1`  → calibrate a `tail_final_nack` lookup → `results/sigma_lookup_tail_nack.json`.
3. baseline + fixed σ sweep                    → `results/safe_sweep_scalar.csv` (sweep_tag `baseline_fixed_a*_b*`).
4. baseline + fixed + adaptive (hand-crafted)  → CSV (sweep_tag `compare_adaptive_a*_b*`).
5. four-mode plot using SIG_JSON if it exists  → CSV (sweep_tag `four_modes_proxy_a*_b*`).
6. four-mode plot using TAIL_JSON if it exists → CSV (sweep_tag `four_modes_tail_nack_a*_b*`).

`RUN_CALIBRATE=1` and `RUN_RECALIBRATE=1` continue to work as legacy
aliases for steps 1 and 2.

---

## 6. Sign convention — `xhat_gt0` is the default

Sionna's formal LLR convention is `log P(b=0)/P(b=1)`, which would
suggest `x_hat < 0 → bit = 1`.  Empirically on this LDPC graph
configuration, however:

| Convention   | Mean syndrome ratio at 1.2 dB | Interpretation |
|--------------|------------------------------:|---|
| `xhat_gt0`   | ≈ 0.018                       | Near zero — codeword consistent. |
| `xhat_lt0`   | ≈ 0.573                       | Large — inverted bit mapping. |

So `xhat_gt0` is the default in every script.  `xhat_lt0` is kept as a
debug option (`--syndrome-hard-decision xhat_lt0`).  Add
`--syndrome-sign-debug` to any run to print both conventions and which
one is better, without storing heavy per-sample arrays.

---

## 7. Calibrated-lookup JSON v2 — quick reference

`calibrate_sigma_lookup.py` writes JSON v2 with:

```
schema_version (=2)         per_chunk (bool)
method                      objective (source_posterior_ber | tail_final_ber | tail_final_nack)
alpha, beta                 syndrome_hard_decision
schedule, ebno_list         collection_sigma, trace_tail_sigma
batch, rounds               candidate_sigmas
sigma_min, sigma_max        binning_type (quantile | manual)
monotonic_sigma             objective_notes
num_samples_total

# global lookup keys (per_chunk=False):
thresholds, sigma_levels, samples_per_bin, per_bin

# per-chunk lookup keys (per_chunk=True, recommended):
per_chunk_lookup = {
    "<chunk_idx>": {
        "thresholds": [...], "sigma_levels": [...], "num_bins": int,
        "per_bin": [
            {"bin_idx", "bin_edges_open_closed", "num_samples",
             "candidate_objective_values", "selected_sigma",
             "selected_objective_value"},
            ...
        ]
    },
    ...
}
```

`load_sigma_lookup_json` accepts both v1 (no `schema_version`) and v2.

Recommended objective: **`tail_final_nack`**.  `source_posterior_ber`
is a cheap proxy and should not be used as the main calibrated lookup
unless explicitly requested.

---

## 8. Checkpoint

- **File**: `checkpoints/denoiser.pt`
- **Size**: ~27 MB
- **Format**: PyTorch `state_dict` of `SourcePriorDenoiser` (includes `net.*` keys for EDMPrecond and buffer keys for `bit_weights`, `bit_masks`, `pixel_values`)
- **Training data**: Fashion-MNIST train set (60K images, 28×28 grayscale)
- **Training loss**: EDM denoising score matching, `E[w(σ) · ||D(x+σn; σ) − x||²]`
- **Architecture**: SongUNet, model_channels=64, channel_mult=(1,2,2), num_blocks=2, attn_resolutions=(7,), sigma_data=0.5

The denoiser is **pretrained and not retrained** in this branch.  Do
not run `train_denoiser.py` unless explicitly asked.

---

## 9. Dependencies

- Python 3.8+
- TensorFlow 2.x (with Keras)
- PyTorch (CPU or CUDA)
- Sionna (LDPC encoder/decoder, mapper, AWGN channel)
- NumPy, Matplotlib, torchvision

The communication chain runs in **TensorFlow** (Sionna LDPC
encoder/decoder, mapper, AWGN); the denoiser runs in **PyTorch**
(EDM UNet).  The bridge is `denoiser.py`, which converts via NumPy.

---

## 10. Best known results

Eb/N0 = 0.8 dB, **σ = 0.3**, schedule = `[10, 10, 10]`, Fashion-MNIST
test set, BATCH = 200, ROUNDS = 5 → 1000 codewords:

| Setting | BLER | ΔACK vs baseline |
|---|---:|---:|
| Baseline BP (no denoiser) | 0.535 | — |
| α = 0.05, β = 0.05 | 0.072 | +463 |
| α = 0.1, β = 0   | 0.057 | +478 |
| α = 0.1, β = 0.05 | 0.035 | +500 |
| **α = 0.1, β = 0.1** | **0.030** | **+505** |

Eb/N0 sweep at the best α = 0.1, β = 0.1 (fixed σ = 0.3):

| Eb/N0 (dB) | Baseline BLER | Denoiser BLER |
|---:|---:|---:|
| 0.4 | 1.000 | 0.946 |
| 0.6 | 0.985 | 0.407 |
| 0.7 | 0.869 | 0.142 |
| 0.8 | 0.518 | 0.030 |
| 0.9 | 0.167 | 0.003 |
| 1.0 | 0.032 | 0.000 |

Waterfall shift: ~0.3 dB to the left.  At 0.8 dB: BLER 17×
improvement, BER 100× improvement.

> **Adaptive σ status.**  Hand-crafted and calibrated adaptive σ are
> implemented and exercised, but they are **not yet shown to beat the
> fixed σ = 0.3 baseline** above.  See `adaptive_sigma_review.md` for
> details.

---

## 11. Known assumptions and caveats

1. **Source extrinsic is approximate.**  `src_ext = src_post − BP_post`
   is the standard practical approximation; the EDM denoiser is
   non-linear, so this is not an exact turbo extrinsic in the
   information-theoretic sense.
2. **BP message state persists across chunks.**  `msg_v2c` is warm-started
   from chunk to chunk; BP does not fully restart.
3. **Channel LLR is frozen.**  `payload0` (the original channel
   reception) is always used as the base of the update equation.
4. **Denoiser σ is configurable but the *strongest baseline is still
   fixed σ = 0.3*.**  Adaptive σ schedulers (hand-crafted / calibrated)
   exist but have not yet improved on this baseline empirically.
5. **Fashion-MNIST only.**  The denoiser is trained on 28×28 grayscale
   Fashion-MNIST; applying to other image types requires retraining.
6. **BPSK (PAM-1).**  All experiments use `num_bits_per_symbol = 1`.
7. **Coderate ≈ 0.5.**  K_PAYLOAD = 6272, N_CODEWORD = 12600, with CRC24A.
8. **TF↔PyTorch bridge is CPU-bound.**  `tensor.numpy()` →
   `torch.from_numpy()` forces a CPU sync; correctness is unaffected.
9. **CRC false-positive rate.**  CRC24A has a small but nonzero
   probability of undetected errors; not accounted for in BLER.
10. **GPU stability.**  Large concurrent GPU jobs have caused host
    kernel lockups.  Always run sequentially via `safe_sweep.sh`,
    cap GPU memory with `--gpu-memory-mb`, and prefer
    `--no-save-plot --append-csv` for chained runs.

---

## 12. Recommended next steps

1. **Larger calibration sample.**  Repeat `tail_final_nack` calibration
   with a bigger sample set to see if a per-chunk lookup can finally
   beat fixed σ = 0.3.
2. **Joint sweep.**  Sweep α, β, σ jointly (all small, e.g.
   3 × 3 × 5) with ROUNDS ≥ 10 to reduce noise.
3. **End-to-end fine-tuning.**  Make the TF↔PyTorch bridge
   differentiable and jointly optimise denoiser weights with the turbo
   loop (currently the denoiser is frozen).
4. **Scale to larger codes / images.**  CIFAR-10 (32×32×3), other LDPC
   rates, etc.  Requires retraining the denoiser.
5. **Profile the bridge.**  A pure-PyTorch LDPC decoder would remove
   the per-chunk CPU sync.
6. **Three-way extrinsic.**  Decompose BP further if sub-components
   identifiable; currently BP is a single block.
