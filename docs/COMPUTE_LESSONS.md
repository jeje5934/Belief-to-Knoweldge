# Compute lessons — CPU vs GPU for this repo

Practical notes gathered while running the `pure-EP_practical` experiments
(low-SNR baseline sweep + denoiser training-state probe). These are about *how
to run*, not about the algorithm.

## 1. The decode sweep is CPU-bound, not GPU-bound

The decoder loop crosses the **TF ↔ PyTorch bridge** (`denoiser.py`) via NumPy
on every denoiser call — and the EP schedule `[2]×15` makes **14 denoiser calls
per decode**. The per-call NumPy round-trip (TF tensor → NumPy → torch → NumPy →
TF) dominates wall-clock.

Measured on the low-SNR baseline (batch 64):

| device | GPU utilization | wall-clock per SNR point |
|---|---|---|
| CPU-only (`CUDA_VISIBLE_DEVICES=""`) | — | ~comparable |
| GPU (`CUDA_VISIBLE_DEVICES=0`) | **~12 %** | ~comparable |

**GPU barely helps the sweep** — utilization sat at ~12 %, confirming the
bottleneck is the bridge/transfer, not UNet or BP compute. The repo note
(`README §6`, `HANDOFF.md`) "denoiser bridge is CPU-bound" is empirically true.

**Takeaway:** for the *decode sweep*, CPU-only is fine and avoids GPU risk. The
real speed lever is **reducing per-call bridge overhead** (keep tensors
on-device / batch the denoiser), not moving to GPU.

## 2. Long GPU *training* is crash-prone here — checkpoint & resume

Pure model training (`denoiser_training_probe.py`) **is** GPU-bound and much
faster on GPU (~26 s/epoch, GPU ~99 %). But a 100-epoch run **crashed at epoch
45** with a CUDA runtime fault:

```
RuntimeError: d.is_cuda() INTERNAL ASSERT FAILED at
".../c10/cuda/impl/CUDAGuardImpl.h":34
```

This matches the host's documented history of **GPU lockups / instability on
long runs**. The run had no resume path, so 45 epochs of progress would have
been lost if intermediate checkpoints hadn't been saved.

**Rules for long training runs:**
- Save **intermediate checkpoints** (`--save-at 5 20 50 100`) so a crash is
  recoverable — and write the metrics CSV **incrementally per epoch** (done).
- Never save probe/experiment weights over the working `checkpoints/denoiser.pt`
  — the probe asserts `ckpt_out != ref_ckpt`.
- Prefer a **resume-from-checkpoint** loop for anything > ~30 epochs, or run
  under `safe_sweep.sh` from a tmux session rather than from chat.
- If GPU faults recur, CPU training is slower but stable; the plateau conclusions
  don't need 100 epochs anyway (see below).

## 3. Stopping early is often justified

The training-state probe reached a **clear plateau by epoch ~20** (test MSE
0.0099 → 0.0096 from ep20→ep45). The epoch-45 crash did **not** change the
conclusion; forcing 100 epochs would have added instability risk for <3 % MSE
change. Decide "enough" from the curve, not from the target epoch count.

## 4. TF GPU memory growth

When a GPU is used, enable `tf.config.experimental.set_memory_growth(gpu, True)`
before TF allocates, so TF does not grab all 16 GB up-front (torch shares the
device). Both `practical_lowsnr_baseline.py` and the training probe do this /
run CPU-only. This prevents the OOM-style lockups.
