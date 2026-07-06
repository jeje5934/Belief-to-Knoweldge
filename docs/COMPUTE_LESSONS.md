# Compute lessons — CPU vs GPU for this repo

Practical notes gathered while running the `pure-EP_practical` experiments
(low-SNR baseline sweep + denoiser training-state probe). These are about *how
to run*, not about the algorithm.

## 1. The denoiser dominates the decode — and it was silently running on CPU

The decoder loop crosses the **TF ↔ PyTorch bridge** (`denoiser.py`) on every
denoiser call; the EP schedule `[2]×15` makes **14 denoiser calls per decode**.
The **denoiser forward is the dominant per-decode cost.**

**CORRECTION (was wrong before).** An earlier version of this note claimed "GPU
barely helps (12 % util), the sweep is bridge-bound." That was a
**mis-diagnosis**: `SoftDenoiser` defaults to `device='cpu'` and the decoder
never overrode it (`denoiser.py:32`, `decoder.py`), so **even when we ran with
`CUDA_VISIBLE_DEVICES=0`, the torch denoiser ran on CPU** — only TF (BP/mapper)
used the GPU, hence 12 % util. "CPU run" and "GPU run" both had a **CPU
denoiser**, so of course they were comparable.

Benchmark (denoiser forward, batch 64, through the bridge):

| denoiser device | ms / call |
|---|---|
| CPU (uncontended) | **231** |
| CPU (while another CPU sweep hogs all cores) | 3757 |
| **GPU (RTX 4080)** | **14** |

So the GPU denoiser is genuinely **~16× faster per call** (231 → 14 ms). The
"260×" you get by benchmarking CPU *during* a running sweep is a **contention
artefact**, not the real ratio. Enable it with
`denoiser_kwargs=dict(device='cuda')` (see §5 for the safe recipe).

**Takeaway:** CPU is stable and, uncontended, tolerable (~231 ms/call → a
multi-config sweep is tens of minutes). GPU cuts the denoiser ~16× but needs the
process discipline in §5.

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
before TF allocates, so TF does not grab all 16 GB up-front. This prevents the
OOM-style lockups. (Moot under the §5 recipe, where TF is on CPU.)

## 5. GPU recipe for the decode path — TF on CPU, torch denoiser on GPU

**Mixing TF-CUDA and torch-CUDA in one process SEGFAULTS on this host** (exit
139; the `Unable to register cuDNN/cuFFT/cuBLAS factory` warnings are the tell —
both frameworks fight over the CUDA context). This is what the earlier "GPU
lockups / crashes" actually were. The stable, fast recipe:

```python
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")          # TF (BP/mapper/AWGN) on CPU
import torch                                       # torch keeps the GPU
DENOISER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ...build every soft decoder with:
LDPC5GDecoder_soft(..., denoiser_kwargs=dict(device=DENOISER_DEVICE))
```
Run with `CUDA_VISIBLE_DEVICES=0`. Verified: a single GPU-denoiser decoder with
TF on CPU decodes fine (no segfault); the ~16× denoiser speedup applies.

**Hard rules (violating either segfaults):**
1. **Never let TF and torch both touch CUDA** in one process — hide the GPU from
   TF (`set_visible_devices([], "GPU")`).
2. **Never build both a CPU-denoiser and a CUDA-denoiser decoder in the same
   process** — a cpu/cuda torch mix crashes too. Pick one device per process
   (to compare CPU vs GPU output, use two separate processes).

CPU↔GPU denoiser outputs differ only by float epsilon → occasional single-bit
flips, decode quality equivalent (already seen as ~0.115 vs 0.126 run-to-run).

Training (`denoiser_training_probe.py`, `train_denoiser_trouser.py`) is pure
torch (no TF in the hot loop) so it can use CUDA directly — that path is
genuinely GPU-bound and fast, but still crash-prone on very long runs, so keep
intermediate checkpoints + incremental CSV (§2).
