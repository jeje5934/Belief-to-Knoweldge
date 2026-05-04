"""
CUDA / memory cleanup helper — run between sweep steps.

  python3 cuda_cleanup.py

Actions:
  1. torch.cuda.synchronize()   — flush pending GPU ops
  2. torch.cuda.empty_cache()   — release cached but unused VRAM
  3. torch.cuda.ipc_collect()   — reclaim IPC shared tensors
  4. gc.collect() × 3           — Python garbage collection (3 passes)

Prints a brief nvidia-smi snapshot (if available) before and after.
"""
import gc
import os
import subprocess
import sys


def _smi_snapshot(label: str) -> None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode()
        for line in out.strip().splitlines():
            print(f"[smi/{label}] {line.strip()}", flush=True)
    except Exception as e:
        print(f"[smi/{label}] unavailable: {e}", flush=True)


def main() -> None:
    _smi_snapshot("before")

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("[cleanup] torch.cuda: synchronize / empty_cache / ipc_collect done", flush=True)
        else:
            print("[cleanup] torch.cuda not available — skipping GPU ops", flush=True)
    except ImportError:
        print("[cleanup] torch not importable — skipping GPU ops", flush=True)
    except Exception as e:
        print(f"[cleanup] torch cuda cleanup error: {e}", flush=True)

    for i in range(3):
        collected = gc.collect()
        print(f"[cleanup] gc.collect() pass {i+1}: {collected} objects collected", flush=True)

    _smi_snapshot("after")
    print("[cleanup] done", flush=True)


if __name__ == "__main__":
    main()
