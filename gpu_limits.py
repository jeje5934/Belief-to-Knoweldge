"""
TensorFlow GPU 설정 — 스윕 시 커널 OOM/워치독 완화용.

우선순위:
  1) 환경변수 ``FMNIST_GPU_MEM_MB`` (정수 MB) — 첫 GPU에 hard cap
  2) ``sys.argv`` 안의 ``--gpu-memory-mb N`` 또는 ``--gpu-memory-mb=N``
  3) 그 외: ``memory_growth=True`` 만 사용

주의: 일부 TF 빌드에서는 ``memory_growth``와 ``memory_limit`` 동시 설정이
불가할 수 있어, cap이 지정되면 growth는 시도하지 않습니다.
"""
from __future__ import annotations

import os
import sys
from typing import Optional


def _parse_gpu_memory_mb_from_argv() -> Optional[int]:
    for i, a in enumerate(sys.argv):
        if a.startswith("--gpu-memory-mb="):
            return int(a.split("=", 1)[1])
        if a == "--gpu-memory-mb" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return None


def setup_tensorflow_gpu() -> None:
    import tensorflow as tf

    env = os.environ.get("FMNIST_GPU_MEM_MB", "").strip()
    cap_mb = int(env) if env else None
    if cap_mb is None:
        cap_mb = _parse_gpu_memory_mb_from_argv()

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    try:
        if cap_mb is not None and cap_mb > 0:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(cap_mb))],
            )
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[gpu_limits] WARN: {e}; falling back to memory_growth only.")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
