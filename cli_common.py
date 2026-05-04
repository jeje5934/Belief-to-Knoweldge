"""
Shared CLI helpers for adaptive sigma experiments.

Multiple scripts expose the same adaptive-sigma knobs (``--sigma``,
``--syndrome-thresholds``, ``--adaptive-sigmas``, ``--sigma-lookup-json``,
``--syndrome-hard-decision``, ``--syndrome-sign-debug``, GPU/batch flags,
``--monotonic-sigma`` …).  Centralising them here keeps the surface
consistent and avoids drift between ``experiment.py``,
``plot_comparison.py``, ``syndrome_diagnostics.py`` and friends.

Usage
-----
    import argparse
    from cli_common import add_adaptive_sigma_args, build_scheduler

    p = argparse.ArgumentParser()
    add_adaptive_sigma_args(p, fixed_sigma_default=0.3)
    args = p.parse_args()
    scheduler, info = build_scheduler(args)

The scheduler is one of FixedSigmaScheduler / HandcraftedLookupScheduler
/ CalibratedLookupScheduler (see ``syndrome_sigma_schedule``).  The
``info`` dict carries the lookup origin label, the calibrated JSON config
(if any), etc., so that downstream code can include it in result files.
"""

from __future__ import annotations

import argparse
from typing import Optional

from syndrome_sigma_schedule import (
    DEFAULT_SIGMA_LEVELS,
    DEFAULT_SYNDROME_HARD_DECISION,
    DEFAULT_SYNDROME_THRESHOLDS,
    CalibratedLookupScheduler,
    FixedSigmaScheduler,
    HandcraftedLookupScheduler,
    build_calibrated_scheduler_from_json,
    load_sigma_lookup_json,
)


# ──────────────────────────────────────────────────────────────────────
# CLI argument groups
# ──────────────────────────────────────────────────────────────────────

def add_adaptive_sigma_args(
    p: argparse.ArgumentParser,
    *,
    fixed_sigma_default: float = 0.3,
    include_adaptive_flag: bool = True,
) -> None:
    """Add the standard adaptive-sigma flags to an argparse parser.

    All scripts that interact with the σ schedule should call this so the
    argument names stay aligned across the project.
    """
    g = p.add_argument_group("adaptive sigma")
    if include_adaptive_flag:
        g.add_argument(
            "--adaptive-sigma", action="store_true",
            help="Use syndrome-based σ lookup instead of the fixed --sigma.",
        )
    g.add_argument(
        "--sigma", type=float, default=fixed_sigma_default,
        help="Fixed denoiser σ when adaptive scheduling is off.",
    )
    g.add_argument(
        "--sigma-min", type=float, default=None,
        help="Clamp lower bound on the σ produced by the lookup.",
    )
    g.add_argument(
        "--sigma-max", type=float, default=None,
        help="Clamp upper bound on the σ produced by the lookup.",
    )
    g.add_argument(
        "--syndrome-thresholds", type=float, nargs="+",
        default=list(DEFAULT_SYNDROME_THRESHOLDS),
        help="Hand-crafted lookup thresholds (ascending).",
    )
    g.add_argument(
        "--adaptive-sigmas", type=float, nargs="+",
        default=list(DEFAULT_SIGMA_LEVELS),
        help="Hand-crafted lookup σ levels, length = len(thresholds)+1.",
    )
    g.add_argument(
        "--sigma-lookup-json", default=None,
        help="Calibrated lookup JSON path. When set, overrides hand-crafted.",
    )
    g.add_argument(
        "--monotonic-sigma", action="store_true",
        help="Enforce that larger syndrome-ratio bins map to non-smaller σ "
             "(post-processing repair on the active lookup).",
    )
    g.add_argument(
        "--syndrome-hard-decision", default=DEFAULT_SYNDROME_HARD_DECISION,
        choices=["xhat_gt0", "xhat_lt0"],
        help="Sign convention for syndrome hard decision (default xhat_gt0).",
    )
    g.add_argument(
        "--syndrome-sign-debug", action="store_true",
        help="Compute syndrome under both sign conventions and report a "
             "summary; does NOT store heavy per-sample arrays.",
    )


def add_runtime_args(
    p: argparse.ArgumentParser,
    *,
    batch_default: int = 200,
    rounds_default: int = 5,
    add_gpu_mem: bool = True,
) -> None:
    """Add the standard runtime knobs (batch / rounds / GPU memory)."""
    g = p.add_argument_group("runtime")
    g.add_argument(
        "--batch", type=int, default=batch_default,
        help="Blocks per round.",
    )
    g.add_argument(
        "--rounds", type=int, default=rounds_default,
        help="Random rounds per Eb/N0.",
    )
    if add_gpu_mem:
        g.add_argument(
            "--gpu-memory-mb", type=int, default=None,
            help="GPU memory cap (MB).  Also picked up before TF import via "
                 "FMNIST_GPU_MEM_MB env var.",
        )


# ──────────────────────────────────────────────────────────────────────
# Scheduler construction
# ──────────────────────────────────────────────────────────────────────

def build_scheduler(args, *, force_adaptive: Optional[bool] = None):
    """Build a sigma scheduler from parsed CLI args.

    Parameters
    ----------
    args            : argparse.Namespace produced after add_adaptive_sigma_args.
    force_adaptive  : Override args.adaptive_sigma (e.g. for compare-mode where
                      one branch needs adaptive, another fixed).

    Returns
    -------
    (scheduler, info) where info is a dict with keys:
        origin               : "fixed" | "handcrafted" | "calibrated_json"
        json_path            : str | None
        per_chunk_calibrated : bool
        monotonic_applied    : bool
        thresholds           : tuple | None
        sigma_levels         : tuple | None
        sigma_min / sigma_max: float | None
        json_cfg             : dict | None  (raw load_sigma_lookup_json output)
    """
    adaptive = (
        force_adaptive
        if force_adaptive is not None
        else getattr(args, "adaptive_sigma", False)
    )

    info: dict = {
        "origin": None,
        "json_path": None,
        "per_chunk_calibrated": False,
        "monotonic_applied": False,
        "thresholds": None,
        "sigma_levels": None,
        "sigma_min": None,
        "sigma_max": None,
        "json_cfg": None,
    }

    if not adaptive:
        info["origin"] = "fixed"
        return FixedSigmaScheduler(args.sigma), info

    if getattr(args, "sigma_lookup_json", None):
        cfg = load_sigma_lookup_json(args.sigma_lookup_json)
        smin = args.sigma_min if args.sigma_min is not None else cfg["sigma_min"]
        smax = args.sigma_max if args.sigma_max is not None else cfg["sigma_max"]
        mono = bool(getattr(args, "monotonic_sigma", False)) or cfg["monotonic_sigma"]
        sched = build_calibrated_scheduler_from_json(
            args.sigma_lookup_json,
            sigma_min=smin, sigma_max=smax, monotonic=mono,
        )
        info.update({
            "origin": "calibrated_json",
            "json_path": args.sigma_lookup_json,
            "per_chunk_calibrated": cfg["per_chunk"],
            "monotonic_applied": mono,
            "thresholds": cfg["thresholds"],
            "sigma_levels": cfg["sigma_levels"],
            "sigma_min": smin,
            "sigma_max": smax,
            "json_cfg": cfg,
        })
        return sched, info

    sched = HandcraftedLookupScheduler(
        thresholds=tuple(args.syndrome_thresholds),
        sigma_levels=tuple(args.adaptive_sigmas),
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        monotonic=bool(getattr(args, "monotonic_sigma", False)),
    )
    info.update({
        "origin": "handcrafted",
        "monotonic_applied": bool(getattr(args, "monotonic_sigma", False)),
        "thresholds": sched.thresholds,
        "sigma_levels": sched.sigma_levels,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
    })
    return sched, info


def early_gpu_mb_argv(argv) -> Optional[int]:
    """Parse --gpu-memory-mb from sys.argv before TF is imported.

    Used by scripts to set the FMNIST_GPU_MEM_MB env var prior to TF
    initialisation.  Mirrors the previous _early_gpu_mb_argv() helpers.
    """
    for i, a in enumerate(argv):
        if a.startswith("--gpu-memory-mb="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return None
        if a == "--gpu-memory-mb" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return None
    return None


# ──────────────────────────────────────────────────────────────────────
# Decoder kwargs
# ──────────────────────────────────────────────────────────────────────

def decoder_sigma_kwargs(args, *, force_adaptive: Optional[bool] = None):
    """Build (kwargs, info) for ``LDPC5GDecoder_soft``.

    Returns
    -------
    kwargs : dict     Pass directly via ``LDPC5GDecoder_soft(..., **kwargs)``.
    info   : dict     Lookup-origin metadata (see :func:`build_scheduler`).

    Example
    -------
        kwargs, info = decoder_sigma_kwargs(args, force_adaptive=True)
        dec = LDPC5GDecoder_soft(ldpc_enc, ..., **kwargs)
    """
    sched, info = build_scheduler(args, force_adaptive=force_adaptive)
    kwargs = {
        "sigma_scheduler": sched,
        "syndrome_hard_decision": getattr(
            args, "syndrome_hard_decision", DEFAULT_SYNDROME_HARD_DECISION),
        "syndrome_sign_debug": bool(getattr(args, "syndrome_sign_debug", False)),
    }
    return kwargs, info
