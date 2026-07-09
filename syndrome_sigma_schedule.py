"""
Syndrome-weighted denoiser sigma scheduling (DDECC-style, inference-only).

This is the **canonical module** for adaptive sigma logic.  It exposes:

  * Hard-decision and syndrome helpers
        - hard_bits_from_logits()
        - compute_syndrome()
        - compute_syndrome_weight_and_ratio()

  * Schedulers (object-oriented, used by the decoder)
        - FixedSigmaScheduler            : returns scalar σ everywhere
        - HandcraftedLookupScheduler     : piecewise lookup σ(syndrome_ratio)
        - CalibratedLookupScheduler      : JSON-loaded lookup, optionally
                                           per-chunk and/or monotonic

  * Lookup utilities
        - sigma_from_syndrome_ratio()    : raw piecewise gather
        - validate_lookup()
        - enforce_monotonic_sigma()      : optional monotonic repair
        - load_sigma_lookup_json()       : v1 + v2 schema loader

  * Diagnostics
        - lookup_sigma_distribution()
        - summarize_chunk_diagnostics()

The decoder imports schedulers from here only; it does not parse JSON or
deal with bin thresholds directly.

────────────────────────────────────────────────────────────────────────
Sign convention
────────────────────────────────────────────────────────────────────────
Sionna's formal LLR convention is log P(b=0)/P(b=1), which in isolation
would suggest x_hat < 0 → bit=1.  However, end-to-end syndrome checks at
high Eb/N0 with this LDPC graph configuration consistently show:

    xhat_gt0 mean syndrome ratio ≈ 0.018   (near zero — codeword consistent)
    xhat_lt0 mean syndrome ratio ≈ 0.573   (large    — inverted mapping)

The default in this codebase is therefore "xhat_gt0".  Use the
``--syndrome-sign-debug`` flag (in any CLI tool) to re-verify if the
underlying Sionna version or LDPC config changes.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf


# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_SYNDROME_THRESHOLDS = (0.03, 0.10, 0.20)
DEFAULT_SIGMA_LEVELS = (0.15, 0.25, 0.35, 0.45)
DEFAULT_SYNDROME_HARD_DECISION = "xhat_gt0"

# Recommended main-experiment sigma range (matches the strong fixed=0.3
# baseline neighbourhood).  Wider ranges are allowed for diagnostics only.
DEFAULT_MAIN_SIGMA_MIN = 0.20
DEFAULT_MAIN_SIGMA_MAX = 0.40
DEFAULT_MAIN_CANDIDATE_SIGMAS = (0.20, 0.25, 0.30, 0.35, 0.40)

# Current calibrated-lookup JSON schema version produced by
# calibrate_sigma_lookup.py.  Older files (no schema_version key) are
# treated as v1 and still load correctly.
SIGMA_LOOKUP_SCHEMA_VERSION = 2


# ──────────────────────────────────────────────────────────────────────
# Sparse PCM helper
# ──────────────────────────────────────────────────────────────────────

def csr_matrix_to_sparse_tensor(csr) -> tf.SparseTensor:
    """Convert a scipy CSR PCM into a float32 SparseTensor [num_cns, num_vns]."""
    coo = csr.tocoo()
    idx = np.stack([coo.row, coo.col], axis=1).astype(np.int64)
    vals = np.ones(len(coo.row), dtype=np.float32)
    st = tf.sparse.SparseTensor(idx, vals, dense_shape=tuple(csr.shape))
    return tf.sparse.reorder(st)


# ──────────────────────────────────────────────────────────────────────
# Hard decision + syndrome
# ──────────────────────────────────────────────────────────────────────

def hard_bits_from_logits(
    x_hat: tf.Tensor,
    decision_rule: str = DEFAULT_SYNDROME_HARD_DECISION,
) -> tf.Tensor:
    """Convert BP logits to hard bits {0, 1} for syndrome computation.

    decision_rule:
      * "xhat_gt0": hard_bit = 1 iff x_hat >  0  (default — empirically correct)
      * "xhat_lt0": hard_bit = 1 iff x_hat <  0  (debug only)
    """
    if decision_rule == "xhat_gt0":
        return tf.cast(x_hat > 0.0, tf.float32)
    if decision_rule == "xhat_lt0":
        return tf.cast(x_hat < 0.0, tf.float32)
    raise ValueError(f"Unsupported decision_rule: {decision_rule}")


def _assert_syndrome_domain(hard_bits: tf.Tensor, h_sparse: tf.SparseTensor) -> None:
    num_vns_h = tf.cast(h_sparse.dense_shape[1], tf.int32)
    num_vns_bits = tf.shape(hard_bits)[-1]
    tf.debugging.assert_equal(
        num_vns_h,
        num_vns_bits,
        message=(
            "Syndrome domain mismatch: H.shape[1] != hard_bits.shape[-1].  "
            "The syndrome must be computed on the full LDPC graph-domain "
            "codeword (same length as the parity-check matrix), not on the "
            "payload / systematic / rate-matched slice."
        ),
    )


def compute_syndrome(
    hard_bits: tf.Tensor,
    h_sparse: tf.SparseTensor,
) -> tf.Tensor:
    """Compute the syndrome vector s = H · c_hat  mod 2.

    Parameters
    ----------
    hard_bits : [B, num_vns]  float / 0-1 tensor of full graph-domain bits.
    h_sparse  : [num_cns, num_vns]  sparse parity-check matrix.

    Returns
    -------
    syndrome : [B, num_cns] int32, entries in {0, 1}.
    """
    _assert_syndrome_domain(hard_bits, h_sparse)
    ct = tf.transpose(hard_bits)                       # [num_vns, B]
    s = tf.sparse.sparse_dense_matmul(h_sparse, ct)    # [num_cns, B]
    s_int = tf.cast(tf.math.round(s), tf.int32)
    synd = tf.math.floormod(s_int, 2)
    return tf.transpose(synd)                          # [B, num_cns]


def _syndrome_count_from_vec(syndrome: tf.Tensor) -> tf.Tensor:
    """Per-codeword count of unsatisfied checks from a [B, num_cns] vector."""
    return tf.reduce_sum(syndrome, axis=-1)


def compute_syndrome_weight_and_ratio(
    x_hat: tf.Tensor,
    h_sparse: tf.SparseTensor,
    decision_rule: str = DEFAULT_SYNDROME_HARD_DECISION,
    compare_both_signs: bool = False,
):
    """Return (counts, ratios, sign_debug).

    Parameters
    ----------
    x_hat       : [B, num_vns] BP marginal logits (full graph domain).
    h_sparse    : [num_cns, num_vns] parity-check matrix in sparse form.
    decision_rule, compare_both_signs : sign-convention controls.

    Returns
    -------
    counts      : [B] int32 — unsatisfied-parity count per codeword.
    ratios      : [B] float32 — counts / num_cns.
    sign_debug  : dict | None — populated only when ``compare_both_signs``.
                  Contains both conventions' counts and ratios.
    """
    c = hard_bits_from_logits(x_hat, decision_rule=decision_rule)
    syndrome = compute_syndrome(c, h_sparse)
    counts = _syndrome_count_from_vec(syndrome)
    num_cns = tf.cast(h_sparse.dense_shape[0], tf.float32)
    ratios = tf.cast(counts, tf.float32) / num_cns

    if not compare_both_signs:
        return counts, ratios, None

    alt_rule = "xhat_lt0" if decision_rule == "xhat_gt0" else "xhat_gt0"
    c_alt = hard_bits_from_logits(x_hat, decision_rule=alt_rule)
    counts_alt = _syndrome_count_from_vec(compute_syndrome(c_alt, h_sparse))
    ratios_alt = tf.cast(counts_alt, tf.float32) / num_cns
    sign_debug = {
        "primary_rule": decision_rule,
        "primary_counts": counts,
        "primary_ratios": ratios,
        "alt_rule": alt_rule,
        "alt_counts": counts_alt,
        "alt_ratios": ratios_alt,
    }
    return counts, ratios, sign_debug


# Back-compatible alias (used by older imports inside this repo).
syndrome_stats = compute_syndrome_weight_and_ratio


# ──────────────────────────────────────────────────────────────────────
# Lookup primitives
# ──────────────────────────────────────────────────────────────────────

def validate_lookup(thresholds: Sequence, sigma_levels: Sequence) -> None:
    if len(sigma_levels) != len(thresholds) + 1:
        raise ValueError(
            "sigma_levels must have length len(thresholds)+1, "
            f"got thresholds={len(thresholds)} sigma_levels={len(sigma_levels)}"
        )
    thr = list(thresholds)
    if thr != sorted(thr):
        raise ValueError("syndrome thresholds must be strictly sorted ascending")


def sigma_from_syndrome_ratio(
    ratios: tf.Tensor,
    thresholds: Sequence[float],
    sigma_levels: Sequence[float],
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
) -> tf.Tensor:
    """Piecewise-constant σ lookup from syndrome ratio.

    Bin i: use sigma_levels[i] where i = searchsorted(thresholds, ratio, "right").
    """
    thr = tf.constant(list(thresholds), dtype=tf.float32)
    sig = tf.constant(list(sigma_levels), dtype=tf.float32)
    r = tf.cast(ratios, tf.float32)
    idx = tf.searchsorted(thr, r, side="right")
    sigma = tf.gather(sig, idx)
    if sigma_min is not None or sigma_max is not None:
        lo = -np.inf if sigma_min is None else float(sigma_min)
        hi = np.inf if sigma_max is None else float(sigma_max)
        sigma = tf.clip_by_value(sigma, lo, hi)
    return sigma


def enforce_monotonic_sigma(sigma_levels: Sequence[float]) -> tuple:
    """Repair a per-bin σ vector so that larger bin index ⇒ no smaller σ.

    Bins are assumed to be ordered by increasing syndrome-ratio threshold,
    which means a higher index corresponds to *more* unsatisfied checks
    and the denoiser should not back off to a *smaller* sigma.

        repaired[i] = max(repaired[i-1], sigma_levels[i])
    """
    out = []
    prev = -np.inf
    for s in sigma_levels:
        cur = max(prev, float(s))
        out.append(cur)
        prev = cur
    return tuple(out)


def lookup_sigma_distribution(sigma_tensor: tf.Tensor) -> dict:
    """Histogram a batch sigma vector → {sigma_value(str): count}."""
    vals = tf.cast(tf.reshape(sigma_tensor, [-1]), tf.float32).numpy()
    uniq, cnt = np.unique(np.round(vals, 6), return_counts=True)
    return {f"{float(u):.6f}": int(c) for u, c in zip(uniq, cnt)}


# ──────────────────────────────────────────────────────────────────────
# Schedulers
# ──────────────────────────────────────────────────────────────────────

class _SchedulerBase:
    """Common interface used by the decoder."""

    name = "base"
    is_adaptive = False

    def select_sigma(self, chunk_idx: int, ratios: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError


class FixedSigmaScheduler(_SchedulerBase):
    """Always returns a scalar sigma broadcast to batch.

    The decoder may also choose to *not* call this scheduler at all and
    instead pass the denoiser a None sigma; both behaviours are equivalent.
    """

    name = "fixed"
    is_adaptive = False

    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    def select_sigma(self, chunk_idx: int, ratios: tf.Tensor) -> tf.Tensor:
        B = tf.shape(ratios)[0]
        return tf.fill([B], tf.cast(self.sigma, tf.float32))


class HandcraftedLookupScheduler(_SchedulerBase):
    """Hand-crafted piecewise σ lookup over normalized syndrome ratio."""

    name = "handcrafted"
    is_adaptive = True

    def __init__(
        self,
        thresholds: Sequence[float] = DEFAULT_SYNDROME_THRESHOLDS,
        sigma_levels: Sequence[float] = DEFAULT_SIGMA_LEVELS,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        monotonic: bool = False,
    ):
        validate_lookup(thresholds, sigma_levels)
        self.thresholds = tuple(float(x) for x in thresholds)
        self.sigma_levels = tuple(float(x) for x in sigma_levels)
        if monotonic:
            self.sigma_levels = enforce_monotonic_sigma(self.sigma_levels)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.monotonic = bool(monotonic)

    def select_sigma(self, chunk_idx: int, ratios: tf.Tensor) -> tf.Tensor:
        return sigma_from_syndrome_ratio(
            ratios, self.thresholds, self.sigma_levels,
            sigma_min=self.sigma_min, sigma_max=self.sigma_max,
        )


class CalibratedLookupScheduler(_SchedulerBase):
    """Calibrated lookup loaded from a JSON file.

    Two modes:
      * Global lookup     : single (thresholds, sigma_levels) for all chunks.
      * Per-chunk lookup  : separate (thresholds, sigma_levels) per chunk_idx.
                            If a chunk index is missing (e.g. final chunk),
                            falls back to the largest available chunk ≤ idx,
                            and finally to the smallest available chunk.

    Optional monotonic repair:
        sigma_levels[i] = max(sigma_levels[i], sigma_levels[i-1])   per chunk.
    """

    name = "calibrated"
    is_adaptive = True

    def __init__(
        self,
        *,
        thresholds: Optional[Sequence[float]] = None,
        sigma_levels: Optional[Sequence[float]] = None,
        per_chunk: Optional[dict] = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        monotonic: bool = False,
        source: str = "json",
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.monotonic = bool(monotonic)
        self.source = source

        if per_chunk:
            tidy = {}
            for k, v in per_chunk.items():
                ci = int(k)
                thr = tuple(float(x) for x in v["thresholds"])
                sig = tuple(float(x) for x in v["sigma_levels"])
                validate_lookup(thr, sig)
                if monotonic:
                    sig = enforce_monotonic_sigma(sig)
                tidy[ci] = {"thresholds": thr, "sigma_levels": sig}
            if not tidy:
                raise ValueError("per_chunk lookup is empty")
            self.per_chunk = tidy
            self.thresholds = None
            self.sigma_levels = None
        else:
            if thresholds is None or sigma_levels is None:
                raise ValueError(
                    "CalibratedLookupScheduler requires either per_chunk or "
                    "(thresholds, sigma_levels)."
                )
            validate_lookup(thresholds, sigma_levels)
            self.thresholds = tuple(float(x) for x in thresholds)
            self.sigma_levels = tuple(float(x) for x in sigma_levels)
            if monotonic:
                self.sigma_levels = enforce_monotonic_sigma(self.sigma_levels)
            self.per_chunk = None

    @property
    def is_per_chunk(self) -> bool:
        return self.per_chunk is not None

    def _get_chunk_lookup(self, chunk_idx: int):
        avail = sorted(self.per_chunk.keys())
        candidates = [k for k in avail if k <= int(chunk_idx)]
        ci_use = candidates[-1] if candidates else avail[0]
        d = self.per_chunk[ci_use]
        return d["thresholds"], d["sigma_levels"]

    def select_sigma(self, chunk_idx: int, ratios: tf.Tensor) -> tf.Tensor:
        if self.is_per_chunk:
            thr, sig = self._get_chunk_lookup(chunk_idx)
        else:
            thr, sig = self.thresholds, self.sigma_levels
        return sigma_from_syndrome_ratio(
            ratios, thr, sig,
            sigma_min=self.sigma_min, sigma_max=self.sigma_max,
        )


class AnnealingSigmaScheduler(_SchedulerBase):
    """Open-loop denoiser-σ annealing: σ is a deterministic function of the BP
    chunk index t (0..T-1), independent of the syndrome/cavity state.

    Motivation (practical_sigma [A]): start at a LARGE σ so the denoiser
    posterior is smoothed and several image modes stay on the table, then DECAY
    to a small σ so the modes separate and the late chunks commit.  This is the
    dual of the failed *increasing-α* schedule: lowering σ actually splits the
    modes (so committing is meaningful), whereas increasing α injected hard
    before the modes were separable.

        linear : σ_t = σ_s − (σ_s − σ_e)·t/(T−1)
        geom   : σ_t = σ_s·(σ_e/σ_s)^{t/(T−1)}

    Only ``chunk_idx`` is used; ``ratios`` is ignored (open-loop).  ``T`` is the
    number of denoiser chunks (schedule length − 1 in EP, since the final chunk
    has no denoiser); indices ≥ T−1 clamp to σ_e.
    """

    name = "annealing"
    is_adaptive = True     # varies per chunk → decoder must call select_sigma

    def __init__(self, sigma_start: float, sigma_end: float, n_chunks: int,
                 mode: str = "linear"):
        if mode not in ("linear", "geom"):
            raise ValueError("mode must be 'linear' or 'geom'")
        if mode == "geom" and (sigma_start <= 0 or sigma_end <= 0):
            raise ValueError("geom annealing needs positive sigmas")
        self.sigma_start = float(sigma_start)
        self.sigma_end = float(sigma_end)
        self.n_chunks = max(int(n_chunks), 1)
        self.mode = mode
        self.path = tuple(self._sigma_at(t) for t in range(self.n_chunks))

    def _sigma_at(self, t: int) -> float:
        if self.n_chunks == 1:
            return self.sigma_end
        f = min(max(t, 0), self.n_chunks - 1) / (self.n_chunks - 1)
        if self.mode == "linear":
            return self.sigma_start - (self.sigma_start - self.sigma_end) * f
        return self.sigma_start * (self.sigma_end / self.sigma_start) ** f

    def select_sigma(self, chunk_idx: int, ratios: tf.Tensor) -> tf.Tensor:
        s = self._sigma_at(int(chunk_idx))
        B = tf.shape(ratios)[0]
        return tf.fill([B], tf.cast(s, tf.float32))


# ──────────────────────────────────────────────────────────────────────
# JSON loading (v1 + v2 aware)
# ──────────────────────────────────────────────────────────────────────

def load_sigma_lookup_json(path: str) -> dict:
    """Load a calibrated lookup JSON, returning a dict suitable to construct
    a :class:`CalibratedLookupScheduler` and to inspect schedule metadata.

    Returned dict keys (always present):
      - schema_version           : int
      - per_chunk                : bool
      - thresholds, sigma_levels : tuple[float] | None  (global lookup)
      - per_chunk_lookup         : dict[int, {thresholds, sigma_levels}] | None
      - sigma_min, sigma_max     : float | None
      - monotonic_sigma          : bool
      - syndrome_hard_decision   : str | None
      - objective                : str | None
      - alpha, beta              : float | None
      - candidate_sigmas         : tuple[float] | None
      - raw                      : the full parsed JSON (for diagnostics)
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    schema_version = int(cfg.get("schema_version", 1))

    per_chunk_lookup = None
    if cfg.get("per_chunk", False) and "per_chunk_lookup" in cfg:
        raw_pc = cfg["per_chunk_lookup"]
        per_chunk_lookup = {}
        for k, v in raw_pc.items():
            thr = v.get("thresholds")
            sig = v.get("sigma_levels")
            if thr is None or sig is None:
                raise ValueError(
                    f"per_chunk_lookup chunk {k}: must contain "
                    "'thresholds' and 'sigma_levels'."
                )
            validate_lookup(thr, sig)
            per_chunk_lookup[int(k)] = {
                "thresholds": tuple(float(x) for x in thr),
                "sigma_levels": tuple(float(x) for x in sig),
            }

    if per_chunk_lookup is not None:
        thresholds = None
        sigma_levels = None
    else:
        thr = cfg.get("thresholds", cfg.get("syndrome_thresholds", None))
        sig = cfg.get("sigma_levels", cfg.get("adaptive_sigmas", None))
        if thr is None or sig is None:
            raise ValueError(
                f"Lookup JSON {path}: must include either 'per_chunk_lookup' "
                "or top-level ('thresholds' and 'sigma_levels')."
            )
        validate_lookup(thr, sig)
        thresholds = tuple(float(x) for x in thr)
        sigma_levels = tuple(float(x) for x in sig)

    sigma_min = cfg.get("sigma_min", None)
    sigma_max = cfg.get("sigma_max", None)
    monotonic = bool(cfg.get("monotonic_sigma", False))
    objective = cfg.get("objective", cfg.get("calibration_objective", None))

    return {
        "schema_version": schema_version,
        "per_chunk": per_chunk_lookup is not None,
        "thresholds": thresholds,
        "sigma_levels": sigma_levels,
        "per_chunk_lookup": per_chunk_lookup,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "monotonic_sigma": monotonic,
        "syndrome_hard_decision": cfg.get("syndrome_hard_decision", None),
        "objective": objective,
        "alpha": cfg.get("alpha", None),
        "beta": cfg.get("beta", None),
        "candidate_sigmas": (
            tuple(float(x) for x in cfg["candidate_sigmas"])
            if "candidate_sigmas" in cfg else None
        ),
        "raw": cfg,
    }


def build_calibrated_scheduler_from_json(
    path: str,
    *,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    monotonic: Optional[bool] = None,
) -> CalibratedLookupScheduler:
    """Convenience wrapper: load JSON and instantiate a scheduler.

    CLI overrides take precedence:
      - sigma_min/sigma_max: overrides JSON values (if not None).
      - monotonic         : overrides JSON value (if not None).
    """
    cfg = load_sigma_lookup_json(path)
    smin = sigma_min if sigma_min is not None else cfg["sigma_min"]
    smax = sigma_max if sigma_max is not None else cfg["sigma_max"]
    mono = bool(monotonic) if monotonic is not None else cfg["monotonic_sigma"]
    if cfg["per_chunk"]:
        return CalibratedLookupScheduler(
            per_chunk=cfg["per_chunk_lookup"],
            sigma_min=smin, sigma_max=smax, monotonic=mono,
            source=path,
        )
    return CalibratedLookupScheduler(
        thresholds=cfg["thresholds"], sigma_levels=cfg["sigma_levels"],
        sigma_min=smin, sigma_max=smax, monotonic=mono,
        source=path,
    )


# ──────────────────────────────────────────────────────────────────────
# Diagnostics summarization
# ──────────────────────────────────────────────────────────────────────

def summarize_chunk_diagnostics(chunk_diags: list[dict]) -> list[dict]:
    """Aggregate per-chunk diagnostics over multiple decode calls.

    If sign_debug is present in any row (from --syndrome-sign-debug), also
    aggregates the alternative-convention syndrome ratios so they appear
    in the JSON/CSV output.

    Heavy per-sample arrays are NOT stored unless explicitly requested
    via the diagnostics input itself.
    """
    if not chunk_diags:
        return []
    num_chunks = max(int(d["chunk_idx"]) for d in chunk_diags) + 1
    out = []
    for cidx in range(num_chunks):
        rows = [d for d in chunk_diags if int(d["chunk_idx"]) == cidx]
        if not rows:
            continue
        ratios_all, weights_all, sigma_all = [], [], []
        sigma_dist: dict = {}
        alt_ratios_all: list = []
        # EP convergence/evidence metrics (Task 6; present only in EP mode).
        ep_float_keys = ("src_site_l2", "src_site_delta_l2", "src_site_max_abs",
                         "source_logZ", "source_free_energy")
        ep_vals: dict = {k: [] for k in ep_float_keys}
        ep_any_diverged = False

        for d in rows:
            r = tf.cast(d["syndrome_ratio"], tf.float32).numpy().reshape(-1)
            w = tf.cast(d["syndrome_weight"], tf.float32).numpy().reshape(-1)
            s = tf.cast(d["sigma"], tf.float32).numpy().reshape(-1)
            ratios_all.append(r)
            weights_all.append(w)
            sigma_all.append(s)
            for k, v in lookup_sigma_distribution(d["sigma"]).items():
                sigma_dist[k] = sigma_dist.get(k, 0) + int(v)
            for k in ep_float_keys:
                if d.get(k) is not None:
                    ep_vals[k].append(float(d[k]))
            if d.get("ep_diverged"):
                ep_any_diverged = True

            sd = d.get("sign_debug")
            if sd is not None and isinstance(sd, dict):
                alt_r = sd.get("alt_ratios")
                if alt_r is not None:
                    try:
                        alt_ratios_all.append(
                            tf.cast(alt_r, tf.float32).numpy().reshape(-1)
                        )
                    except Exception:
                        pass

        ratios = np.concatenate(ratios_all, axis=0)
        weights = np.concatenate(weights_all, axis=0)
        sigmas = np.concatenate(sigma_all, axis=0)

        entry = {
            "chunk_idx": cidx,
            "mean_syndrome_weight": float(np.mean(weights)),
            "mean_syndrome_ratio": float(np.mean(ratios)),
            "std_syndrome_ratio": float(np.std(ratios)),
            "min_syndrome_ratio": float(np.min(ratios)),
            "max_syndrome_ratio": float(np.max(ratios)),
            "mean_sigma": float(np.mean(sigmas)),
            "sigma_distribution": sigma_dist,
            "num_samples": int(ratios.size),
        }
        if alt_ratios_all:
            alt_ratios = np.concatenate(alt_ratios_all, axis=0)
            entry["sign_debug"] = {
                "primary_rule": "xhat_gt0",
                "alt_rule": "xhat_lt0",
                "primary_mean_ratio": float(np.mean(ratios)),
                "alt_mean_ratio": float(np.mean(alt_ratios)),
                "primary_better": bool(np.mean(ratios) <= np.mean(alt_ratios)),
            }
        # Attach aggregated EP metrics if this chunk carried them (EP mode).
        if any(ep_vals[k] for k in ep_float_keys):
            ep_summary = {f"mean_{k}": float(np.mean(ep_vals[k]))
                          for k in ep_float_keys if ep_vals[k]}
            ep_summary["ep_diverged"] = bool(ep_any_diverged)
            entry["ep_metrics"] = ep_summary
        out.append(entry)
    return out
