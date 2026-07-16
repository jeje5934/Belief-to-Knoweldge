"""
LDPC5GDecoder_soft — BP + source-extrinsic denoiser.

[onlyextrinsic variant — independent alpha/beta, turbo-style extrinsic]

  Turbo-style extrinsic definitions
  ---------------------------------
  At each comma in the schedule, let ``payload_intr`` be the current a priori
  input that was fed into BP for this chunk.  Then:

    bp_ext  = BP_post  - payload_intr     (BP extrinsic = posterior - a priori)
    src_ext = src_post - BP_post          (source extrinsic = posterior - denoiser input)

  The denoiser is applied to BP_post, so ``BP_post`` is exactly the a priori
  input seen by the denoiser, and ``src_ext`` matches the textbook
  ``posterior - a_priori`` form.

  Update equation
  ---------------
    new_input = channel + beta * bp_ext + alpha * src_ext

  Knobs:
    alpha  — source/denoiser extrinsic weight
    beta   — BP extrinsic weight

  Special cases:
    alpha=0, beta=0     →  new_input = channel  (baseline BP, no denoiser)
    alpha=0.1, beta=0   →  new_input = channel + 0.1*src_ext
    alpha=0,   beta=1   →  new_input = BP_post (turbo BP feedback)

  EP mode (ep_mode=True) — Expectation Propagation
  ------------------------------------------------
  See docs/EP_THEORY.md (§4.1, §4.4) and docs/EP_MIGRATION_PLAN.md.  The turbo
  blend above is replaced by an explicit EP source-factor cycle.  Sites compose
  as a product (LLR addition, §3):  posterior = channel_site + code_site +
  src_site, where channel_site = payload0 is FROZEN (§4.2), code_site is realized
  by BP (msg_v2c), and src_site is explicit state (t̃_src, §2; fixes [어긋남 3]).

  Two site-update laws are selectable via ``ep_update`` (EP_THEORY.md §4.4):

    channel_site = payload0                              # frozen (§4.2)
    A_bp         = channel_site + src_site               # BP prior (§3)
    BP_post      = BP(A_bp)                               # = channel + code + src
    code_added   = BP_post - A_bp                         # code evidence this round
    cavity_src   = channel_site + β_ep * code_added       # (= BP_post - src_site if β_ep=1)
    src_full     = denoiser(cavity_src)                   # src_post - cavity (§4.1 steps 2-4)
    src_site     = (1-α_ep)*src_site + α_ep*src_full       # damped source site (§4.4)

    * ep_update="full_ep"       (default): α_ep=β_ep=1 ⇒ pure site replacement,
      cavity = BP_post - src_site exactly (§4.1).  EP-fidelity ALIGNMENT TARGET,
      but it DECODES AT BLER 1.0 — the denoiser is an over-confident, un-calibrated
      factor and full site trust corrupts the belief every round
      (docs/EP_SCHEDULING_EXPERIMENT.md §5).  Use damped_ep to actually decode.
    * ep_update="damped_ep": DAMPED EP (§4.4) for the non-linear/over-confident
      denoiser — the site update is EMA-blended by the fraction α_ep=ep_source_power
      (choose small enough that the accumulated site stays ~20-30% of the full
      belief, e.g. [2]*15 → ~0.02), code damping β_ep=ep_code_power (Approx E).
      This is the WORKING configuration (BLER ≈ 0.004, matching turbo).
      NOTE: this is *damped* EP, NOT true power/fractional EP (which tempers the
      factor by f^η in the projection; a distinct, unimplemented method — §4.4).
      "fractional_ep" is accepted as a deprecated alias of "damped_ep".

  ``alpha``/``beta`` are IGNORED in EP mode (their EP re-interpretation is
  ep_source_power / ep_code_power).  Divergence of full_ep (src_site LLR blow-up)
  is detected and logged with a recommendation to use damped_ep.

  Knobs (EP mode):
    ep_update        — "full_ep" | "damped_ep"  ("fractional_ep" = deprecated alias).
    ep_source_power  — α_ep, source DAMPING FRACTION (ρ), not a power-EP exponent;
                       aliases ep_source_damping / ep_damping.
    ep_code_power    — β_ep, code DAMPING FRACTION (Approx E); alias ep_code_damping.
    ep_divergence_llr— |src_site| LLR threshold for the divergence warning.

  Named approximations (see docs/EP_APPENDIX.md §A.6):
    * Approx C (2nd-moment calibration): the projected-posterior variance is a
      FIXED sigma_post on this branch (over-confident); src_site carries the mean
      (LLR) only.  A per-pixel precision slot exists (source_prior.py::
      projected_pixel_precision); the diagonal Tweedie σ²·diag(∂D/∂x̃) is the
      drop-in candidate, implemented on the sibling branch
      pure-EP_tweedie-2nd-diagonal-precision and shown insufficient (correlated,
      not diagonal).
    * Approx D (sigma not from cavity variance): the denoiser sigma is chosen by
      the syndrome scheduler (a proxy for global per-image cavity uncertainty),
      not the cavity std (EP_THEORY.md §4.1 step 3).

  Adaptive denoiser sigma
  -----------------------
  After each BP chunk (except the last), the syndrome ratio of the hard
  decision on the FULL ``x_hat`` (graph domain) selects σ from the
  configured ``sigma_scheduler`` (see :mod:`syndrome_sigma_schedule`):

    * FixedSigmaScheduler        : default when ``adaptive_sigma=False``.
    * HandcraftedLookupScheduler : created from ``syndrome_thresholds`` /
                                   ``sigma_levels`` if no explicit scheduler.
    * CalibratedLookupScheduler  : pass via ``sigma_scheduler=`` kwarg.

  The decoder itself contains no JSON parsing, no calibration logic, and
  no threshold parsing.  All of that lives in
  :mod:`syndrome_sigma_schedule` and :mod:`cli_common`.

  Notes / approximations:
    * In the FIRST chunk, payload_intr == channel, so bp_ext reduces to
      ``BP_post - channel`` (what the previous variant always used).
      For later chunks, ``payload_intr`` already contains the previous
      extrinsic terms, so subtracting it gives a strictly cleaner extrinsic.
    * The denoiser is non-linear, so ``src_post - BP_post`` is the standard
      practical approximation of true source extrinsic.
"""
from __future__ import annotations

import tensorflow as tf
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder, LDPCBPDecoder

from denoiser import SoftDenoiser
from syndrome_sigma_schedule import (
    DEFAULT_SIGMA_LEVELS,
    DEFAULT_SYNDROME_HARD_DECISION,
    DEFAULT_SYNDROME_THRESHOLDS,
    FixedSigmaScheduler,
    HandcraftedLookupScheduler,
    compute_syndrome_weight_and_ratio,
    csr_matrix_to_sparse_tensor,
    summarize_chunk_diagnostics,
)


class LDPC5GDecoder_soft(LDPC5GDecoder):

    def __init__(self, encoder, *, bp_schedule=None,
                 alpha=0.0, beta=0.0,
                 ep_mode=False, ep_damping=1.0,
                 true_turbo=False,
                 altproj=False, altproj_delta=0.1, altproj_rho=1.0,
                 altproj_sigma_den=0.3, altproj_warm_start=True,
                 altproj_early_stop=False, altproj_es_patience=3,
                 altproj_crc_check=None,
                 altproj_delta_schedule=None, altproj_rho_schedule=None,
                 source_input_mode="bp_post",
                 ep_update="full_ep",
                 ep_source_power=None, ep_code_power=1.0,
                 ep_source_power_schedule=None, ep_source_off_tail=0,
                 ep_divergence_llr=1.0e4,
                 bp_convergence=False, bp_conv_tol=1.0e-3,
                 bp_max_iter=200, bp_conv_verbose=False,
                 bp_track_delta=False,
                 bp_reset_each_chunk=False,
                 ep_track_payload_hist=False,
                 k_payload=None, img_h=28, img_w=28, bits_per_pixel=8,
                 denoiser_kwargs=None,
                 # Adaptive sigma — either pass a scheduler or legacy params.
                 sigma_scheduler=None,
                 adaptive_sigma=False,
                 syndrome_thresholds=None,
                 sigma_levels=None,
                 sigma_min=None,
                 sigma_max=None,
                 monotonic_sigma=False,
                 syndrome_hard_decision=DEFAULT_SYNDROME_HARD_DECISION,
                 syndrome_sign_debug=False,
                 **kwargs):
        super().__init__(encoder, **kwargs)
        self._bp_schedule_custom = (
            [int(x) for x in bp_schedule] if bp_schedule else [20]
        )
        self._alpha = float(alpha)
        self._beta = float(beta)
        # EP mode: Expectation Propagation source-factor cycle.
        # See docs/EP_THEORY.md (§4.1, §4.4) and docs/EP_MIGRATION_PLAN.md.
        self._ep_mode = bool(ep_mode)
        # ── TRUE TURBO mode (mathematically exact turbo extrinsic exchange) ──
        # Distinct from the legacy turbo-like path (ep_mode=False) and EP
        # (ep_mode=True); takes precedence over both when set.  Per chunk t
        # (docs/EP_RESEARCH_SUMMARY §A, and the spec below):
        #   a^(t)     = alpha_t · e_src^(t-1)            (source a priori)
        #   l_in^(t)  = L_ch + Pi(a^(t))                 (BP input; payload += a)
        #   P^(t)     = BP_{k_t}(l_in^(t))   from a FRESH state (NO warm-start)
        #   c^(t)     = P^(t)|payload − a^(t)            (source-free cavity)
        #   s^(t)     = D(c^(t)) ;  e_src^(t) = s^(t) − c^(t)
        # The ONLY carried state is e_src (last extrinsic) — no BP state, no site
        # accumulation, no EMA.  Excludes (guaranteed in call()): (1) denoiser
        # input is c not P (subtract a^(t)); (2) BP reset every chunk (msg_v2c
        # forced None); (3) no EMA on e_src; (4) no beta·bp_ext feedback term.
        # alpha_t = ep_source_power_schedule[t] if set, else the scalar `alpha`.
        self._true_turbo = bool(true_turbo)
        # [source_input_mode] Legacy-turbo denoiser input purity (warm-start ALWAYS
        # kept):
        #   "bp_post"               — P_t|payload (legacy; contains this chunk's
        #                             explicit injected source feedback).
        #   "minus_source_feedback" — P_t|payload − src_feedback, subtracting the
        #                             EXACT α·src_ext tensor injected into this
        #                             chunk's BP prior (removes the explicit
        #                             source self-message; the implicit residue in
        #                             warm-started msg_v2c is unremovable — an
        #                             acknowledged honest limit).  bp_ext unchanged.
        assert source_input_mode in ("bp_post", "minus_source_feedback")
        self._source_input_mode = source_input_mode
        # ── ALTPROJ mode (channel-anchored alternating projection) ──────────
        # A DISTINCT decoding path (decoder.py, `_decode_altproj`).  Interpretation
        # 1 (channel anchor + accumulated correction): the two beliefs travel on
        # SEPARATE channels — code info flows through BP's internal msg_v2c (its
        # double-count-free machinery), source info enters ONLY as an additive,
        # code-FREE correction term C_t on the intrinsic.  The posterior is NEVER
        # fed back into the intrinsic (that was interpretation 2 — replacing the
        # code-free intrinsic slot with the code-bearing posterior compounds the
        # code evidence; its ±30 saturation was the symptom).  Per spec ([1]):
        #   C_0 = 0                                         (correction; payload LLR)
        #   code:   P_t = BP_{k_t}(payload_intr = L_ch + C_t, parity = L_ch)
        #                 → intrinsic is ALWAYS the channel LLR plus the code-free
        #                   correction; L_ch is the anchor.
        #   source: C_{t+1} = ρ·C_t + δ·D(P_t|payload ; σ_den)
        #                 D(·) is the denoiser call as-is (returns the extrinsic =
        #                 source_posterior − input, i.e. the source-manifold pull
        #                 vector — used DIRECTLY, no posterior reconstruction).
        #   final:  hard-decide the LAST code step's P_T → CRC (same as legacy).
        # C is a STAIRCASE ACCUMULATION of the source pull.  δ is the source step
        # size, ρ the correction memory (1.0 = pure accumulation, <1 = leaky).
        # Safety: ±25 clip on C (NOT on L_ch — the channel is never clipped), the
        # per-round mean|C_t|/mean|D_t|/syndrome monitor, and a NaN guard.  altproj
        # takes precedence over ep/true_turbo/legacy.
        self._altproj = bool(altproj)
        self._altproj_delta = float(altproj_delta)          # δ source step size
        self._altproj_rho = float(altproj_rho)              # ρ correction memory rate
        self._altproj_sigma_den = float(altproj_sigma_den)  # σ_den into the denoiser
        self._altproj_warm_start = bool(altproj_warm_start) # keep msg_v2c across steps
        # [altproj early-stop] PER-CODEWORD syndrome-gated stop.  The syndrome is
        # decoder-side observable (no genie).  Per example: track the min-syndrome
        # state; FREEZE C once the codeword hits syn=0 (solved — later rounds only
        # self-sustain, so they are wasted compute) or once its syndrome turns UP
        # past its own best (the collapse onset — later rounds actively destroy the
        # belief).  The decision returned is the frozen/best-syndrome posterior, i.e.
        # the "revert to the pre-degradation state" of spec [1].  Whole-batch frozen
        # ⇒ break the loop.  Codewords resolve at DIFFERENT rounds, so this must be
        # per-example — a batch-mean stop would be meaningless.  Default False keeps
        # the plain run-all-rounds path.
        self._altproj_early_stop = bool(altproj_early_stop)
        # Consecutive strictly-worse rounds required before the up-turn freeze fires.
        # The per-codeword syndrome wiggles early on, so patience=1 traps codewords at
        # a transient high point (measured: it wrecked the sweetspot, 0.0 → 0.56).
        # 0 disables the up-turn freeze entirely, leaving the syn=0 freeze + the
        # best-syndrome decision — the strictly-safe subset.
        self._altproj_es_patience = int(altproj_es_patience)
        # [early-stop criterion] MEASURED: a zero syndrome does NOT mean the codeword
        # is right — BP can sit on a VALID BUT WRONG codeword, and the source pull
        # then migrates it to the correct one.  (Probe, sweetspot [5]x20: at round 6
        # syn=0 for 32/32 while CRC passed only 27/32; by round 7 CRC passed 32/32.)
        # So the syndrome is NOT a sufficient stopping statistic and the "post-syn=0
        # rounds are just self-sustain" premise is false — those rounds are exactly
        # where the source prior arbitrates among valid codewords, i.e. where this
        # scheme earns its gain.  `altproj_crc_check` supplies the sufficient one: a
        # callable u_hat_logits[B, k] -> [B] bool (CRC pass), i.e. standard 5G CRC
        # early termination.  When set, a codeword is captured+frozen the round its
        # CRC passes; codewords that never pass fall back to the best-syndrome state.
        self._altproj_crc_check = altproj_crc_check
        # [altproj δ/ρ schedules] Per-round overrides of the scalars; round idx uses
        # sched[min(idx, len-1)].  Motivation: aggressive early (large δ, ρ=1 → fast
        # resolve) then conservative late (small δ, ρ<1 → hold + forget), so the
        # staircase takes big steps while the belief is poor and small ones once it
        # is good.  None → constant scalar (default).
        self._altproj_delta_schedule = (
            [float(x) for x in altproj_delta_schedule]
            if altproj_delta_schedule is not None else None)
        self._altproj_rho_schedule = (
            [float(x) for x in altproj_rho_schedule]
            if altproj_rho_schedule is not None else None)
        self._last_altproj_diag = []
        self._last_altproj_stop_round = None
        # EP site-update law (EP_THEORY.md §4.4):
        #   "full_ep"   — pure site replacement (α_ep=β_ep=1); alignment target.
        #   "damped_ep" — DAMPED EP; the site update is EMA-blended by the fraction
        #                 α_ep (and code by β_ep).  NOTE: this is *damped* EP, NOT
        #                 true power/fractional EP (which tempers the factor by f^η
        #                 in the projection — a distinct, unimplemented method, §4.4).
        #   "fractional_ep" is accepted as a DEPRECATED alias of "damped_ep"
        #                 (the old name was a misnomer).
        self._ep_update = self._normalize_ep_update(ep_update)
        # Damped-EP site fractions (α_ep, β_ep = ρ).  The parameter names contain
        # "power" for historical reasons ONLY — they are DAMPING FRACTIONS, not the
        # exponent η of power EP (§4.4).  ep_source_power defaults to the Task-3
        # `ep_damping` knob (the source damping ρ) for backward compatibility.
        self._ep_source_power = float(
            ep_source_power if ep_source_power is not None else ep_damping)
        self._ep_code_power = float(ep_code_power)
        self._ep_damping = float(ep_damping)   # alias of ep_source_power (damping ρ)
        # [Guide-α schedule] Per-chunk source damping rate α_ep.  If a list is
        # given it OVERRIDES the scalar ep_source_power per source-injection chunk
        # (chunk idx uses schedule[min(idx, len-1)]).  Lowering α late gradually
        # slows source accumulation; α=0 FREEZES the site (EMA: no new source
        # absorbed, accumulated site retained).  Default None → constant scalar.
        self._ep_source_power_schedule = (
            [float(x) for x in ep_source_power_schedule]
            if ep_source_power_schedule is not None else None)
        # [Source-off tail] For the last M source-injection chunks, REMOVE the
        # source site (src_site←0 ⇒ payload_intr = channel_site only ⇒ pure-BP
        # tail).  This is TRUE source-off (site removed), distinct from an α=0
        # tail (site frozen).  Default 0 preserves behaviour.
        self._ep_source_off_tail = int(ep_source_off_tail)
        # Divergence monitor: |src_site| LLR above this ⇒ log & flag (full_ep may
        # diverge because the denoiser is non-linear; suggest damped_ep).
        self._ep_divergence_llr = float(ep_divergence_llr)
        self._ep_config_logged = False
        # Inner-BP refinement SCHEDULE (EP_THEORY.md §4.3, §4.5, §4.6).  BP
        # iterations ARE the EP refinement steps of the individual parity-check
        # factors (Minka 2001 §4); the iteration count is a *schedule*, NOT an
        # approximation and NOT a "code projection run to convergence".
        #   bp_convergence=False (default) — run the chunk's fixed ``iters``.
        #   bp_convergence=True            — keep refining the check factors until
        #                     their messages settle (Linf < bp_conv_tol, cap
        #                     bp_max_iter); one schedule choice among many.
        self._bp_convergence = bool(bp_convergence)
        self._bp_conv_tol = float(bp_conv_tol)      # Linf msg_v2c settle threshold
        self._bp_max_iter = int(bp_max_iter)        # cap when converging
        self._bp_conv_verbose = bool(bp_conv_verbose)
        # Diagnostic: record per-chunk message settledness (final Linf Δmsg_v2c)
        # WITHOUT changing behaviour — fixed mode is stepped one iter at a time,
        # which is bit-identical to a single num_iter call (EP_THEORY.md §4.5).
        self._bp_track_delta = bool(bp_track_delta)
        # DIAGNOSTIC (Part C): if True, reset the BP warm-start state
        # (curr_msg_v2c=None) after each non-final chunk, so every chunk's BP
        # starts cold from the current prior.  Used to test whether the warm-start
        # state hides a source double-count.  Default False preserves warm-start.
        self._bp_reset_each_chunk = bool(bp_reset_each_chunk)
        # DIAGNOSTIC (Prompt B): when True, record the BP-posterior payload
        # logits at EVERY chunk (incl. final) into last_payload_hist, so the
        # per-round BER trajectory (correlated-hallucination explosion probe)
        # can be computed against ground truth outside the decoder.  Default
        # False preserves behaviour (nothing stored).
        self._ep_track_payload_hist = bool(ep_track_payload_hist)
        self._last_payload_hist = []
        # DIAGNOSTIC (mode-agnostic, recording only — never alters the decode):
        # when True, append x_hat[:, :encoder.k] (payload + CRC bits, i.e. the full
        # info block) after EVERY BP chunk, in legacy / EP / true_turbo / altproj
        # alike.  Lets a harness apply CRC early-stop EXTERNALLY and IDENTICALLY to
        # every arm — the compute saving of CRC-ES belongs to CRC-ES, not to any one
        # decoder, so comparing an arm that has it against arms that don't would be
        # an unfair harness.  Unlike _last_payload_hist (payload only, k_payload) this
        # keeps the CRC bits, which the CRC check needs.
        self._track_u_hat = False
        self._last_u_hat_hist = []
        # DIAGNOSTIC: per-chunk mean|src_ext| (legacy turbo), recorded when
        # ep_track_payload_hist is on.  Used to test whether "minus_source_feedback"
        # weakens the source output (H2 input-shift) vs. only redirects it (H1).
        self._last_src_ext_mag = []
        # DIAGNOSTIC: per-chunk src_ext VECTORS (legacy turbo), recorded only when
        # ep_track_src_ext is on (heavier; for the mode-locking cosine probe).
        self._ep_track_src_ext = False
        self._last_src_ext = []
        self._last_bp_stats = []
        self._last_src_site = None       # final source site of the latest decode
        self._k_payload_custom = int(k_payload) if k_payload is not None else None
        dn_kw = denoiser_kwargs or {}
        self._denoiser = SoftDenoiser(
            img_h=img_h, img_w=img_w, bits_per_pixel=bits_per_pixel,
            **dn_kw)

        if sigma_scheduler is not None:
            self._scheduler = sigma_scheduler
            self._adaptive_sigma = bool(getattr(sigma_scheduler, "is_adaptive", True))
        elif adaptive_sigma:
            thr = (tuple(syndrome_thresholds) if syndrome_thresholds is not None
                   else DEFAULT_SYNDROME_THRESHOLDS)
            sigs = (tuple(sigma_levels) if sigma_levels is not None
                    else DEFAULT_SIGMA_LEVELS)
            self._scheduler = HandcraftedLookupScheduler(
                thresholds=thr, sigma_levels=sigs,
                sigma_min=sigma_min, sigma_max=sigma_max,
                monotonic=bool(monotonic_sigma),
            )
            self._adaptive_sigma = True
        else:
            self._scheduler = None
            self._adaptive_sigma = False

        self._syndrome_hard_decision = syndrome_hard_decision
        self._syndrome_sign_debug = bool(syndrome_sign_debug)
        self._h_sparse = csr_matrix_to_sparse_tensor(self.pcm)
        self._last_chunk_diagnostics = []
        self._last_chunk_summary = []

    # ────── public properties ──────

    @property
    def adaptive_sigma(self):
        return self._adaptive_sigma

    @property
    def sigma_scheduler(self):
        return self._scheduler

    @sigma_scheduler.setter
    def sigma_scheduler(self, value):
        self._scheduler = value
        self._adaptive_sigma = bool(getattr(value, "is_adaptive", value is not None))

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = float(value)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = float(value)

    @property
    def ep_mode(self):
        """True → EP source-factor cycle; False → legacy turbo path."""
        return self._ep_mode

    @ep_mode.setter
    def ep_mode(self, value):
        self._ep_mode = bool(value)

    @property
    def true_turbo(self):
        """True → mathematically exact turbo (fresh BP each chunk, source-free
        cavity, single carried extrinsic, no EMA, no β).  Takes precedence over
        ep_mode."""
        return self._true_turbo

    @true_turbo.setter
    def true_turbo(self, value):
        self._true_turbo = bool(value)

    @property
    def altproj(self):
        """True → alternating-projection path (code/source manifold alternation,
        NO extrinsic; whole posterior exchanged).  Takes precedence over
        ep_mode/true_turbo/legacy."""
        return self._altproj

    @altproj.setter
    def altproj(self, value):
        self._altproj = bool(value)

    @property
    def altproj_delta(self):
        """δ source step size: C_{t+1} = ρ·C_t + δ·D(P_t).  Default 0.1.  Scales how
        strongly each source pull is added to the code-free correction."""
        return self._altproj_delta

    @altproj_delta.setter
    def altproj_delta(self, value):
        self._altproj_delta = float(value)

    @property
    def altproj_rho(self):
        """ρ correction memory rate in C_{t+1} = ρ·C_t + δ·D(P_t).  1.0 = pure
        accumulation (default); <1.0 = leaky (forgets old correction)."""
        return self._altproj_rho

    @altproj_rho.setter
    def altproj_rho(self, value):
        self._altproj_rho = float(value)

    @property
    def altproj_sigma_den(self):
        """σ_den passed to the denoiser (cavity/observation noise level) in the
        altproj source step.  Distinct from denoiser.sigma_post (read-out temp)."""
        return self._altproj_sigma_den

    @altproj_sigma_den.setter
    def altproj_sigma_den(self, value):
        self._altproj_sigma_den = float(value)

    @property
    def altproj_early_stop(self):
        """Per-codeword syndrome-gated early stop: freeze C at syn=0 (solved) or at
        the first syndrome up-turn past that codeword's best (collapse onset), and
        return the best-syndrome posterior.  Breaks the loop once all are frozen."""
        return self._altproj_early_stop

    @altproj_early_stop.setter
    def altproj_early_stop(self, value):
        self._altproj_early_stop = bool(value)

    @property
    def altproj_crc_check(self):
        """Callable u_hat_logits[B, k] -> [B] bool (CRC pass) used as the early-stop
        criterion.  None ⇒ fall back to the syndrome, which is NOT sufficient (a zero
        syndrome can be a valid-but-WRONG codeword; see __init__ notes)."""
        return self._altproj_crc_check

    @altproj_crc_check.setter
    def altproj_crc_check(self, value):
        self._altproj_crc_check = value

    @property
    def altproj_es_patience(self):
        """Consecutive strictly-worse rounds before the up-turn freeze fires.
        0 ⇒ only the syn=0 freeze (plus the best-syndrome decision)."""
        return self._altproj_es_patience

    @altproj_es_patience.setter
    def altproj_es_patience(self, value):
        self._altproj_es_patience = int(value)

    @property
    def altproj_delta_schedule(self):
        """Per-round δ list overriding the scalar (idx → sched[min(idx, len-1)]), or
        None for constant."""
        return (list(self._altproj_delta_schedule)
                if self._altproj_delta_schedule is not None else None)

    @altproj_delta_schedule.setter
    def altproj_delta_schedule(self, value):
        self._altproj_delta_schedule = (
            [float(x) for x in value] if value is not None else None)

    @property
    def altproj_rho_schedule(self):
        """Per-round ρ list overriding the scalar (idx → sched[min(idx, len-1)]), or
        None for constant."""
        return (list(self._altproj_rho_schedule)
                if self._altproj_rho_schedule is not None else None)

    @altproj_rho_schedule.setter
    def altproj_rho_schedule(self, value):
        self._altproj_rho_schedule = (
            [float(x) for x in value] if value is not None else None)

    @property
    def track_u_hat(self):
        """Record x_hat[:, :encoder.k] after every BP chunk, in EVERY mode
        (diagnostic only).  For applying CRC early-stop externally + uniformly
        across compared arms."""
        return self._track_u_hat

    @track_u_hat.setter
    def track_u_hat(self, value):
        self._track_u_hat = bool(value)

    @property
    def last_u_hat_hist(self):
        """Per-round list of x_hat[:, :encoder.k] from the latest decode (needs
        track_u_hat=True)."""
        return list(self._last_u_hat_hist)

    @property
    def last_altproj_stop_round(self):
        """Round at which the whole batch became frozen (early-stop), else None."""
        return self._last_altproj_stop_round

    @property
    def altproj_warm_start(self):
        """If True keep BP msg_v2c across alternations; if False reset (cold BP)
        each code step."""
        return self._altproj_warm_start

    @altproj_warm_start.setter
    def altproj_warm_start(self, value):
        self._altproj_warm_start = bool(value)

    @property
    def last_altproj_diag(self):
        """Per-round altproj monitor: list of dicts with round idx, mean|P_t|,
        mean|C_t| (correction used this round), mean|D_t| (source pull),
        syndrome_weight (the accumulation stability probe)."""
        return list(self._last_altproj_diag)

    @property
    def source_input_mode(self):
        """Legacy-turbo denoiser input: 'bp_post' (P_t) or
        'minus_source_feedback' (P_t − src_feedback)."""
        return self._source_input_mode

    @source_input_mode.setter
    def source_input_mode(self, value):
        assert value in ("bp_post", "minus_source_feedback")
        self._source_input_mode = value

    @staticmethod
    def _normalize_ep_update(value):
        """Canonicalize the update-law name.  'fractional_ep' is a DEPRECATED
        alias of 'damped_ep' — the old name was a misnomer (the code implements
        damped EP, not true power/fractional EP; EP_THEORY.md §4.4)."""
        v = str(value)
        return "damped_ep" if v == "fractional_ep" else v

    @property
    def ep_update(self):
        """EP site-update law: 'full_ep' (pure replacement) | 'damped_ep' (site
        EMA-blended by the fraction α_ep).  'damped_ep' is DAMPED EP, NOT true
        power/fractional EP (§4.4).  'fractional_ep' is a deprecated alias."""
        return self._ep_update

    @ep_update.setter
    def ep_update(self, value):
        self._ep_update = self._normalize_ep_update(value)

    @property
    def ep_source_power(self):
        """Damped-EP source site-damping fraction α_ep (= ρ; EP_THEORY.md §4.4);
        1.0 = full step.  NOTE: "power" in the name is HISTORICAL — this is a
        DAMPING FRACTION, not the exponent η of power EP.  Alias: ep_source_damping."""
        return self._ep_source_power

    @ep_source_power.setter
    def ep_source_power(self, value):
        self._ep_source_power = float(value)

    @property
    def ep_source_power_schedule(self):
        """Per-chunk source damping rate α_ep (list) overriding the scalar, or
        None for constant.  chunk idx → schedule[min(idx, len-1)]."""
        return (list(self._ep_source_power_schedule)
                if self._ep_source_power_schedule is not None else None)

    @ep_source_power_schedule.setter
    def ep_source_power_schedule(self, value):
        self._ep_source_power_schedule = (
            [float(x) for x in value] if value is not None else None)

    @property
    def ep_source_off_tail(self):
        """Number of trailing source-injection chunks with the source site
        REMOVED (true source-off / pure-BP tail).  0 = off."""
        return self._ep_source_off_tail

    @ep_source_off_tail.setter
    def ep_source_off_tail(self, value):
        self._ep_source_off_tail = int(value)
        self._ep_damping = float(value)   # keep the alias in sync

    @property
    def ep_code_power(self):
        """Damped-EP code site-damping fraction β_ep (EP_THEORY.md §4.4, Approx E);
        1.0 = full.  "power" is historical — a damping fraction, not power-EP η.
        Alias: ep_code_damping."""
        return self._ep_code_power

    @ep_code_power.setter
    def ep_code_power(self, value):
        self._ep_code_power = float(value)

    # Clearer aliases (the canonical params keep the historical "power" name for
    # API stability); these make the damping-fraction meaning explicit (§4.4).
    @property
    def ep_source_damping(self):
        """Alias of :attr:`ep_source_power` — source damping fraction α_ep (ρ)."""
        return self._ep_source_power

    @ep_source_damping.setter
    def ep_source_damping(self, value):
        self._ep_source_power = float(value)
        self._ep_damping = float(value)

    @property
    def ep_code_damping(self):
        """Alias of :attr:`ep_code_power` — code damping fraction β_ep."""
        return self._ep_code_power

    @ep_code_damping.setter
    def ep_code_damping(self, value):
        self._ep_code_power = float(value)

    @property
    def ep_damping(self):
        """Alias of :attr:`ep_source_power` — the source damping fraction α_ep (ρ)."""
        return self._ep_source_power

    @ep_damping.setter
    def ep_damping(self, value):
        self._ep_source_power = float(value)
        self._ep_damping = float(value)

    @property
    def bp_schedule(self):
        return list(self._bp_schedule_custom)

    @bp_schedule.setter
    def bp_schedule(self, value):
        self._bp_schedule_custom = [int(x) for x in value]

    @property
    def denoiser(self):
        return self._denoiser

    @property
    def last_chunk_diagnostics(self):
        """List[dict] for the latest decode call; one entry per denoiser chunk."""
        return list(self._last_chunk_diagnostics)

    @property
    def last_chunk_summary(self):
        """Aggregated chunk stats for the latest decode call."""
        return list(self._last_chunk_summary)

    @property
    def bp_convergence(self):
        """True → refine the check factors until their messages settle per chunk
        (a schedule choice, EP_THEORY.md §4.5); False → fixed ``iters``."""
        return self._bp_convergence

    @bp_convergence.setter
    def bp_convergence(self, value):
        self._bp_convergence = bool(value)

    @property
    def bp_track_delta(self):
        """True → record per-chunk message settledness (final Δmsg_v2c) without
        changing behaviour (diagnostic, EP_THEORY.md §4.5/§4.6)."""
        return self._bp_track_delta

    @bp_track_delta.setter
    def bp_track_delta(self, value):
        self._bp_track_delta = bool(value)

    @property
    def bp_reset_each_chunk(self):
        """True → drop the BP warm-start (msg_v2c) after each non-final chunk
        (diagnostic; Part C — probes for warm-start-hidden source double-count)."""
        return self._bp_reset_each_chunk

    @bp_reset_each_chunk.setter
    def bp_reset_each_chunk(self, value):
        self._bp_reset_each_chunk = bool(value)

    @property
    def bp_conv_tol(self):
        """Linf msg_v2c change threshold for inner-BP convergence."""
        return self._bp_conv_tol

    @bp_conv_tol.setter
    def bp_conv_tol(self, value):
        self._bp_conv_tol = float(value)

    @property
    def bp_max_iter(self):
        """Iteration cap when bp_convergence=True."""
        return self._bp_max_iter

    @bp_max_iter.setter
    def bp_max_iter(self, value):
        self._bp_max_iter = int(value)

    @property
    def bp_conv_verbose(self):
        """Print a per-chunk [EP-BP] convergence line when bp_convergence=True."""
        return self._bp_conv_verbose

    @bp_conv_verbose.setter
    def bp_conv_verbose(self, value):
        self._bp_conv_verbose = bool(value)

    @property
    def last_bp_stats(self):
        """Per-chunk inner-BP stats (iters used, final Δmsg, settled) for the
        latest decode call.  See docs/EP_THEORY.md §4.5, §4.6."""
        return list(self._last_bp_stats)

    @property
    def last_src_site(self):
        """Final source site (t̃_src, bit-LLR) of the latest decode; None before
        any decode.  Diagnostic for scheduling/convergence studies (§4.6)."""
        return self._last_src_site

    @property
    def ep_track_payload_hist(self):
        return self._ep_track_payload_hist

    @ep_track_payload_hist.setter
    def ep_track_payload_hist(self, value):
        self._ep_track_payload_hist = bool(value)

    @property
    def last_payload_hist(self):
        """List of per-chunk BP-posterior payload LLR tensors [B, k_payload]
        (one per BP chunk, incl. final) when ep_track_payload_hist=True; empty
        otherwise.  Chunk r's entry = BP posterior AFTER r source injections
        (r=0 is pure BP, no source yet) → per-round BER trajectory."""
        return list(self._last_payload_hist)

    @property
    def last_src_ext_mag(self):
        """Per-chunk mean|src_ext| (legacy turbo) of the latest decode when
        ep_track_payload_hist=True; empty otherwise."""
        return list(self._last_src_ext_mag)

    @property
    def ep_track_src_ext(self):
        return self._ep_track_src_ext

    @ep_track_src_ext.setter
    def ep_track_src_ext(self, value):
        self._ep_track_src_ext = bool(value)

    @property
    def last_src_ext(self):
        """Per-chunk src_ext [B, K] tensors (legacy turbo) when
        ep_track_src_ext=True; empty otherwise.  For the mode-locking cosine
        probe."""
        return list(self._last_src_ext)

    # ────── EP convergence / evidence diagnostics (Task 6) ──────

    def _run_bp_chunk(self, llr_bp, iters, msg_v2c, idx):
        """Run one BP chunk: ``iters`` EP refinement steps of the parity-check
        factors (EP_THEORY.md §4.3, §4.5).

        BP iterations *are* the EP messages of the individual check factors; the
        iteration count is a REFINEMENT SCHEDULE, not an approximation and not a
        "projection run to convergence".

        Modes:
          * fixed (default): run exactly ``iters`` steps.
          * bp_convergence=True: keep stepping until the messages settle
            (``max|Δ msg_v2c| < bp_conv_tol``), capped at ``bp_max_iter`` — one
            schedule choice (refine the check factors to a BP fixed point).
          * bp_track_delta=True: still run exactly ``iters`` steps, but measure
            the final message settledness (diagnostic only).

        Because ``msg_v2c`` is the complete BP state (c2v is recomputed from it
        each step), stepping one iter at a time with warm-start is bit-identical
        to a single ``num_iter`` call (verified) — bp_convergence / bp_track_delta
        change only the iteration count / logging, never the result.

        Returns (x_hat, msg_v2c, bp_iters_used, bp_final_delta, converged).
        ``converged`` is None unless bp_convergence=True.
        """
        if not self._bp_convergence and not self._bp_track_delta:
            x_hat, msg_v2c = LDPCBPDecoder.call(
                self, llr_bp, num_iter=int(iters), msg_v2c=msg_v2c)
            return x_hat, msg_v2c, int(iters), float("nan"), None

        # Step one iteration at a time (bit-identical) to run the check factors
        # to a settled point and/or to measure their per-chunk settledness.
        prev = msg_v2c
        x_hat = None
        used = 0
        delta = float("inf")
        converged = False if self._bp_convergence else None
        cap = self._bp_max_iter if self._bp_convergence else int(iters)
        for it in range(1, cap + 1):
            x_hat, msg_v2c = LDPCBPDecoder.call(
                self, llr_bp, num_iter=1, msg_v2c=msg_v2c)
            used = it
            if prev is not None:
                delta = float(tf.reduce_max(tf.abs(msg_v2c - prev)))
                if self._bp_convergence and delta < self._bp_conv_tol:
                    converged = True
                    break
            prev = msg_v2c
        return x_hat, msg_v2c, used, delta, converged

    def _ep_diagnostics(self, cavity_source, src_full, src_site, src_site_old):
        """Compute EP convergence/evidence diagnostics for one source update.

        See docs/EP_DIAGNOSTICS.md.  These are RECORDED ONLY — they never alter
        the decode (σ scheduling is untouched, requirement 3).  Returns python
        floats:

          src_site_l2        mean_b ||src_site||_2            — site magnitude
          src_site_delta_l2  mean_b ||src_site − prev||_2     — EP stationarity;
                             → 0 at a fixed point (Minka 2001 §3.3)
          src_site_max_abs   max |src_site|                   — divergence guard
          source_logZ        Σ_bit log Z_bit, mean over batch — source tilted
                             log-partition (per-round evidence contribution)
          source_free_energy − source_logZ                    — free-energy proxy

        Z_bit is the Bernoulli tilted normalizer of folding the source
        projection message ``src_full`` into the cavity:
            Z_bit = P_cav(bit=0) + P_cav(bit=1)·exp(src_full)
                  = E_{cavity}[ source factor message ]   (per bit).
        Its product over bits (here summed in log-domain) is the EP evidence
        contribution of the source factor (docs/EP_DIAGNOSTICS.md §2).
        """
        delta = src_site - src_site_old
        site_delta_l2 = tf.reduce_mean(tf.norm(delta, axis=-1))
        site_l2 = tf.reduce_mean(tf.norm(src_site, axis=-1))
        site_max_abs = tf.reduce_max(tf.abs(src_site))
        # source tilted log-partition per bit, via stable log-sum-exp.
        # q1 = P_cav(bit=1) in the denoiser's convention (sigmoid(llr)=P(1)).
        q1 = tf.clip_by_value(tf.sigmoid(cavity_source), 1e-7, 1.0 - 1e-7)
        log_p0 = tf.math.log(1.0 - q1)                     # log P_cav(0)
        log_p1 = tf.math.log(q1) + src_full                # log P_cav(1) + site msg
        logZ_bit = tf.reduce_logsumexp(
            tf.stack([log_p0, log_p1], axis=0), axis=0)     # [B, k_payload]
        source_logZ = tf.reduce_mean(tf.reduce_sum(logZ_bit, axis=-1))
        return {
            "src_site_l2": float(site_l2),
            "src_site_delta_l2": float(site_delta_l2),
            "src_site_max_abs": float(site_max_abs),
            "source_logZ": float(source_logZ),
            "source_free_energy": float(-source_logZ),
        }

    # ────── altproj: alternating projection (code ↔ source manifold) ──────

    def _decode_altproj(self, llr_ch_shape, schedule, k_payload, msg_v2c,
                        payload0, crc_and_rest, z_short, x2_par):
        """[altproj, spec [1]] Channel-anchored alternating projection.

        Two beliefs, two SEPARATE channels: the code travels through BP's internal
        messages (msg_v2c — its own double-count-free machinery), the source enters
        ONLY as an additive, code-FREE correction term C_t on the intrinsic.  The
        posterior is NEVER fed back into the intrinsic.  C is a STAIRCASE
        ACCUMULATION of the source pull.

            C_0 = 0                                     (correction; payload LLR)
            for t = 0 … T-1 (schedule = [k_t] × T):
              code step   P_t = BP_{k_t}( payload_intr = L_ch + C_t, parity = L_ch )
                          → the intrinsic is ALWAYS the channel LLR (= payload0, the
                            anchor) plus the code-free correction; parity/filler stay
                            at L_ch.
              source step C_{t+1} = ρ·C_t + δ·D(P_t|payload ; σ_den)
                            D(·) is the denoiser call as-is — it returns the extrinsic
                            (source_posterior − input), i.e. the source-manifold pull
                            vector, used DIRECTLY (no posterior reconstruction).
            final: hard-decide the LAST code step's P_T → CRC (same return path as
            legacy).

        δ is the source step size, ρ the correction memory (1.0 = pure accumulation,
        <1 = leaky).  Because the intrinsic re-anchors on L_ch every round and only a
        code-free correction is added, the code evidence is NOT compounded (the
        interpretation-2 defect).  Safety (spec [2]): (a) C_t clipped to ±25 each
        round — the CHANNEL L_ch is never clipped; (b) per-round
        mean|C_t|/mean|D_t|/syndrome-weight monitor (accumulation-stability probe);
        (c) NaN guard on C_t and D_t.
        """
        delta = tf.cast(self._altproj_delta, self.rdtype)   # δ source step size
        rho = tf.cast(self._altproj_rho, self.rdtype)       # ρ correction memory
        d_sched = self._altproj_delta_schedule              # per-round δ override
        r_sched = self._altproj_rho_schedule                # per-round ρ override
        es = self._altproj_early_stop
        es_pat = int(self._altproj_es_patience)             # 0 ⇒ syn=0 freeze only
        crc_fn = self._altproj_crc_check                    # sufficient stop statistic
        sigma_den = float(self._altproj_sigma_den)          # σ_den → denoiser obs level
        warm = self._altproj_warm_start
        c_clip = tf.cast(25.0, self.rdtype)                 # spec [2](a): ±25 on C_t

        prev_return_state = getattr(self, "_return_state", False)
        prev_hard_out = getattr(self, "_hard_out", False)
        x_hat = None
        curr_msg_v2c = msg_v2c                               # same seed as legacy chunk-0
        C = tf.zeros_like(payload0)                          # C_0 = 0 (code-free correction)
        n_steps = len(schedule)
        # [early-stop] per-codeword best-syndrome state (see __init__ notes).
        best_x = None; best_syn = None; frozen = None; bad = None; captured = None

        try:
            self._return_state = True
            self._hard_out = False
            self._last_altproj_diag = []
            self._last_bp_stats = []
            self._last_u_hat_hist = []
            self._last_altproj_stop_round = None

            for idx, iters in enumerate(schedule):
                # ── code step: payload intrinsic = L_ch + C_t (channel anchor +
                #    code-free correction), parity/filler = L_ch (always).  The code
                #    belief propagates through BP's internal msg_v2c, NOT the intrinsic
                #    — so it is never double-counted. ──
                payload_intr = payload0 + C
                x1_stage = tf.concat([payload_intr, crc_and_rest], axis=1)
                llr_bp = tf.concat([x1_stage, z_short, x2_par], axis=1)
                x_hat, curr_msg_v2c, bp_used, bp_delta, bp_conv = self._run_bp_chunk(
                    llr_bp, iters, curr_msg_v2c, idx)
                self._last_bp_stats.append({
                    "chunk_idx": idx,
                    "bp_iters_used": bp_used,
                    "bp_final_delta": bp_delta,
                    "bp_converged": bp_conv,
                    "bp_cap": (self._bp_max_iter if self._bp_convergence
                               else int(iters)),
                })
                post_payload = x_hat[:, :k_payload]      # P_t|payload (posterior)
                mean_abs_P = float(tf.reduce_mean(tf.abs(post_payload)))
                mean_abs_C = float(tf.reduce_mean(tf.abs(C)))   # mean|C_t| used this round
                # DIAGNOSTIC: full info block (payload+CRC) for external CRC early-stop.
                if self._track_u_hat:
                    self._last_u_hat_hist.append(x_hat[:, :self.encoder.k])

                # syndrome weight on the full graph-domain hard decision.
                counts, ratios, _ = compute_syndrome_weight_and_ratio(
                    x_hat, self._h_sparse,
                    decision_rule=self._syndrome_hard_decision,
                    compare_both_signs=self._syndrome_sign_debug,
                )
                syn_w = float(tf.reduce_mean(tf.cast(counts, self.rdtype)))

                # ── [early-stop] per-codeword best-syndrome tracking + freeze ──
                # `worse` / `upd` are compared against the PREVIOUS best, so a
                # codeword freezes the moment its syndrome turns up past its own
                # minimum (collapse onset) or reaches 0 (solved).  The returned
                # decision is the best-syndrome posterior = "revert to the state
                # before degradation".
                if es and crc_fn is not None:
                    # ── CRC-gated stop (the sufficient statistic; 5G-style early
                    #    termination).  Capture+freeze a codeword the round its CRC
                    #    passes; never revisit it.  Codewords that never pass keep the
                    #    best-syndrome fallback so they still return their best try.
                    counts_f = tf.cast(counts, self.rdtype)
                    # flatten to rank 1 — CRC helpers often return [B, 1], which would
                    # broadcast into a rank-3 mess downstream.
                    passed = tf.reshape(
                        tf.cast(crc_fn(x_hat[:, :self.encoder.k]), tf.bool), [-1])
                    if best_x is None:
                        best_x = x_hat
                        best_syn = counts_f
                        captured = passed
                        frozen = passed
                    else:
                        upd = tf.logical_and(counts_f < best_syn,
                                             tf.logical_not(captured))
                        best_x = tf.where(upd[:, None], x_hat, best_x)
                        best_syn = tf.where(upd, counts_f, best_syn)
                        newly = tf.logical_and(passed, tf.logical_not(captured))
                        best_x = tf.where(newly[:, None], x_hat, best_x)
                        captured = tf.logical_or(captured, passed)
                        frozen = captured
                elif es:
                    counts_f = tf.cast(counts, self.rdtype)
                    solved = tf.equal(counts, 0)
                    if best_x is None:
                        best_x = x_hat
                        best_syn = counts_f
                        bad = tf.zeros_like(counts, tf.int32)
                        frozen = solved
                    else:
                        # A per-codeword syndrome is NOT monotone — it wiggles for a
                        # few rounds before settling.  Freezing on the FIRST up-tick
                        # traps codewords at a transient high point, so degradation
                        # must be confirmed over `es_patience` CONSECUTIVE strictly-
                        # worse rounds (a plateau resets the streak).  syn=0 freezes
                        # at once: the codeword is already valid, nothing to recover.
                        worse = counts_f > best_syn
                        upd = tf.logical_and(counts_f < best_syn,
                                             tf.logical_not(frozen))
                        best_x = tf.where(upd[:, None], x_hat, best_x)
                        best_syn = tf.where(upd, counts_f, best_syn)
                        bad = tf.where(worse, bad + 1, tf.zeros_like(bad))
                        frozen = tf.logical_or(frozen, solved)
                        if es_pat > 0:
                            frozen = tf.logical_or(frozen, bad >= es_pat)

                # Final code step: no source step; P_T is the decision (legacy path).
                if idx >= n_steps - 1:
                    self._last_altproj_diag.append({
                        "round": idx, "mean_abs_P": mean_abs_P,
                        "mean_abs_C": mean_abs_C, "mean_abs_D": None,
                        "syndrome_weight": syn_w, "final": True,
                        "n_frozen": (int(tf.reduce_sum(tf.cast(frozen, tf.int32)))
                                     if es else None),
                    })
                    continue

                # [early-stop] whole batch frozen ⇒ every later round is either
                # self-sustain (solved) or destructive (degrading): stop.
                if es and bool(tf.reduce_all(frozen).numpy()):
                    self._last_altproj_stop_round = idx
                    self._last_altproj_diag.append({
                        "round": idx, "mean_abs_P": mean_abs_P,
                        "mean_abs_C": mean_abs_C, "mean_abs_D": None,
                        "syndrome_weight": syn_w, "final": True,
                        "n_frozen": int(tf.reduce_sum(tf.cast(frozen, tf.int32))),
                    })
                    break

                # ── source step: D_t = D(P_t|payload ; σ_den).  The denoiser returns
                #    the extrinsic (source_posterior − input) directly — this IS the
                #    source-manifold pull vector; no posterior reconstruction. ──
                D_t = self._denoiser(post_payload, sigma=sigma_den)
                tf.debugging.assert_all_finite(D_t, "altproj denoiser output D_t non-finite")
                mean_abs_D = float(tf.reduce_mean(tf.abs(D_t)))

                # Per-round monitor (spec [2]b) — accumulation-stability probe.
                self._last_altproj_diag.append({
                    "round": idx, "mean_abs_P": mean_abs_P,
                    "mean_abs_C": mean_abs_C, "mean_abs_D": mean_abs_D,
                    "syndrome_weight": syn_w, "final": False,
                    "n_frozen": (int(tf.reduce_sum(tf.cast(frozen, tf.int32)))
                                 if es else None),
                })

                # ── accumulate correction: C_{t+1} = ρ_t·C_t + δ_t·D_t, then clip ±25
                #    (spec [2]a; only C is clipped — the channel L_ch stays untouched).
                #    δ_t/ρ_t come from the per-round schedules when set.
                delta_t = (tf.cast(d_sched[min(idx, len(d_sched) - 1)], self.rdtype)
                           if d_sched is not None else delta)
                rho_t = (tf.cast(r_sched[min(idx, len(r_sched) - 1)], self.rdtype)
                         if r_sched is not None else rho)
                C_new = rho_t * C + delta_t * D_t
                C_new = tf.clip_by_value(C_new, -c_clip, c_clip)
                # [early-stop] frozen codewords keep their reverted C (no further pull).
                C = tf.where(frozen[:, None], C, C_new) if es else C_new
                # NaN guard (spec [2]c).
                tf.debugging.assert_all_finite(C, "altproj correction C_t non-finite")

                # warm_start=False → reset BP state so the next code step starts cold
                # from the (updated) L_ch + C; True → carry msg_v2c across steps.
                if not warm:
                    curr_msg_v2c = None

            # [early-stop] the decision is the per-codeword best-syndrome posterior
            # (= revert to the pre-degradation state), not the last round's P_T.
            if es and best_x is not None:
                x_hat = best_x

        finally:
            self._return_state = prev_return_state
            self._hard_out = prev_hard_out

        if self._return_infobits:
            u_hat_logits = x_hat[:, :self.encoder.k]
            if self._hard_out:
                u_hat = tf.cast(u_hat_logits > 0.0, tf.int32)
            else:
                u_hat = u_hat_logits
            out_shape = llr_ch_shape[:-1] + [self.encoder.k]
            out_shape[0] = -1
            u_hat = tf.reshape(u_hat, out_shape)
            if prev_return_state:
                return u_hat, curr_msg_v2c
            return u_hat

        if prev_return_state:
            return x_hat, curr_msg_v2c
        return x_hat

    # ────── core decode ──────

    def call(self, llr_ch, num_iter=None, msg_v2c=None):
        llr_ch_shape = llr_ch.get_shape().as_list()
        llr = tf.reshape(llr_ch, [-1, self.encoder.n])
        B = tf.shape(llr)[0]

        if self._encoder.num_bits_per_symbol is not None:
            llr = tf.gather(llr, self._encoder.out_int_inv, axis=-1)

        llr_5g = tf.concat(
            [tf.zeros([B, 2 * self.encoder.z], self.rdtype), llr], axis=1)

        k_filler = self.encoder.k_ldpc - self.encoder.k
        nb_punc_bits = (
            (self.encoder.n_ldpc - k_filler)
            - self.encoder.n
            - 2 * self.encoder.z)
        llr_5g = tf.concat(
            [llr_5g,
             tf.zeros([B, nb_punc_bits - self._nb_pruned_nodes], self.rdtype)],
            axis=1)

        x1_sys = llr_5g[:, :self.encoder.k]
        nb_par_bits = (
            self.encoder.n_ldpc - k_filler
            - self.encoder.k - self._nb_pruned_nodes)
        x2_par = llr_5g[:, self.encoder.k:self.encoder.k + nb_par_bits]
        z_short = (
            -tf.cast(self._llr_max, self.rdtype)
            * tf.ones([B, k_filler], self.rdtype))

        schedule = self._bp_schedule_custom
        k_payload = self._k_payload_custom
        if k_payload is None:
            k_payload = int(self.encoder.k)
        k_payload = min(k_payload, int(self.encoder.k))

        # channel_site (t̃_ch): the frozen channel LLR over payload bits
        # (EP_THEORY.md §4.2, requirement 3).  It is the always-present base of
        # the site product  posterior = channel_site + code_site + src_site.
        channel_site = x1_sys[:, :k_payload]      # = payload0 (channel intrinsic, frozen)
        payload0 = channel_site
        crc_and_rest = x1_sys[:, k_payload:]

        payload_intr = payload0                   # a-priori payload LLR fed to BP (A_bp)
        curr_msg_v2c = msg_v2c                    # warm-start preserved (t̃_code)

        # ── [altproj] alternating-projection path (fully separate; no extrinsic).
        # Dispatched here after the shared graph-domain preamble so its first code
        # step is bit-identical to legacy chunk-0 (same payload0, same msg_v2c seed).
        if self._altproj:
            return self._decode_altproj(
                llr_ch_shape, schedule, k_payload, msg_v2c,
                payload0, crc_and_rest, z_short, x2_par)

        # ── TRUE TURBO state (mathematically exact turbo) ───────────
        true_turbo = self._true_turbo
        # e_src^(t): the ONLY carried state in true turbo (last source
        # extrinsic, payload domain).  e_src^(0) = 0.  No BP state, no EMA.
        e_src = tf.zeros_like(payload0)
        tt_alpha_sched = self._ep_source_power_schedule   # per-chunk α_t (reused)

        # [source_input_mode] separated legacy-turbo feedback state.  Invariant:
        # payload_intr = payload0 + bp_feedback + src_feedback.  src_feedback is
        # the EXACT α·src_ext tensor injected this chunk (kept for exact removal).
        src_feedback = tf.zeros_like(payload0)
        bp_feedback = tf.zeros_like(payload0)

        # ── EP state (EP_THEORY.md §2, §4.4) ────────────────────────
        ep_mode = self._ep_mode and not true_turbo        # true_turbo wins
        ep_update = self._ep_update
        # Source site t̃_src: bit-LLR domain, initialised to 0 (Bernoulli site
        # "1" = no information, EP_THEORY.md §2 / EP_MIGRATION_PLAN.md §2).
        # Warm-started across chunks so the source cavity can be formed
        # (fixes [어긋남 3]).
        src_site = tf.zeros_like(payload0)
        # Damped-EP site fractions (EP_THEORY.md §4.4).  full_ep ⇒ α_ep=β_ep=1
        # (pure replacement); damped_ep ⇒ user damping fractions.
        if ep_update == "full_ep":
            a_ep = tf.cast(1.0, self.rdtype)      # source damping fraction α_ep
            b_ep = tf.cast(1.0, self.rdtype)      # code damping fraction β_ep
        else:
            a_ep = tf.cast(self._ep_source_power, self.rdtype)
            b_ep = tf.cast(self._ep_code_power, self.rdtype)
        # [Guide-α schedule / source-off tail] per-chunk α_ep + pure-BP tail.
        ep_alpha_sched = self._ep_source_power_schedule
        n_src_chunks = max(len(schedule) - 1, 1)   # source injected on chunks 0..n-1
        ep_off_tail = int(self._ep_source_off_tail)
        ep_diverged = False
        ep_div_logged = False
        ep_prev_delta = None                       # previous chunk Δsite_l2 (growth check)
        # Log the chosen update law once per instance.
        if true_turbo and not self._ep_config_logged:
            print("[TRUE-TURBO] exact turbo: fresh BP each chunk (no warm-start), "
                  "source-free cavity c=P|payload−a, single carried extrinsic "
                  "(no EMA, no β).")
            self._ep_config_logged = True
        if ep_mode and not self._ep_config_logged:
            if ep_update == "full_ep":
                print("[EP] update=full_ep (pure site replacement, α_ep=β_ep=1; "
                      "alignment target).")
            else:
                print(f"[EP] update=damped_ep (damped EP; source damping "
                      f"α_ep={self._ep_source_power}, code damping "
                      f"β_ep={self._ep_code_power}; stabilizes non-linear denoiser).")
            self._ep_config_logged = True

        prev_return_state = getattr(self, "_return_state", False)
        prev_hard_out = getattr(self, "_hard_out", False)
        x_hat = None

        a = tf.cast(self._alpha, self.rdtype)
        b = tf.cast(self._beta, self.rdtype)

        try:
            self._return_state = True
            self._hard_out = False
            self._last_chunk_diagnostics = []
            self._last_chunk_summary = []
            self._last_bp_stats = []
            self._last_payload_hist = []
            self._last_u_hat_hist = []
            self._last_src_ext_mag = []
            self._last_src_ext = []

            for idx, iters in enumerate(schedule):
                # [TRUE TURBO] a priori a^(t) = alpha_t · e_src^(t-1); BP input
                # l_in^(t) = L_ch + Pi(a^(t)) (payload += a, rest = channel).
                if true_turbo:
                    alpha_t = (tt_alpha_sched[min(idx, len(tt_alpha_sched) - 1)]
                               if tt_alpha_sched is not None else self._alpha)
                    a_prior = tf.cast(alpha_t, self.rdtype) * e_src
                    payload_intr = payload0 + a_prior
                    curr_msg_v2c = None    # exclusion (2): FRESH BP every chunk
                # ── BP chunk: ``iters`` EP refinement steps of the parity-check
                #    factors (EP_THEORY.md §4.3, §4.5, §4.6).  Iteration count is
                #    a refinement SCHEDULE, not an approximation.
                x1_stage = tf.concat([payload_intr, crc_and_rest], axis=1)
                llr_bp = tf.concat([x1_stage, z_short, x2_par], axis=1)

                # Fixed ``iters`` (default), run-to-settle, or fixed+measure.
                x_hat, curr_msg_v2c, bp_used, bp_delta, bp_conv = self._run_bp_chunk(
                    llr_bp, iters, curr_msg_v2c, idx)
                self._last_bp_stats.append({
                    "chunk_idx": idx,
                    "bp_iters_used": bp_used,
                    "bp_final_delta": bp_delta,
                    "bp_converged": bp_conv,
                    "bp_cap": (self._bp_max_iter if self._bp_convergence
                               else int(iters)),
                })
                if self._bp_convergence and self._bp_conv_verbose:
                    if bp_conv:
                        print(f"[EP-BP] chunk {idx}: converged in {bp_used} iters "
                              f"(max|Δmsg_v2c|={bp_delta:.2e} < {self._bp_conv_tol:g})")
                    else:
                        print(f"[EP-BP] chunk {idx}: NOT converged in {bp_used} "
                              f"iters (max|Δmsg_v2c|={bp_delta:.2e} ≥ "
                              f"{self._bp_conv_tol:g}); capped at bp_max_iter")

                # DIAGNOSTIC (Prompt B): record BP-posterior payload at EVERY
                # chunk (incl. final) for the per-round BER trajectory probe.
                if self._ep_track_payload_hist:
                    self._last_payload_hist.append(x_hat[:, :k_payload])
                # DIAGNOSTIC: full info block (payload+CRC) for external CRC early-stop.
                if self._track_u_hat:
                    self._last_u_hat_hist.append(x_hat[:, :self.encoder.k])

                # Final chunk: no denoiser feedback, channel stays frozen.
                if idx >= len(schedule) - 1:
                    continue

                # ── Syndrome diagnostics on full graph-domain x_hat ──
                counts, ratios, sign_debug = compute_syndrome_weight_and_ratio(
                    x_hat,
                    self._h_sparse,
                    decision_rule=self._syndrome_hard_decision,
                    compare_both_signs=self._syndrome_sign_debug,
                )

                # BP_post over the payload sub-block.  In EP mode this equals
                # t̃_ch + t̃_code + t̃_src (EP_THEORY.md §2), because BP was fed
                # the prior payload0 + src_site.
                post_payload = x_hat[:, :k_payload]

                # ── Choose denoiser sigma via scheduler ──────────────
                # NOTE (Approx D, EP_THEORY.md §4.1 step 3): in pure EP this
                # sigma should be the CAVITY standard deviation.  We keep the
                # syndrome-scheduler sigma for now; Task 5 replaces it.
                if self._adaptive_sigma and self._scheduler is not None:
                    batch_sigma = self._scheduler.select_sigma(idx, ratios)
                    use_sched_sigma = True
                else:
                    # Fixed sigma path: denoiser uses its own scalar self.sigma;
                    # we still record a per-batch tensor for diagnostics.
                    batch_sigma = tf.fill(
                        [tf.shape(post_payload)[0]],
                        tf.cast(self._denoiser.sigma, self.rdtype),
                    )
                    use_sched_sigma = False

                ep_diag = None      # filled in EP mode; stays None in legacy mode
                if true_turbo:
                    # ═══ TRUE TURBO source-factor extrinsic exchange ═══
                    # (1) source-free CAVITY: subtract EXACTLY the a priori
                    #     injected this chunk so the source decoder never sees
                    #     its own information — c^(t) = P^(t)|payload − a^(t).
                    c_cav = post_payload - a_prior
                    # verification (spec §5.3): c^(t) + a^(t) = P^(t)|payload
                    # holds by construction (the cavity removes exactly the a
                    # priori injected this chunk).
                    tf.debugging.assert_near(
                        c_cav + a_prior, post_payload, rtol=1e-5, atol=1e-5,
                        message="true-turbo cavity check c+a != P|payload failed")
                    # (2) source posterior s = D(c); the denoiser returns
                    #     (posterior − input), so this IS e_src = s − c directly.
                    if use_sched_sigma:
                        e_src = self._denoiser(c_cav, sigma=batch_sigma)
                    else:
                        e_src = self._denoiser(c_cav)
                    # exclusion (3): no EMA — e_src is REPLACED, only the last
                    # extrinsic is carried.  exclusion (4): no β·bp_ext term.
                    tf.debugging.assert_all_finite(
                        e_src, "true-turbo e_src became non-finite")
                    # payload_intr for the next chunk is rebuilt at the loop top
                    # from e_src (= L_ch + Pi(alpha·e_src)); nothing to do here.
                elif ep_mode:
                    # ═══ EP source-factor cycle (EP_THEORY.md §4.1, §4.4) ═══
                    # Sites compose as a product: posterior = channel_site +
                    # code_site + src_site (LLR addition, §3).  channel_site is
                    # frozen (§4.2); code_site is what BP added this round:
                    code_added = post_payload - payload_intr        # = BP_post − A_bp
                    # (1) CAVITY = posterior − source site.  In full_ep (β_ep=1)
                    #     this is exactly BP_post − src_site (removes the source
                    #     self-message → fixes [어긋남 1]).  In damped_ep,
                    #     β_ep tempers the code evidence entering the cavity
                    #     (Approx E, §4.4).  Feeding the CAVITY (not BP_post) is
                    #     the EP division = LLR subtraction (§3).
                    cavity_source = channel_site + b_ep * code_added
                    # (2)+(3) tilted × projection amortized by the denoiser
                    #     (Tweedie, §4.1 step 3).  It returns src_post − input;
                    #     with input = cavity this is the FULL EP site update
                    # (4) src_full = src_post − cavity (§4.1 step 4).
                    if use_sched_sigma:
                        src_full = self._denoiser(cavity_source, sigma=batch_sigma)
                    else:
                        src_full = self._denoiser(cavity_source)
                    # Site update (EP_THEORY.md §4.4):
                    #   full_ep    (α_ep=1): src_site = src_full  → pure replacement,
                    #                        fixes [어긋남 2] (no damped addition).
                    #   damped_ep  (α_ep<1): EMA-damped site; same fixed points as
                    #                        full EP (damped EP, NOT power EP).
                    src_site_old = src_site
                    one = tf.cast(1.0, self.rdtype)
                    # [Guide-α schedule] per-chunk source damping rate: override
                    # a_ep for THIS source-injection chunk (idx) if a schedule is
                    # set.  α=0 ⇒ freeze site (no new source absorbed this round).
                    if ep_alpha_sched is not None:
                        a_ep = tf.cast(
                            ep_alpha_sched[min(idx, len(ep_alpha_sched) - 1)],
                            self.rdtype)
                    src_site = (one - a_ep) * src_site_old + a_ep * src_full
                    # [Source-off tail] TRUE source-off for the last M chunks:
                    # remove the site (pure-BP tail).  Distinct from α=0 (freeze).
                    if ep_off_tail > 0 and idx >= n_src_chunks - ep_off_tail:
                        src_site = tf.zeros_like(src_site)
                    # Finiteness guard: the source site must stay finite.
                    tf.debugging.assert_all_finite(
                        src_site, "EP source site (src_site) became non-finite")

                    # ── EP convergence / evidence diagnostics (Task 6) ──
                    # normalizer Z_i and site change; recorded only, decode
                    # unaffected (docs/EP_DIAGNOSTICS.md).
                    ep_diag = self._ep_diagnostics(
                        cavity_source, src_full, src_site, src_site_old)
                    site_max = ep_diag["src_site_max_abs"]
                    delta_l2 = ep_diag["src_site_delta_l2"]

                    # Divergence / non-convergence detection (requirement 2, 4):
                    #   (a) site LLR blow-up          → full_ep diverging.
                    #   (b) site change GROWING       → oscillation / non-convergence.
                    site_blowup = site_max > float(self._ep_divergence_llr)
                    delta_growing = (
                        ep_prev_delta is not None
                        and delta_l2 > 1.5 * ep_prev_delta
                        and delta_l2 > 1.0)
                    ep_prev_delta = delta_l2
                    if site_blowup or delta_growing:
                        ep_diverged = True
                    # full_ep divergence → recommend switching to damped_ep.
                    recommend_fractional = (
                        ep_update == "full_ep" and (site_blowup or delta_growing))
                    if (site_blowup or delta_growing) and not ep_div_logged:
                        why = "site LLR blow-up" if site_blowup else "site-change growth"
                        msg = (f"[EP] WARNING: {why} at chunk {idx} "
                               f"(max|src_site|={site_max:.3g}, "
                               f"Δsite_l2={delta_l2:.3g}, ep_update={ep_update}).")
                        if recommend_fractional:
                            msg += (" Consider ep_update='damped_ep' with a "
                                    "smaller ep_source_power for stability.")
                        print(msg)
                        ep_div_logged = True
                    ep_diag.update({
                        "ep_diverged": bool(ep_diverged),
                        "ep_delta_growing": bool(delta_growing),
                        "ep_recommend_fractional": bool(recommend_fractional),
                    })

                    # Reconstruction: next BP prior = channel_site ⊗ src_site
                    # (posterior = cavity + new site, §3, requirement 3).  The
                    # code factor re-enters through BP's warm-started msg_v2c, so
                    # it is NOT re-added here (avoids the code double-count, §1.4).
                    # alpha/beta are UNUSED in EP mode (see ep_source/code_power).
                    payload_intr = channel_site + src_site
                else:
                    # ═══ Legacy turbo path (warm-start ALWAYS preserved) ═══
                    # [source_input_mode] denoiser input purity.  In
                    # "minus_source_feedback" we subtract the EXACT src_feedback
                    # tensor that was injected into THIS chunk's BP prior (not a
                    # recomputed α·src_ext — avoids index drift).  At chunk 0
                    # src_feedback=0 so both modes feed P_t|payload (bit-exact).
                    if self._source_input_mode == "minus_source_feedback":
                        source_in = post_payload - src_feedback
                        # verification: source_in + src_feedback = P_t|payload
                        tf.debugging.assert_near(
                            source_in + src_feedback, post_payload,
                            rtol=1e-5, atol=1e-5,
                            message="turbo source_in + src_feedback != P|payload")
                    else:
                        source_in = post_payload
                    if use_sched_sigma:
                        src_ext = self._denoiser(source_in, sigma=batch_sigma)
                    else:
                        src_ext = self._denoiser(source_in)
                    if self._ep_track_payload_hist:
                        self._last_src_ext_mag.append(
                            float(tf.reduce_mean(tf.abs(src_ext))))
                    if self._ep_track_src_ext:
                        self._last_src_ext.append(src_ext)
                    # bp_ext definition UNCHANGED.
                    bp_ext = post_payload - payload_intr
                    # Separated feedback state; payload_intr = payload0 +
                    # bp_feedback + src_feedback (invariant by construction).
                    bp_feedback = b * bp_ext
                    src_feedback = a * src_ext
                    payload_intr = payload0 + bp_feedback + src_feedback

                # Finiteness guard shared by both paths.
                tf.debugging.assert_all_finite(
                    payload_intr, "payload_intr became non-finite")

                chunk_diag = {
                    "chunk_idx": idx,
                    "iters": int(iters),
                    "syndrome_weight": counts,
                    "syndrome_ratio": ratios,
                    "sigma": batch_sigma,
                    "sign_debug": sign_debug,
                    # EP bookkeeping (None in legacy mode).
                    "ep_update": ep_update if ep_mode else None,
                }
                # EP convergence/evidence metrics (Task 6), only in EP mode.
                if ep_diag is not None:
                    chunk_diag.update(ep_diag)
                self._last_chunk_diagnostics.append(chunk_diag)

                # DIAGNOSTIC (Part C): optionally drop the BP warm-start so the
                # next chunk starts cold from the (updated) prior.  Isolates
                # whether msg_v2c carries a source residue across chunks.
                if self._bp_reset_each_chunk:
                    curr_msg_v2c = None

        finally:
            self._return_state = prev_return_state
            self._hard_out = prev_hard_out
            self._last_src_site = src_site        # final source site (diagnostic)
            self._last_chunk_summary = summarize_chunk_diagnostics(
                self._last_chunk_diagnostics
            )

        if self._return_infobits:
            u_hat_logits = x_hat[:, :self.encoder.k]
            if self._hard_out:
                u_hat = tf.cast(u_hat_logits > 0.0, tf.int32)
            else:
                u_hat = u_hat_logits
            out_shape = llr_ch_shape[:-1] + [self.encoder.k]
            out_shape[0] = -1
            u_hat = tf.reshape(u_hat, out_shape)
            if prev_return_state:
                return u_hat, curr_msg_v2c
            return u_hat

        if prev_return_state:
            return x_hat, curr_msg_v2c
        return x_hat
