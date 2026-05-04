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

        payload0 = x1_sys[:, :k_payload]          # channel intrinsic (frozen)
        crc_and_rest = x1_sys[:, k_payload:]

        payload_intr = payload0
        curr_msg_v2c = msg_v2c                    # warm-start preserved

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

            for idx, iters in enumerate(schedule):
                # ── BP chunk ─────────────────────────────────────────
                x1_stage = tf.concat([payload_intr, crc_and_rest], axis=1)
                llr_bp = tf.concat([x1_stage, z_short, x2_par], axis=1)

                x_hat, curr_msg_v2c = LDPCBPDecoder.call(
                    self, llr_bp,
                    num_iter=int(iters),
                    msg_v2c=curr_msg_v2c)

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

                # ── Choose sigma via scheduler ───────────────────────
                post_payload = x_hat[:, :k_payload]
                if self._adaptive_sigma and self._scheduler is not None:
                    batch_sigma = self._scheduler.select_sigma(idx, ratios)
                    src_ext = self._denoiser(post_payload, sigma=batch_sigma)
                else:
                    # Fixed sigma path: denoiser uses its own scalar self.sigma;
                    # we still record a per-batch tensor for diagnostics.
                    batch_sigma = tf.fill(
                        [tf.shape(post_payload)[0]],
                        tf.cast(self._denoiser.sigma, self.rdtype),
                    )
                    src_ext = self._denoiser(post_payload)

                # ── Turbo-style extrinsics ──────────────────────────
                bp_ext = post_payload - payload_intr   # subtract a priori IN

                self._last_chunk_diagnostics.append({
                    "chunk_idx": idx,
                    "iters": int(iters),
                    "syndrome_weight": counts,
                    "syndrome_ratio": ratios,
                    "sigma": batch_sigma,
                    "sign_debug": sign_debug,
                })

                # ── Update payload intrinsic (channel frozen) ───────
                payload_intr = payload0 + b * bp_ext + a * src_ext

        finally:
            self._return_state = prev_return_state
            self._hard_out = prev_hard_out
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
