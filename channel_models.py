"""
Channel models producing the (mismatched) LLR the decoders consume.

Two options, selectable at runtime — the DECODER comparison structure is
unchanged; only the channel/LLR front-end differs:

  awgn_bpsk                     : the original real BPSK/PAM + AWGN path.
  qpsk_fast_fading_imperfect_csi: 4-QAM/QPSK over symbol-wise fast flat Rayleigh
        fading with additive imperfect CSI.
            y_i = h_i x_i + n_i ,  h_i ~ CN(0,1) i.i.d. per symbol,  n_i ~ CN(0,N0)
            ĥ_i = h_i + e_i ,      e_i ~ CN(0, sigma_e2)   (ĥ = h if perfect_csi)
        LLR uses ĥ (NOT h) — mismatched CSI.  We equalise by ĥ and hand the
        per-symbol effective noise variance to the QAM APP demapper:
            ỹ_i = y_i / ĥ_i ,   Ñ0_i = N0 / |ĥ_i|²
        which is *identically* the direct likelihood exp(-|y_i - ĥ_i x|²/N0)
        (|ỹ-x|²·|ĥ|²/N0 = |y-ĥx|²/N0), i.e. method A ≡ method B.  Reusing Sionna's
        "app" demapper keeps the SAME LLR sign convention as the PAM path
        (LLR>0 ⇔ bit 1), so the decoder + CRC are untouched (verified separately).

AR(1) COLORED NOISE (`ar_coeff` = a ∈ [0,1)), on top of the fading+CSI path:
        n_i = a·n_{i-1} + √(1−a²)·w_i,  w_i ~ CN(0,N0),  n_0 ~ CN(0,N0)
    The √(1−a²) normalisation holds the marginal variance at N0, so a adds
    CORRELATION ONLY and leaves the SNR unchanged.  a=0 reproduces the white path
    bit-exactly (regression anchor).

    *** MISMATCHED BY DESIGN — DO NOT ADD A WHITENING FILTER. ***
    The receiver is assumed NOT to know the noise correlation a: the LLR is
    computed with the white model (diagonal N0), i.e. method B is used AS-IS and
    the noise is NOT whitened.  This is deliberate and is the point of the
    experiment: "the channel decoder does not know the noise structure" is the
    adverse condition we are constructing.  Colored noise then produces correlated
    symbol errors which the white-assumed LLR mis-calibrates → CORRELATED LLR
    errors, which BP cannot handle because it treats them as independent (its
    interleaver assumption).  The question under test is whether source-domain
    knowledge (altproj / legacy) indirectly exploits the correlation structure that
    the channel decoder cannot.  A whitening receiver would defeat this purpose and
    must not be used here.
"""
import tensorflow as tf
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.channel.awgn import AWGN

AWGN_BPSK = "awgn_bpsk"
QPSK_FADING = "qpsk_fast_fading_imperfect_csi"
_C64 = tf.complex64

# QPSK/4-QAM Gray labelling (verified): bits [b0,b1] -> point index 2*b0+b1.
#   points: 0:+.707+.707j 1:+.707-.707j 2:-.707+.707j 3:-.707-.707j
#   bit0(MSB)=1 <=> Re<0 (idx {2,3});  bit1(LSB)=1 <=> Im<0 (idx {1,3}).
_QAM_BIT1 = ([2, 3], [1, 3])     # per-bit indices where the bit == 1
_QAM_BIT0 = ([0, 1], [0, 2])     # per-bit indices where the bit == 0


def num_bps_for(kind):
    return 1 if kind == AWGN_BPSK else 2


class ChannelModel:
    def __init__(self, kind=AWGN_BPSK, sigma_e2=0.0, perfect_csi=False,
                 llr_method="B", ar_coeff=0.0):
        if kind not in (AWGN_BPSK, QPSK_FADING):
            raise ValueError(f"unknown channel {kind}")
        if llr_method not in ("A", "B"):
            raise ValueError("llr_method must be 'A' (equalise, N0/|ĥ|²) or "
                             "'B' (direct likelihood, fixed N0)")
        if not 0.0 <= float(ar_coeff) < 1.0:
            raise ValueError("ar_coeff must be in [0, 1)")
        self.kind = kind
        self.sigma_e2 = float(sigma_e2)
        self.perfect_csi = bool(perfect_csi)
        self.num_bps = num_bps_for(kind)
        self.llr_method = llr_method
        self.ar_coeff = float(ar_coeff)
        if kind == AWGN_BPSK:
            self.mapper = Mapper("pam", num_bits_per_symbol=1)
            self.demapper = Demapper("app", "pam", num_bits_per_symbol=1)
            self.awgn = AWGN()
        else:
            self.mapper = Mapper("qam", num_bits_per_symbol=2)
            self.demapper = Demapper("app", "qam", num_bits_per_symbol=2)
            self.qam_points = Constellation("qam", 2).points   # [4] complex
        self.last = {}      # diagnostics: h, h_hat for the latest transmit

    def _qpsk_llr_B(self, y, h_hat, no):
        """Direct QPSK likelihood with the TRUE (fixed) noise variance N0 and ĥ
        only in the signal term:  L_k = log Σ_{x:b_k=1} e^{-|y-ĥx|²/N0}
                                        − log Σ_{x:b_k=0} e^{-|y-ĥx|²/N0}.
        (Method B — unlike A it does NOT rescale the noise by |ĥ|².)"""
        pts = tf.reshape(self.qam_points, [1, 1, 4])                # [1,1,4]
        s = tf.expand_dims(h_hat, -1) * pts                        # ĥ·x  [B,Nsym,4]
        d2 = tf.abs(tf.expand_dims(y, -1) - s) ** 2                # |y-ĥx|² [B,Nsym,4]
        logp = -d2 / tf.cast(no, tf.float32)                      # [B,Nsym,4]
        llrs = []
        for one, zero in zip(_QAM_BIT1, _QAM_BIT0):
            l1 = tf.reduce_logsumexp(tf.gather(logp, one, axis=-1), -1)
            l0 = tf.reduce_logsumexp(tf.gather(logp, zero, axis=-1), -1)
            llrs.append(l1 - l0)                                   # [B,Nsym]
        llr = tf.stack(llrs, axis=-1)                             # [B,Nsym,2]
        B = tf.shape(y)[0]
        return tf.reshape(llr, [B, -1])                          # [B,N]

    def _cn(self, shape):
        """i.i.d. standard complex normal CN(0,1): var 1 (0.5 per real dim)."""
        r = tf.random.normal(shape); i = tf.random.normal(shape)
        return tf.complex(r, i) * tf.cast(tf.sqrt(0.5), _C64)

    def _ar1(self, w):
        """AR(1) time-correlated noise along the SYMBOL axis (axis 1) of w [B, Nsym].

            n_i = a·n_{i-1} + √(1−a²)·w_i ,   n_0 = w_0 ,   w_i ~ CN(0, N0)

        `w` is already CN(0, N0), so the √(1−a²) normalisation keeps the MARGINAL
        variance at N0 for every i (stationary): Var(n_i) = a²N0 + (1−a²)N0 = N0.
        SNR is therefore unchanged — only the correlation is added, and
        E[n_i n*_{i+k}] = N0·a^{|k|}.

        Written as an explicit recursion over the symbol axis, exactly as the
        formula reads.  (scipy.signal.lfilter was tried and INTERMITTENTLY SEGFAULTS
        on complex64 input with float64 coefficients — 3 of 4 a>0 sweep runs died
        with SIGSEGV.  The loop is ~13 ms per batch and the channel is generated
        once per batch, so the cost is irrelevant.)  At a=0 this is the identity,
        so the a=0 path reproduces the white channel bit-exactly.
        """
        a = self.ar_coeff
        if a == 0.0:
            return w
        import numpy as np
        s = np.complex64(np.sqrt(1.0 - a * a))
        ac = np.complex64(a)
        wn = np.ascontiguousarray(w.numpy(), dtype=np.complex64)
        n = np.empty_like(wn)
        n[:, 0] = wn[:, 0]                                   # n_0 = w_0 ~ CN(0,N0)
        for i in range(1, wn.shape[1]):
            n[:, i] = ac * n[:, i - 1] + s * wn[:, i]
        return tf.constant(n, dtype=_C64)

    def transmit(self, c, no):
        """c: [B, N] float bits (0/1).  no: scalar N0.  Returns LLR [B, N]."""
        if self.kind == AWGN_BPSK:
            x = self.mapper(c)
            y = self.awgn(x, no)
            return self.demapper(y, no)

        x = self.mapper(c)                                  # [B, Nsym] complex
        shape = tf.shape(x)
        h = self._cn(shape)                                 # CN(0,1) per symbol
        n = self._cn(shape) * tf.cast(tf.sqrt(no), _C64)    # CN(0, no)
        # AR(1) time-correlated noise (ar_coeff>0).  Fading/CSI are untouched — only
        # the noise becomes colored; the marginal variance stays N0 (SNR invariant).
        n = self._ar1(n)
        y = h * x + n
        if self.perfect_csi or self.sigma_e2 == 0.0:
            h_hat = h
        else:
            h_hat = h + self._cn(shape) * tf.cast(tf.sqrt(self.sigma_e2), _C64)
        self.last = {"h": h, "h_hat": h_hat}
        if self.llr_method == "B":
            # direct likelihood, TRUE fixed N0 (ĥ only in the signal term)
            return self._qpsk_llr_B(y, h_hat, no)
        # method A: equalise by ĥ; per-symbol effective noise variance N0/|ĥ|²
        h_abs2 = tf.maximum(tf.math.real(h_hat * tf.math.conj(h_hat)), 1e-12)
        no_tilde = tf.cast(no, tf.float32) / h_abs2         # [B, Nsym]
        return self.demapper(y / h_hat, no_tilde)
