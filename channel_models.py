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
"""
import tensorflow as tf
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.channel.awgn import AWGN

AWGN_BPSK = "awgn_bpsk"
QPSK_FADING = "qpsk_fast_fading_imperfect_csi"
_C64 = tf.complex64


def num_bps_for(kind):
    return 1 if kind == AWGN_BPSK else 2


class ChannelModel:
    def __init__(self, kind=AWGN_BPSK, sigma_e2=0.0, perfect_csi=False):
        if kind not in (AWGN_BPSK, QPSK_FADING):
            raise ValueError(f"unknown channel {kind}")
        self.kind = kind
        self.sigma_e2 = float(sigma_e2)
        self.perfect_csi = bool(perfect_csi)
        self.num_bps = num_bps_for(kind)
        if kind == AWGN_BPSK:
            self.mapper = Mapper("pam", num_bits_per_symbol=1)
            self.demapper = Demapper("app", "pam", num_bits_per_symbol=1)
            self.awgn = AWGN()
        else:
            self.mapper = Mapper("qam", num_bits_per_symbol=2)
            self.demapper = Demapper("app", "qam", num_bits_per_symbol=2)
        self.last = {}      # diagnostics: h, h_hat for the latest transmit

    def _cn(self, shape):
        """i.i.d. standard complex normal CN(0,1): var 1 (0.5 per real dim)."""
        r = tf.random.normal(shape); i = tf.random.normal(shape)
        return tf.complex(r, i) * tf.cast(tf.sqrt(0.5), _C64)

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
        y = h * x + n
        if self.perfect_csi or self.sigma_e2 == 0.0:
            h_hat = h
        else:
            h_hat = h + self._cn(shape) * tf.cast(tf.sqrt(self.sigma_e2), _C64)
        # equalise by ĥ; per-symbol effective noise variance N0/|ĥ|²
        h_abs2 = tf.maximum(tf.math.real(h_hat * tf.math.conj(h_hat)), 1e-12)
        y_tilde = y / h_hat
        no_tilde = tf.cast(no, tf.float32) / h_abs2         # [B, Nsym]
        self.last = {"h": h, "h_hat": h_hat}
        return self.demapper(y_tilde, no_tilde)
