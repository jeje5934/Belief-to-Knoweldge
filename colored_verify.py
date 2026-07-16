"""
[3] Verification of the AR(1) colored-noise channel — BEFORE any performance work.

  (1) a=0 regression   : AR(1) path at a=0 == the existing white fading path
                         (bit-exact, method B).
  (2) noise statistics : autocorrelation of n_i follows a^|k|; marginal var = N0.
  (3) sign convention  : logit>0 -> bit 1 preserved (pre-decode BER at high SNR).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.crc import CRCEncoder
from sionna.phy.utils import ebnodb2no
from channel_models import ChannelModel, QPSK_FADING, _C64

K = 6272; N = 12600


def bits(B, n, seed):
    tf.random.set_seed(seed)
    return tf.cast(tf.random.uniform([B, n], 0, 2, dtype=tf.int32), tf.float32)


print("=== (1) a=0 regression: AR(1) path vs existing white path ===")
crc = CRCEncoder("CRC24A")
ldpc = LDPC5GEncoder(K + 24, N, num_bits_per_symbol=2)
u = bits(8, K, 1)
c = ldpc(crc(u))
no = ebnodb2no(6.0, 2, ldpc.coderate)

for se2, pcsi in [(0.0, True), (0.1, False)]:
    ch_w = ChannelModel(QPSK_FADING, sigma_e2=se2, perfect_csi=pcsi, llr_method="B")
    ch_a = ChannelModel(QPSK_FADING, sigma_e2=se2, perfect_csi=pcsi, llr_method="B",
                        ar_coeff=0.0)
    tf.random.set_seed(77); llr_w = ch_w.transmit(c, no).numpy()
    tf.random.set_seed(77); llr_a = ch_a.transmit(c, no).numpy()
    d = float(np.max(np.abs(llr_w - llr_a)))
    print(f"  se2={se2} perfect_csi={pcsi}: max|white - AR(a=0)| = {d:.3e}  "
          f"{'PASS (bit-exact)' if d == 0.0 else 'FAIL'}")

print("\n=== (2) noise statistics: autocorr a^|k| and marginal var N0 ===")
N0 = 2.0
Nsym = 4096
for a in (0.0, 0.3, 0.5, 0.7, 0.9):
    ch = ChannelModel(QPSK_FADING, ar_coeff=a)
    tf.random.set_seed(5)
    w = ch._cn([256, Nsym]) * tf.cast(np.sqrt(N0), _C64)
    n = ch._ar1(w).numpy()                       # [256, Nsym] complex
    var = float(np.mean(np.abs(n) ** 2))
    # complex autocorrelation r_k = E[n_i conj(n_{i+k})] / E|n|^2, averaged
    rs = []
    for k in (1, 2, 3, 5):
        num = np.mean(n[:, :-k] * np.conj(n[:, k:]))
        rs.append(np.abs(num) / var)
    exp = [a ** k for k in (1, 2, 3, 5)]
    ok_v = abs(var - N0) / N0 < 0.05
    ok_r = all(abs(r - e) < 0.04 for r, e in zip(rs, exp))
    print(f"  a={a}: var={var:.3f} (N0={N0}, {'ok' if ok_v else 'BAD'})  "
          f"|r_k| k=1,2,3,5: {' '.join(f'{r:.3f}' for r in rs)}  "
          f"expect {' '.join(f'{e:.3f}' for e in exp)}  {'PASS' if ok_r else 'FAIL'}")

print("\n=== (3) sign convention: logit>0 -> bit1 (pre-decode BER, high SNR) ===")
no_hi = ebnodb2no(20.0, 2, ldpc.coderate)
for a in (0.0, 0.5, 0.9):
    ch = ChannelModel(QPSK_FADING, sigma_e2=0.0, perfect_csi=True,
                      llr_method="B", ar_coeff=a)
    tf.random.set_seed(9)
    llr = ch.transmit(c, no_hi).numpy()
    hard = (llr > 0).astype(np.int32)
    ber = float(np.mean(hard != c.numpy().astype(np.int32)))
    print(f"  a={a}: pre-decode BER @20dB perfect CSI = {ber:.5f}  "
          f"{'PASS (sign ok)' if ber < 0.01 else 'FAIL (sign flipped?)'}")
