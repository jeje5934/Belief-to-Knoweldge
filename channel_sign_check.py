"""
Sign-convention + sanity verification for the channel front-ends (requirement 9).

Confirms, for BOTH channels, that the LLR the new QPSK front-end produces obeys
the SAME convention the decoder+CRC already assume (LLR>0 ⇔ bit 1), by checking:

  (1) pre-decode : at high SNR + perfect CSI, hard-decision of the raw LLR
                   (llr>0 → 1) matches the transmitted CODED bits (BER≈0).  A
                   flipped sign would give BER≈1.
  (2) end-to-end : encode a real payload → channel → BP decode → CRC24A recovers
                   the payload bit-exact and CRC passes (BLER≈0).  A sign error
                   would make CRC fail catastrophically.
  (3) physics    : QPSK imperfect CSI degrades end-to-end BLER monotonically.

Pure TF (BP-only, no denoiser) → fast.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.utils import ebnodb2no
from channel_models import ChannelModel, AWGN_BPSK, QPSK_FADING, num_bps_for

K = 6272


def _bank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    imgs = ds.data.numpy().astype(np.uint8)
    return tf.constant(np.unpackbits(imgs.reshape(-1, 784), axis=1), tf.int32)


def check(kind, ebno_hi=25.0, batch=64, sweep=()):
    nb = num_bps_for(kind)
    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=nb)
    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    dec = LDPC5GDecoder(ldpc, num_iter=100, cn_update="boxplus-phi", vn_update="sum",
                        cn_schedule="flooding", hard_out=False, return_infobits=True,
                        llr_max=30.0)
    bank = _bank()
    tf.random.set_seed(1)
    idx = tf.random.uniform([batch], 0, tf.shape(bank)[0], dtype=tf.int32)
    u = tf.gather(bank, idx)
    c = ldpc(crc(tf.cast(u, ldpc.rdtype)))

    ch = ChannelModel(kind, sigma_e2=0.0, perfect_csi=True)
    no = ebnodb2no(ebno_hi, nb, ldpc.coderate)
    llr = ch.transmit(c, no)
    # (1) pre-decode coded-bit BER under llr>0 → 1
    pre_ber = float(tf.reduce_mean(tf.cast(
        tf.not_equal(tf.cast(llr > 0, tf.int32), tf.cast(c, tf.int32)), tf.float32)))
    # (2) end-to-end
    hat = dec(llr); _, cv = hard_crc_decode(crcd, hat)
    crc_pass = float(tf.reduce_mean(tf.cast(cv, tf.float32)))
    payload_ber = float(tf.reduce_mean(tf.cast(
        tf.not_equal(u, tf.cast(hat[:, :K] > 0, tf.int32)), tf.float32)))
    print(f"[{kind}] ebno={ebno_hi} perfect-CSI:")
    print(f"   (1) pre-decode coded-bit BER (llr>0→1 vs tx c) = {pre_ber:.5f}  "
          f"-> {'OK sign' if pre_ber < 0.1 else 'SIGN FLIPPED!'}")
    print(f"   (2) end-to-end: CRC pass = {crc_pass:.3f}  payload BER = {payload_ber:.6f}  "
          f"-> {'OK' if crc_pass > 0.99 and payload_ber < 1e-4 else 'FAIL'}")
    for se2 in sweep:
        ch2 = ChannelModel(kind, sigma_e2=se2, perfect_csi=False)
        no2 = ebnodb2no(6.0, nb, ldpc.coderate)
        hat2 = dec(ch2.transmit(c, no2)); _, cv2 = hard_crc_decode(crcd, hat2)
        print(f"   (3) imperfect CSI sigma_e2={se2:<5} ebno=6: "
              f"CRC pass={float(tf.reduce_mean(tf.cast(cv2, tf.float32))):.3f}")


if __name__ == "__main__":
    check(AWGN_BPSK)
    check(QPSK_FADING, sweep=(0.0, 0.05, 0.1, 0.2))
