"""
Method A (equalise, Ñ0=N0/|ĥ|²) vs Method B (direct likelihood, fixed N0) — are
the LLRs actually different under imperfect CSI?  Compared on the SAME channel
realisation (same RNG seed → same h, n, ĥ) so any difference is the method, not
the draw.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from sionna.phy.fec.crc import CRCEncoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.utils import ebnodb2no
from channel_models import ChannelModel, QPSK_FADING

K = 6272


def _bank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    return tf.constant(np.unpackbits(ds.data.numpy().astype(np.uint8).reshape(-1, 784), 1), tf.int32)


def main(ebno=8.0, batch=64):
    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=2)
    crc = CRCEncoder("CRC24A")
    bank = _bank(); tf.random.set_seed(0)
    idx = tf.random.uniform([batch], 0, tf.shape(bank)[0], dtype=tf.int32)
    c = ldpc(crc(tf.cast(tf.gather(bank, idx), ldpc.rdtype)))
    no = ebnodb2no(ebno, 2, ldpc.coderate)
    print(f"A vs B QPSK-fading LLR @ ebno={ebno} (same realisation per row)")
    print(f"{'sigma_e2':>8} {'max|A-B|':>10} {'mean|A-B|':>10} {'corr':>7} "
          f"{'sign-agree':>10} {'std(A)':>8} {'std(B)':>8}")
    for se2 in [0.0, 0.05, 0.1, 0.2]:
        chA = ChannelModel(QPSK_FADING, sigma_e2=se2, perfect_csi=(se2 == 0.0), llr_method="A")
        chB = ChannelModel(QPSK_FADING, sigma_e2=se2, perfect_csi=(se2 == 0.0), llr_method="B")
        tf.random.set_seed(123); a = chA.transmit(c, no).numpy()
        tf.random.set_seed(123); b = chB.transmit(c, no).numpy()
        d = np.abs(a - b)
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        sgn = float(np.mean(np.sign(a) == np.sign(b)))
        print(f"{se2:>8} {d.max():>10.4f} {d.mean():>10.5f} {corr:>7.4f} "
              f"{sgn:>10.4f} {a.std():>8.2f} {b.std():>8.2f}")


if __name__ == "__main__":
    main()
