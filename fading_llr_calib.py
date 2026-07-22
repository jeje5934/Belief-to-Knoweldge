"""
LLR calibration diagnostic (requirement 3): does imperfect CSI make the channel
LLR OVER-confident (large |LLR| but the sign is wrong more often than |LLR|
implies)?  For each condition we compare, over the coded bits:

  implied error rate  = mean sigmoid(-|LLR|)   (what the magnitude claims)
  actual  error rate  = mean [ sign(LLR) != bit ]

over-confident  ⇔  actual > implied.  We also give a reliability table by |LLR|
bin.  Channel-LLR only (pure TF) — this is the INPUT the denoiser then amplifies.
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


def calib(llr, c):
    a = np.abs(llr); implied = 1.0 / (1.0 + np.exp(a))            # sigmoid(-|llr|)
    actual = ((llr > 0).astype(int) != c).astype(float)          # hard-decision error
    return float(implied.mean()), float(actual.mean())


def main(ebno=5.0, batch=128):
    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=2)
    crc = CRCEncoder("CRC24A")
    bank = _bank(); tf.random.set_seed(0)
    idx = tf.random.uniform([batch], 0, tf.shape(bank)[0], dtype=tf.int32)
    c = ldpc(crc(tf.cast(tf.gather(bank, idx), ldpc.rdtype)))
    cnp = c.numpy().astype(int)
    no = ebnodb2no(ebno, 2, ldpc.coderate)
    print(f"channel-LLR calibration, QPSK fading, ebno={ebno} (method B)")
    print(f"{'condition':>22} {'implied_err':>12} {'actual_err':>11} {'ratio':>7} "
          f"{'mean|LLR|':>9} {'verdict':>14}")
    conds = [("perfect CSI", 0.0, True)] + [(f"sigma_e2={s}", s, False)
                                            for s in (0.05, 0.1, 0.2)]
    for name, se2, perf in conds:
        tf.random.set_seed(123)
        ch = ChannelModel(QPSK_FADING, sigma_e2=se2, perfect_csi=perf, llr_method="B")
        llr = ch.transmit(c, no).numpy()
        imp, act = calib(llr, cnp)
        ratio = act / max(imp, 1e-9)
        verdict = "OVER-confident" if act > 1.3 * imp else ("calibrated" if act < 1.3 * imp else "-")
        print(f"{name:>22} {imp:>12.4f} {act:>11.4f} {ratio:>7.2f} "
              f"{np.abs(llr).mean():>9.2f} {verdict:>14}")
    # reliability table (imperfect vs perfect) by |LLR| bin
    print("\nreliability by |LLR| bin (actual error rate; over-confident if > implied):")
    for name, se2, perf in [("perfect", 0.0, True), ("sigma_e2=0.1", 0.1, False)]:
        tf.random.set_seed(123)
        ch = ChannelModel(QPSK_FADING, sigma_e2=se2, perfect_csi=perf, llr_method="B")
        llr = ch.transmit(c, no).numpy(); a = np.abs(llr).ravel()
        err = ((llr > 0).astype(int) != cnp).astype(float).ravel()
        edges = [0, 2, 5, 10, 20, 1e9]
        row = []
        for i in range(len(edges) - 1):
            m = (a >= edges[i]) & (a < edges[i + 1])
            imp = float(np.mean(1 / (1 + np.exp(a[m])))) if m.any() else float("nan")
            row.append(f"[{edges[i]:g},{edges[i+1]:g}):act={err[m].mean():.3f}/imp={imp:.3f}" if m.any() else "-")
        print(f"  {name:>12}: " + "  ".join(row))


if __name__ == "__main__":
    main()
