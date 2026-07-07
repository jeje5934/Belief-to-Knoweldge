"""
Failure-quality (baseline) stage 1 (TF / Sionna, no torch): at a low Es/N0,
transmit real MAX-container codewords and capture, for CRC-FAILED codewords, the
LDPC hard-decision recovered container bits + the true container bits + the
original pixels.  Stage 2 (channel_fail_base_decode.py, torch) AC-decodes the
corrupted container to show the reconstruction is catastrophic.
"""
import argparse, os
import numpy as np
import tensorflow as tf
for _g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_g, True)
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.channel import AWGN

RESULTS = "/home/LJH/onlyextrinsic_ada_sigma/compression_baseline/results"
STREAMS = os.path.join(RESULTS, "fail_base_enc.npz")   # canonical cuDNN-off bits
OUT = os.path.join(RESULTS, "fail_base_bits.npz")
NUM_BPS = 1; CRC_LEN = 24; SEED = 123


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esno", type=float, default=-4.0)
    ap.add_argument("--n_fail", type=int, default=20)
    ap.add_argument("--batch", type=int, default=320)
    args = ap.parse_args()

    d = np.load(STREAMS)
    bits, lengths, imgs = d["bits"], d["lengths"], d["imgs"]
    K = int(d["K"])                              # fixed MAX container payload (4872)
    k_ldpc = K + CRC_LEN
    N = bits.shape[0]

    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    ldpc = LDPC5GEncoder(k_ldpc, 12600, num_bits_per_symbol=NUM_BPS)
    mp = Mapper("pam", num_bits_per_symbol=NUM_BPS)
    dm = Demapper("app", "pam", num_bits_per_symbol=NUM_BPS); aw = AWGN()
    dec = LDPC5GDecoder(ldpc, cn_update="boxplus-phi", vn_update="sum",
                        cn_schedule="flooding", hard_out=True,
                        return_infobits=True, num_iter=100, llr_max=30.0)
    no = tf.constant(10.0 ** (-args.esno / 10.0), tf.float32)
    tf.random.set_seed(SEED)

    rec_bits, true_bits, orig_px, lens = [], [], [], []
    seen = tot = 0
    for i in range(0, N, args.batch):
        u = bits[i:i + args.batch, :K].astype(np.float32)
        u_tf = tf.constant(u, ldpc.rdtype)
        y = aw(mp(ldpc(crc(u_tf))), no)
        hat = dec(dm(y, no))                      # [b, K+24] hard bits
        _, cv = crcd(hat)
        cv = cv.numpy().astype(bool)
        hb = hat[:, :K].numpy().astype(np.uint8)  # recovered container payload
        tot += u.shape[0]
        for b in range(u.shape[0]):
            if not cv[b]:
                rec_bits.append(hb[b]); true_bits.append(bits[i + b, :K])
                orig_px.append(imgs[i + b]); lens.append(int(lengths[i + b]))
                seen += 1
        if seen >= args.n_fail:
            break
    n = min(args.n_fail, len(rec_bits))
    np.savez(OUT, rec_bits=np.array(rec_bits[:n], dtype=np.uint8),
             true_bits=np.array(true_bits[:n], dtype=np.uint8),
             orig_px=np.array(orig_px[:n], dtype=np.uint8),
             lengths=np.array(lens[:n], dtype=np.int32),
             K=K, esno=args.esno, n_total=tot)
    # how corrupted are the failed containers? (bit errors in the container)
    if n:
        be = (np.array(rec_bits[:n]) != np.array(true_bits[:n])).sum(1)
        print(f"baseline MAX Es/N0={args.esno}: {n} CRC-fail of {tot} cw; "
              f"container bit-errors per fail: mean={be.mean():.1f} "
              f"min={be.min()} max={be.max()} (any>0 => AC desync)")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
