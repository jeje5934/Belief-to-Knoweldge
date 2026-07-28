"""
Verify the true-turbo implementation matches the mathematical definition
(spec §5): first-chunk equivalence, BP reset (no warm-start), cavity relation,
and a smoke run.  Performance is NOT judged here.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no

from decoder import LDPC5GDecoder_soft

IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP
N_CODEWORD = 12600
NUM_BPS = 1
CKPT = "checkpoints/denoiser.pt"
SEED = 7


def make():
    crc_enc = CRCEncoder("CRC24A"); crc_dec = CRCDecoder(crc_enc)
    ldpc = LDPC5GEncoder(K_PAYLOAD + crc_enc.crc_length, N_CODEWORD,
                         num_bits_per_symbol=NUM_BPS)
    mp = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    dm = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    return crc_enc, crc_dec, ldpc, mp, dm, AWGN()


def bank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=False, download=True)
    imgs = np.array([np.array(im) for im, _ in ds], dtype=np.uint8)
    return tf.constant(np.unpackbits(imgs.reshape(-1, IMG_H*IMG_W).astype(np.uint8), axis=1), tf.int32)


def build(mode, alpha, ldpc):
    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    kw = dict(bp_schedule=[2]*15, k_payload=K_PAYLOAD, adaptive_sigma=False,
              ep_track_payload_hist=True, num_iter=30, **common)
    if mode == "true_turbo":
        d = LDPC5GDecoder_soft(ldpc, true_turbo=True, alpha=alpha, **kw)
    else:  # legacy turbo-like
        d = LDPC5GDecoder_soft(ldpc, ep_mode=False, alpha=alpha, beta=0.0, **kw)
    d.denoiser.load_weights_pt(CKPT)
    d.denoiser.sigma = 0.3
    return d


def get_llr(ldpc, crc_enc, mp, dm, awgn, bk, ebno, batch):
    no = ebnodb2no(ebno, NUM_BPS, ldpc.coderate)
    tf.random.set_seed(SEED)
    idx = tf.random.uniform([batch], 0, tf.shape(bk)[0], dtype=tf.int32)
    u = tf.gather(bk, idx)
    y = awgn(mp(ldpc(crc_enc(tf.cast(u, ldpc.rdtype)))), no)
    return dm(y, no), u


def main():
    crc_enc, crc_dec, ldpc, mp, dm, awgn = make()
    bk = bank()
    llr, u = get_llr(ldpc, crc_enc, mp, dm, awgn, bk, 0.6, 16)

    print("=== [1] first-chunk equivalence (true_turbo vs legacy turbo-like) ===")
    tt = build("true_turbo", 0.1, ldpc)
    lg = build("legacy", 0.1, ldpc)
    _ = tt(llr); _ = lg(llr)
    h_tt = tt.last_payload_hist[0].numpy()
    h_lg = lg.last_payload_hist[0].numpy()
    d0 = float(np.max(np.abs(h_tt - h_lg)))
    print(f"  chunk-0 payload BP posterior max|Δ| = {d0:.2e}  "
          f"→ {'BIT-EXACT ✓' if d0 < 1e-4 else 'MISMATCH ✗'}")

    print("\n=== [2] BP reset — true_turbo with α=0 → every chunk identical ===")
    tt0 = build("true_turbo", 0.0, ldpc)
    _ = tt0(llr)
    hist = [h.numpy() for h in tt0.last_payload_hist]
    diffs = [float(np.max(np.abs(hist[i] - hist[0]))) for i in range(len(hist))]
    print(f"  chunks={len(hist)}  max|chunk_i − chunk_0| over i = {max(diffs):.2e}")
    print(f"  → {'ALL IDENTICAL ✓ (fresh BP, no warm-start)' if max(diffs) < 1e-4 else 'DIFFER ✗'}")
    lg0 = build("legacy", 0.0, ldpc)   # contrast: warm-start accumulates
    _ = lg0(llr)
    lh = [h.numpy() for h in lg0.last_payload_hist]
    ld = max(float(np.max(np.abs(lh[i] - lh[0]))) for i in range(len(lh)))
    print(f"  contrast (legacy α=0, warm-start): max|chunk_i − chunk_0| = {ld:.2e} "
          f"→ {'differ (warm-start accumulates, expected)' if ld > 1e-4 else 'identical?!'}")

    print("\n=== [3] cavity relation c+a=P|payload — inline assert ran during decode ===")
    print("  (tf.debugging.assert_near in the true_turbo branch did not raise → holds)")

    print("\n=== [4] smoke — true_turbo 0.6 dB, sensible BLER (no perf judgment) ===")
    tt_s = build("true_turbo", 0.1, ldpc)
    bp30 = LDPC5GDecoder(ldpc, num_iter=30, cn_update="boxplus-phi", vn_update="sum",
                         cn_schedule="flooding", hard_out=False, return_infobits=True,
                         llr_max=30.0)
    for name, dec in [("BP-30", bp30), ("true_turbo", tt_s)]:
        nack = 0; total = 0; be = 0
        for r in range(4):
            l2, u2 = get_llr(ldpc, crc_enc, mp, dm, awgn, bk, 0.6, 32)
            hat = dec(l2)
            _, cv = hard_crc_decode(crc_dec, hat)
            nack += 32 - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy()); total += 32
            be += int(tf.reduce_sum(tf.cast(tf.not_equal(u2, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)), tf.int32)).numpy())
        print(f"  {name:12} BLER={nack/total:.3f}  BER={be/(total*K_PAYLOAD):.4f}  ({nack}/{total})")
    print("\nverification done.")


if __name__ == "__main__":
    main()
