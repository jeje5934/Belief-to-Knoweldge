"""
[2a] check 4 — smoke: legacy decoder + multistep source step, K=10, 64cw, AWGN
0.6 dB.  Crash-only check; BLER recorded, NOT judged (tuning is prompt B).
Reference: same legacy with single-shot source, same seeds.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import time
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no
from decoder import LDPC5GDecoder_soft

H = W = 28; BPP = 8; K = H * W * BPP; N = 12600; NBPS = 1
CKPT = "checkpoints/denoiser.pt"; DEV = "cuda"
SCHED = [5] * 20
COMMON = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
              hard_out=False, return_infobits=True, llr_max=30.0)


def bank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    flat = ds.data.numpy().astype(np.uint8).reshape(-1, H * W)
    return tf.constant(np.unpackbits(flat, axis=1), tf.int32)


def main():
    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    ldpc = LDPC5GEncoder(K + crc.crc_length, N, num_bits_per_symbol=NBPS)
    mapper = Mapper("pam", num_bits_per_symbol=NBPS)
    demapper = Demapper("app", "pam", num_bits_per_symbol=NBPS)
    awgn = AWGN(); bk = bank()

    dec = LDPC5GDecoder_soft(ldpc, k_payload=K, num_iter=sum(SCHED), bp_schedule=SCHED,
                             ep_mode=False, source_input_mode="bp_post",
                             adaptive_sigma=False, denoiser_kwargs=dict(device=DEV),
                             **COMMON)
    dec.denoiser.load_weights_pt(CKPT); dec.denoiser.sigma = 0.3
    dec.alpha = 0.15; dec.beta = 0.15                    # default legacy representative

    B = 64
    tf.random.set_seed(7)
    idx = tf.random.uniform([B], 0, tf.shape(bk)[0], dtype=tf.int32)
    u = tf.gather(bk, idx)
    no = ebnodb2no(0.6, NBPS, ldpc.coderate)
    y = awgn(mapper(ldpc(crc(tf.cast(u, ldpc.rdtype)))), no)
    llr = demapper(y, no)

    for mode in ("single_shot", "multistep"):
        dec.denoiser.sampler = mode
        if mode == "multistep":
            dec.denoiser.configure_multistep(ms_steps=10, ms_sigma_max=1.5,
                                             ms_sigma_min=0.05, ms_guidance=0.6,
                                             ms_use_confidence=True, ms_stochastic=False)
        t0 = time.time()
        hat = dec(llr)
        _, cv = hard_crc_decode(crcd, hat)
        nack = int(B - tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        finite = bool(tf.reduce_all(tf.math.is_finite(hat)).numpy())
        print(f"[{mode:11s}] completed, no crash. finite={finite}  "
              f"BLER(nack)={nack}/{B}  {time.time()-t0:.0f}s "
              f"(a=b=0.15, [5]x20, warm-on{', K=10' if mode=='multistep' else ''})")


if __name__ == "__main__":
    main()
