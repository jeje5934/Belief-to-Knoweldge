"""
Micro-benchmark: which device recipe is faster for a BP-heavy soft decode on
this host?  (BP-100 / [5]x20 legacy decode is CPU-bound under the default recipe.)

  --recipe tfcpu_gpudenoiser : TF on CPU, torch denoiser on GPU  (current default)
  --recipe tfgpu_cpudenoiser : TF on GPU, torch denoiser on CPU  (candidate)

Rule respected in BOTH: only ONE framework touches CUDA (no TF+torch CUDA mix).
Times only the decode loop (build/load excluded).  Launch each recipe in its OWN
process (device visibility is set before importing sionna).
"""
import argparse, os, time
ap = argparse.ArgumentParser()
ap.add_argument("--recipe", required=True,
                choices=["tfcpu_gpudenoiser", "tfgpu_cpudenoiser"])
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--ebno", type=float, default=2.0)
a = ap.parse_args()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import tensorflow as tf
if a.recipe == "tfcpu_gpudenoiser":
    tf.config.set_visible_devices([], "GPU")            # TF -> CPU
    DEV = "cuda"
else:
    for g in tf.config.list_physical_devices("GPU"):    # TF -> GPU
        tf.config.experimental.set_memory_growth(g, True)
    DEV = "cpu"                                          # denoiser -> CPU
import torch

from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.utils import ebnodb2no
from decoder import LDPC5GDecoder_soft
from channel_models import ChannelModel, QPSK_FADING

K = 6272; SCHEDULE = [5] * 20; _COMMON = dict(
    cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
    hard_out=False, return_infobits=True, llr_max=30.0)


def bank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    return tf.constant(np.unpackbits(ds.data.numpy().astype(np.uint8).reshape(-1, 784), 1), tf.int32)


def main():
    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=2)
    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    dec = LDPC5GDecoder_soft(ldpc, k_payload=K, num_iter=sum(SCHEDULE),
                             bp_schedule=SCHEDULE, ep_mode=False,
                             source_input_mode="bp_post", adaptive_sigma=False,
                             denoiser_kwargs=dict(device=DEV), **_COMMON)
    dec.denoiser.load_weights_pt("checkpoints/denoiser.pt")
    dec.denoiser.sigma = 0.3; dec.alpha = 0.15; dec.beta = 0.15
    ch = ChannelModel(QPSK_FADING, sigma_e2=0.1, perfect_csi=False, llr_method="B")
    bk = bank(); no = ebnodb2no(a.ebno, 2, ldpc.coderate)
    # warmup (1 batch, excluded)
    idx = tf.random.uniform([a.batch], 0, tf.shape(bk)[0], dtype=tf.int32)
    u = tf.gather(bk, idx); llr = ch.transmit(ldpc(crc(tf.cast(u, ldpc.rdtype))), no)
    _ = dec(llr)
    t0 = time.time()
    for r in range(a.rounds):
        idx = tf.random.uniform([a.batch], 0, tf.shape(bk)[0], dtype=tf.int32)
        u = tf.gather(bk, idx); llr = ch.transmit(ldpc(crc(tf.cast(u, ldpc.rdtype))), no)
        hat = dec(llr); _, cv = hard_crc_decode(crcd, hat)
    dt = time.time() - t0
    cw = a.batch * a.rounds
    print(f"RECIPE={a.recipe}  {cw} cw legacy[5]x20 decode: {dt:.1f}s  "
          f"({1000*dt/cw:.1f} ms/cw)")


if __name__ == "__main__":
    main()
