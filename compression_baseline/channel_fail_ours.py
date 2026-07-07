"""
Failure-quality (ours): at a low Es/N0, collect CRC-FAILED codewords from the
denoiser-in-the-loop decoder and reconstruct the payload hard-decision as an
image.  Because our payload IS the raw 8-bit image, a CRC failure still yields a
mostly-correct picture (graceful degradation) -- we quantify it with PSNR.

Read-only import of pure-EP_practical decoder from a worktree (--ep_ro).
TF on CPU + torch denoiser on GPU (docs/COMPUTE_LESSONS §5).
"""
import argparse, math, os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.channel.awgn import AWGN

K = 6272; NUM_BPS = 1; SEED = 123
CKPT = "/home/LJH/onlyextrinsic_ada_sigma/checkpoints/denoiser.pt"
OUT = "/home/LJH/onlyextrinsic_ada_sigma/compression_baseline/results/fail_ours.npz"


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2, axis=(1, 2))
    return np.where(mse == 0, 99.0, 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_ro", default=os.environ.get("EP_RO", ""))
    ap.add_argument("--esno", type=float, default=-4.0)
    ap.add_argument("--n_fail", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_rounds", type=int, default=60)
    ap.add_argument("--mode", choices=["legacy", "ep"], default="legacy")
    args = ap.parse_args()
    sys.path.insert(0, args.ep_ro)
    from decoder import LDPC5GDecoder_soft

    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    ldpc = LDPC5GEncoder(K + crc.crc_length, 12600, num_bits_per_symbol=NUM_BPS)
    mp = Mapper("pam", num_bits_per_symbol=NUM_BPS)
    dm = Demapper("app", "pam", num_bits_per_symbol=NUM_BPS); aw = AWGN()
    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    imgs = np.array([np.array(im) for im, _ in ds], dtype=np.uint8)
    bank = tf.constant(np.unpackbits(imgs.reshape(-1, 784).astype(np.uint8), axis=1), tf.int32)

    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    d = LDPC5GDecoder_soft(ldpc, k_payload=K, num_iter=100,
                           denoiser_kwargs=dict(device=DEV),
                           bp_schedule=[5] * 20,
                           ep_mode=(args.mode == "ep"),
                           **({"ep_update": "damped_ep", "ep_source_power": 0.01,
                               "ep_code_power": 1.0, "adaptive_sigma": True}
                              if args.mode == "ep"
                              else {"source_input_mode": "bp_post",
                                    "adaptive_sigma": False}),
                           **common)
    d.denoiser.load_weights_pt(CKPT); d.denoiser.sigma = 0.3
    if args.mode == "legacy":
        d.alpha = 0.1; d.beta = 0.1

    no = tf.constant(10.0 ** (-args.esno / 10.0), tf.float32)
    rec, org, ps = [], [], []
    seen = tot = 0
    for r in range(args.max_rounds):
        tf.random.set_seed(SEED + r)
        idx = tf.random.uniform([args.batch], 0, tf.shape(bank)[0], dtype=tf.int32)
        u = tf.gather(bank, idx)
        y = aw(mp(ldpc(crc(tf.cast(u, ldpc.rdtype)))), no)
        hat = d(dm(y, no)); _, cv = crcd(hat)
        cv = cv.numpy().astype(bool)
        tot += args.batch
        u_np = u.numpy().astype(np.uint8)
        bits = (hat[:, :K].numpy() > 0).astype(np.uint8)
        for b in range(args.batch):
            if not cv[b]:
                ro = np.packbits(bits[b]).reshape(28, 28)
                oo = np.packbits(u_np[b]).reshape(28, 28)
                rec.append(ro); org.append(oo)
                seen += 1
        if seen >= args.n_fail:
            break
    rec = np.array(rec[:args.n_fail]); org = np.array(org[:args.n_fail])
    ps = psnr(rec, org)
    np.savez(OUT, recon=rec, orig=org, psnr=ps, esno=args.esno, mode=args.mode,
             n_seen=seen, n_total=tot)
    print(f"ours({args.mode}) Es/N0={args.esno}: collected {len(rec)} CRC-fail of "
          f"{tot} cw  PSNR mean={ps.mean():.2f} dB median={np.median(ps):.2f} "
          f"min={ps.min():.2f} max={ps.max():.2f}")


if __name__ == "__main__":
    main()
