"""
[Prompt-1] Direction-consistency (mode-locking) diagnostic.

Interpretation: bp_post input (which contains the source's own prior claim)
anchors the denoiser to the previously-chosen mode → chunk-to-chunk DIRECTION
consistency.  minus (purified) input makes it re-judge from neutral each chunk →
the mode jumps in a multimodal posterior.  Magnitude is equal (ratio 0.98,
confirmed).  Direct prediction:

  cos(src_ext^(t), src_ext^(t+1)) is systematically HIGHER for bp_post than minus.

Also logs cos(src_ext^(1), src_ext^(t)) — how far the direction drifts from the
first chunk.  Same seed / same batches for both modes.  Verdict is data-driven:
if bp_post cos is not consistently higher, the mode-locking mechanism is
UNSUPPORTED (report honestly; the D<C BLER result still stands).

β=0, α=0.1, [2]×15, σ_in=0.3, sp=3.0, 0.6 dB.  GPU recipe.  No commit.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import torch
DENOISER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no
from decoder import LDPC5GDecoder_soft
import json

IMG_H, IMG_W, BPP = 28, 28, 8
K = IMG_H*IMG_W*BPP
NUM_BPS = 1
CKPT = "checkpoints/denoiser.pt"
SEED = 42


def build(ldpc, mode):
    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    d = LDPC5GDecoder_soft(ldpc, bp_schedule=[2]*15, k_payload=K, ep_mode=False,
                           source_input_mode=mode, adaptive_sigma=False, num_iter=30,
                           denoiser_kwargs=dict(device=DENOISER_DEVICE), **common)
    d.denoiser.load_weights_pt(CKPT); d.denoiser.sigma = 0.3
    d.ep_track_src_ext = True
    d.alpha = 0.1; d.beta = 0.0
    return d


def cos_consecutive(vecs):
    """vecs: list of [B,K] arrays. returns array[len-1] mean over B of cos(t,t+1)."""
    out = []
    for t in range(len(vecs)-1):
        a, b = vecs[t], vecs[t+1]
        num = (a*b).sum(1)
        den = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1) + 1e-9
        out.append(float(np.mean(num/den)))
    return out


def cos_to_first(vecs):
    a0 = vecs[0]
    out = []
    for t in range(len(vecs)):
        num = (a0*vecs[t]).sum(1)
        den = np.linalg.norm(a0, axis=1)*np.linalg.norm(vecs[t], axis=1)+1e-9
        out.append(float(np.mean(num/den)))
    return out


def main():
    os.makedirs("results", exist_ok=True)
    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    ldpc = LDPC5GEncoder(K+crc.crc_length, 12600, num_bits_per_symbol=NUM_BPS)
    mp = Mapper("pam", num_bits_per_symbol=NUM_BPS); dm = Demapper("app", "pam", num_bits_per_symbol=NUM_BPS); aw = AWGN()
    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    imgs = np.array([np.array(im) for im, _ in ds], dtype=np.uint8)
    bank = tf.constant(np.unpackbits(imgs.reshape(-1, 784).astype(np.uint8), axis=1), tf.int32)
    bp = build(ldpc, "bp_post"); mn = build(ldpc, "minus_source_feedback")
    E = 0.6; no = ebnodb2no(E, NUM_BPS, ldpc.coderate); B = 64; NB = 6
    cc_bp = []; cc_mn = []; cf_bp = []; cf_mn = []
    print(f"direction diag @ {E}dB, {NB} batches × {B}, denoiser={DENOISER_DEVICE}\n", flush=True)
    for r in range(NB):
        tf.random.set_seed(SEED + r + int(E*1000))
        idx = tf.random.uniform([B], 0, tf.shape(bank)[0], dtype=tf.int32)
        u = tf.gather(bank, idx)
        y = aw(mp(ldpc(crc(tf.cast(u, ldpc.rdtype)))), no); llr = dm(y, no)
        _ = bp(llr); _ = mn(llr)
        vb = [v.numpy() for v in bp.last_src_ext]; vm = [v.numpy() for v in mn.last_src_ext]
        cc_bp.append(cos_consecutive(vb)); cc_mn.append(cos_consecutive(vm))
        cf_bp.append(cos_to_first(vb)); cf_mn.append(cos_to_first(vm))
    cc_bp = np.mean(cc_bp, 0); cc_mn = np.mean(cc_mn, 0)
    cf_bp = np.mean(cf_bp, 0); cf_mn = np.mean(cf_mn, 0)
    print("consecutive cos(src_ext^t, src_ext^{t+1}):")
    print("  t      : " + " ".join(f"{i:5d}" for i in range(len(cc_bp))))
    print("  bp_post: " + " ".join(f"{v:5.2f}" for v in cc_bp))
    print("  minus  : " + " ".join(f"{v:5.2f}" for v in cc_mn))
    print(f"  MEAN consecutive cos: bp_post={cc_bp.mean():.3f}  minus={cc_mn.mean():.3f}  "
          f"Δ(bp−minus)={cc_bp.mean()-cc_mn.mean():+.3f}")
    print("\ncos(src_ext^1, src_ext^t) drift:")
    print("  bp_post: " + " ".join(f"{v:5.2f}" for v in cf_bp))
    print("  minus  : " + " ".join(f"{v:5.2f}" for v in cf_mn))
    higher = cc_bp.mean() > cc_mn.mean() + 0.02
    print(f"\nVERDICT: bp_post consecutive cos {'systematically HIGHER → mode-locking SUPPORTED' if higher else 'NOT clearly higher → mode-locking UNSUPPORTED (report open)'}")
    json.dump({"cc_bp": cc_bp.tolist(), "cc_mn": cc_mn.tolist(),
               "cf_bp": cf_bp.tolist(), "cf_mn": cf_mn.tolist()},
              open("results/direction_diag.json", "w"), indent=2)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(range(len(cc_bp)), cc_bp, "s-", color="#48c", label=f"bp_post (mean {cc_bp.mean():.2f})")
    a1.plot(range(len(cc_mn)), cc_mn, "o-", color="#c33", label=f"minus (mean {cc_mn.mean():.2f})")
    a1.set_xlabel("chunk t"); a1.set_ylabel("cos(src_ext^t, src_ext^{t+1})")
    a1.set_title("Consecutive direction consistency (mode-locking test)"); a1.grid(alpha=.3); a1.legend()
    a2.plot(range(len(cf_bp)), cf_bp, "s-", color="#48c", label="bp_post")
    a2.plot(range(len(cf_mn)), cf_mn, "o-", color="#c33", label="minus")
    a2.set_xlabel("chunk t"); a2.set_ylabel("cos(src_ext^1, src_ext^t)")
    a2.set_title("Drift from first-chunk direction"); a2.grid(alpha=.3); a2.legend()
    fig.tight_layout(); fig.savefig("results/direction_diag.png", dpi=150, bbox_inches="tight")
    print("saved: results/direction_diag.png/.json")


if __name__ == "__main__":
    main()
