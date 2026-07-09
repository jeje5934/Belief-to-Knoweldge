"""
practical_sigma [C.3] — mode-smearing diagnostic.

Takes a real early-chunk (chunk-0, ~5 BP iters) cavity at a low SNR and runs the
denoiser at several σ.  Question: does a LARGE σ produce a broad UNIMODAL image
(peaks smeared but the garment still coherent → σ-descent can later sharpen it),
or a GRAY average of competing modes (mode collapse → σ-descent cannot help)?

Saves a montage (true | cavity-mean | denoised@σ...) and per-σ PSNR vs the true
image, for a handful of examples.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no
from decoder import LDPC5GDecoder_soft

K = 6272; CKPT = "checkpoints/denoiser.pt"; SCHEDULE = [5] * 20


def main(ebno=0.5, n=6, sigmas=(0.15, 0.3, 0.6, 1.2, 2.4), chunk=0):
    crc = CRCEncoder("CRC24A")
    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=1)
    mp = Mapper("pam", num_bits_per_symbol=1); dm = Demapper("app", "pam", num_bits_per_symbol=1)
    aw = AWGN()
    dec = LDPC5GDecoder_soft(ldpc, k_payload=K, num_iter=sum(SCHEDULE), bp_schedule=SCHEDULE,
                             ep_mode=False, adaptive_sigma=False, denoiser_kwargs=dict(device=DEV),
                             cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                             hard_out=False, return_infobits=True, llr_max=30.0)
    dec.denoiser.load_weights_pt(CKPT); dec.alpha = 0.0; dec.beta = 0.0
    dec.ep_track_payload_hist = True
    dn = dec.denoiser.prior_model

    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    imgs = ds.data.numpy().astype(np.uint8)
    bank = tf.constant(np.unpackbits(imgs.reshape(-1, 784), axis=1), tf.int32)
    tf.random.set_seed(1)
    idx = tf.random.uniform([n], 0, tf.shape(bank)[0], dtype=tf.int32)
    u = tf.gather(bank, idx)
    no = ebnodb2no(ebno, 1, ldpc.coderate)
    y = aw(mp(ldpc(crc(tf.cast(u, ldpc.rdtype)))), no)
    _ = dec(dm(y, no))
    cav = torch.from_numpy(dec.last_payload_hist[chunk][:, :K].numpy()).float().to(DEV)
    true = (imgs[idx.numpy()].astype(np.float32) / 255.0)
    with torch.no_grad():
        mu = dn.llr_to_soft_field(cav)                       # cavity mean [n,1,28,28]
        outs = {float(s): dn.net(mu, torch.full((n,), float(s), device=DEV)).clamp(0, 1).cpu().numpy()
                for s in sigmas}
    mu_np = mu.cpu().numpy()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = 2 + len(sigmas)
    fig, ax = plt.subplots(n, cols, figsize=(cols * 1.4, n * 1.4))
    def psnr(a, b):
        m = np.mean((a - b) ** 2)
        return 99 if m == 0 else 10 * np.log10(1.0 / max(m, 1e-9))
    for i in range(n):
        panels = [("true", true[i]), ("cavity μ", mu_np[i, 0])] + \
                 [(f"σ={s}", outs[s][i, 0]) for s in sigmas]
        for j, (title, im) in enumerate(panels):
            ax[i, j].imshow(im, cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0: ax[i, j].set_title(title, fontsize=8)
            if j >= 2: ax[i, j].set_xlabel(f"{psnr(outs[float(title.split('=')[1])][i,0], true[i]):.1f}dB", fontsize=7)
    fig.suptitle(f"Denoiser output vs σ at chunk {chunk}, Eb/N0={ebno}dB "
                 f"(large σ: broad-unimodal or gray mode-average?)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "results/sigma_denoise_viz.png"; fig.savefig(out, dpi=120)
    # summary PSNR by σ (mean over n)
    for s in sigmas:
        ps = np.mean([psnr(outs[s][i, 0], true[i]) for i in range(n)])
        print(f"  σ={s}: mean denoised PSNR vs true = {ps:.2f} dB")
    print("saved", out)


if __name__ == "__main__":
    main()
