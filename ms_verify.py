"""
[2a] Verification of the multistep diffusion posterior sampler (implementation,
NOT performance).  torch-only (no sionna/LDPC) for checks 1-3; smoke (check 4)
is a separate script.

  1. single-shot regression sanity : K=1, guidance off, σ0=σmin=σ  ≈  single-shot D
  2. gray-vs-sharp MONTAGE (decisive): at a large σ where single-shot mode-averages
     to gray, does multistep produce a sharp single image?
  3. trajectory log                : per-step x̂0, distance-to-cavity, σ, ζ.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("MS_GPU", "0")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from denoiser import SoftDenoiser

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "checkpoints/denoiser.pt"
H = W = 28; BPP = 8; NPIX = H * W
OUT = "results/ms_verify_montage.png"


def bank(n):
    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    imgs = ds.data.numpy()[:n].astype(np.uint8).reshape(n, NPIX)   # 0..255
    return imgs


def cavity_llr_from_image(img_u8, sigma_n, rng):
    """Build a PARTIAL cavity: BPSK-transmit the image bits over AWGN(σ_n) and
    return per-bit LLR = 2y/σ_n².  Smaller SNR (larger σ_n) → more ambiguous."""
    bits = np.unpackbits(img_u8.astype(np.uint8)).astype(np.float32)       # [n*8]
    x = 2.0 * bits - 1.0
    y = x + rng.normal(0, sigma_n, size=x.shape).astype(np.float32)
    llr = 2.0 * y / (sigma_n ** 2)
    return llr.reshape(1, -1)                                              # [1, nbits]


def to_img(soft_field_tensor):
    return soft_field_tensor.detach().cpu().numpy().reshape(H, W)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dn = SoftDenoiser(img_h=H, img_w=W, bits_per_pixel=BPP, device=DEV)
    dn.load_weights_pt(CKPT)
    prior = dn.prior_model
    print(f"[setup] device={DEV}, weights={CKPT}")

    N = 4
    imgs = bank(N)
    clean = [im.astype(np.float32).reshape(H, W) / 255.0 for im in imgs]
    SIGMA_N = 1.1                          # cavity ambiguity level
    cavities = [torch.from_numpy(cavity_llr_from_image(imgs[i], SIGMA_N, rng)).float().to(DEV)
                for i in range(N)]

    # ---- check 1: single-shot regression sanity -------------------------------
    print("\n=== [1] single-shot regression sanity (K=1, guidance off, σ0=σmin=0.3) ===")
    dn.sampler = "single_shot"
    ss = prior.net(prior.llr_to_soft_field(cavities[0]),
                   torch.tensor([0.3], device=DEV)).clamp(0, 1)
    dn.sampler = "multistep"
    dn.configure_multistep(ms_steps=1, ms_sigma_max=0.3, ms_sigma_min=0.3,
                           ms_guidance=0.0, ms_use_confidence=False,
                           ms_stochastic=False)
    torch.manual_seed(0)
    mu_cav0 = prior.llr_to_soft_field(cavities[0])
    ms1 = prior._multistep_sample(mu_cav0, cavities[0])
    a = ss.detach().cpu().numpy().reshape(-1); b = ms1.detach().cpu().numpy().reshape(-1)
    corr = float(np.corrcoef(a, b)[0, 1]); rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    print(f"  corr(single_shot, multistep K=1 g=0) = {corr:.4f}   rmse = {rmse:.4f}")
    print(f"  (not bit-exact by design — K=1 init adds σ0·ε; high corr = sane reduction)")

    # ---- checks 2+3: gray-vs-sharp montage + trajectory -----------------------
    print("\n=== [2] gray-vs-sharp montage + [3] trajectory ===")
    cols = ["clean", "cavity μ", "single-shot σ=0.3", "single-shot σ=1.5", "multistep"]
    fig, ax = plt.subplots(N, len(cols), figsize=(2.1 * len(cols), 2.1 * N))
    for i in range(N):
        cav = cavities[i]
        mu_cav = prior.llr_to_soft_field(cav)
        # single-shot at small and LARGE sigma
        dn.sampler = "single_shot"
        ss_lo = prior.net(mu_cav, torch.tensor([0.3], device=DEV)).clamp(0, 1)
        ss_hi = prior.net(mu_cav, torch.tensor([1.5], device=DEV)).clamp(0, 1)
        # multistep
        dn.sampler = "multistep"
        dn.configure_multistep(ms_steps=18, ms_sigma_max=1.5, ms_sigma_min=0.05,
                               ms_guidance=0.6, ms_guidance_const=False,
                               ms_use_confidence=True, ms_stochastic=False)
        torch.manual_seed(100 + i)
        ms = prior._multistep_sample(mu_cav, cav)
        panels = [clean[i], to_img(mu_cav), to_img(ss_lo), to_img(ss_hi), to_img(ms)]
        p_lo = psnr(clean[i], to_img(ss_lo)); p_hi = psnr(clean[i], to_img(ss_hi))
        p_ms = psnr(clean[i], to_img(ms))
        # sharpness = per-image pixel std (gray blur → low std)
        std_hi = float(np.std(to_img(ss_hi))); std_ms = float(np.std(to_img(ms)))
        for c in range(len(cols)):
            ax[i, c].imshow(panels[c], cmap="gray", vmin=0, vmax=1)
            ax[i, c].set_xticks([]); ax[i, c].set_yticks([])
            if i == 0:
                ax[i, c].set_title(cols[c], fontsize=9)
        ax[i, 3].set_xlabel(f"PSNR {p_hi:.1f} std {std_hi:.3f}", fontsize=8)
        ax[i, 4].set_xlabel(f"PSNR {p_ms:.1f} std {std_ms:.3f}", fontsize=8)
        print(f"  img {i}: single-shot σ1.5 PSNR={p_hi:5.1f} std={std_hi:.3f} | "
              f"multistep PSNR={p_ms:5.1f} std={std_ms:.3f}  "
              f"(single-shot σ0.3 PSNR={p_lo:.1f})")
        if i == 0:
            print("  [3] trajectory (img 0):")
            for t in prior.last_ms_trace:
                print(f"      step {t['step']:2d}  σ={t['sigma']:.3f}  ζ={t['zeta']:.3f}  "
                      f"x0_mean={t['x0_mean']:.3f}  dist_cav={t['dist_cavity']:.3f}")
    os.makedirs("results", exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=110)
    print(f"\n  montage -> {OUT}")
    print("  KEY: if single-shot σ=1.5 std ≪ multistep std, single-shot grayed out "
          "and multistep selected a sharp mode (the 2a raison d'être).")


if __name__ == "__main__":
    main()
