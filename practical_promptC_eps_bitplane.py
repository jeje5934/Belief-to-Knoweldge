"""
Prompt C, steps 1 & 3 — measure the denoiser error scale ε and verify the
bit-plane LLR stratification when sigma_post = ε (vs the tuned 3.0).

Step 1: ε = 255·√(test MSE) for each denoiser, in the code's 0..255 pixel units
        (sigma_post units).  Multi-cat on full FMNIST test; Trouser on Trouser
        test.  Also residual-distribution stats (Gaussianity: skew, excess
        kurtosis, >3σ tail fraction).

Step 3: per bit-plane (MSB→LSB) mean|proj_LLR| for sigma_post ∈ {3.0, ε}, on the
        denoiser's projected posterior mean.  Confirms that ε auto-stratifies
        confidence: MSB (w_m ≫ ε) stay sharp, LSB (w_m ≲ ε) collapse to ~0.

Fast; GPU ok (short).  Does not touch checkpoints/denoiser.pt.
"""
import os
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from source_prior import SourcePriorDenoiser

IMG_H, IMG_W, BPP = 28, 28, 8
N_PIX = IMG_H * IMG_W
TROUSER_LABEL = 1
SIGMA_DATA = 0.5
P_MEAN, P_STD = -1.2, 1.2
EVAL_SEED = 1234
REPR_SIGMA = 0.30          # representative decode σ for the bit-plane read-out
BIT_W = [2 ** (BPP - 1 - i) for i in range(BPP)]   # place values, index0=MSB=128


def load_split(train, label=None):
    ds = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=train,
                                           download=True, transform=T.ToTensor())
    if label is None:
        idx = range(len(ds))
    else:
        idx = np.where(ds.targets.numpy() == label)[0]
    return torch.stack([ds[i][0] for i in idx])


def build(device):
    return SourcePriorDenoiser(img_h=IMG_H, img_w=IMG_W, bits_per_pixel=BPP,
                               model_channels=64, channel_mult=(1, 2, 2),
                               num_blocks=2, attn_resolutions=(7,),
                               sigma_data=SIGMA_DATA).to(device).eval()


@torch.no_grad()
def measure_eps(net_prior, imgs, device, n=2000):
    """RMSE (0..1) over σ~lognormal, its ε=255·RMSE, and residual stats."""
    g = torch.Generator().manual_seed(EVAL_SEED)
    n = min(n, imgs.shape[0])
    idx = torch.randperm(imgs.shape[0], generator=g)[:n]
    x = imgs[idx].to(device)
    sigma = (torch.randn(n, generator=g) * P_STD + P_MEAN).exp().to(device)
    noise = torch.randn(n, 1, IMG_H, IMG_W, generator=g).to(device)
    resid = []
    se = 0.0
    npix = 0
    for i in range(0, n, 512):
        xb, sb, nb = x[i:i+512], sigma[i:i+512], noise[i:i+512]
        den = net_prior.net(xb + sb.reshape(-1, 1, 1, 1) * nb, sb).clamp(0, 1)
        r = (den - xb)
        se += (r ** 2).sum().item(); npix += r.numel()
        resid.append(r.flatten().cpu())
    mse = se / npix
    rmse01 = mse ** 0.5
    eps255 = 255.0 * rmse01
    r = torch.cat(resid).numpy()
    mu, sd = float(r.mean()), float(r.std())
    z = (r - mu) / (sd + 1e-12)
    stats = {"mse": mse, "rmse01": rmse01, "eps255": eps255,
             "resid_mean": mu, "resid_std": sd,
             "skew": float((z ** 3).mean()),
             "excess_kurtosis": float((z ** 4).mean() - 3.0),
             "tail_gt3sigma_frac": float((np.abs(z) > 3).mean()),
             "gaussian_ref_tail": 0.0027}
    return stats


@torch.no_grad()
def bitplane_llr(prior, imgs, device, sigma_post, n=512):
    """mean|proj_LLR| per bit-plane for a given sigma_post, using the denoiser's
    projected posterior mean on noisy inputs at REPR_SIGMA."""
    g = torch.Generator().manual_seed(EVAL_SEED + 7)
    n = min(n, imgs.shape[0])
    idx = torch.randperm(imgs.shape[0], generator=g)[:n]
    x = imgs[idx].to(device)
    sig = torch.full((n,), REPR_SIGMA, device=device)
    noisy = x + REPR_SIGMA * torch.randn(n, 1, IMG_H, IMG_W, generator=g).to(device)
    mu_proj = prior.net(noisy, sig).clamp(0, 1)
    old = prior.sigma_post
    prior.sigma_post = float(sigma_post)
    llr = prior.soft_field_to_posterior_logits(mu_proj)   # [n, N_PIX*BPP]
    prior.sigma_post = old
    a = llr.abs().reshape(n, N_PIX, BPP).mean(dim=(0, 1)).cpu().numpy()  # [BPP]
    return a   # index 0 = MSB


def main():
    os.makedirs("results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fmnist_test = load_split(train=False, label=None)
    trouser_test = load_split(train=False, label=TROUSER_LABEL)

    denoisers = {
        "multi": ("checkpoints/denoiser.pt", fmnist_test),
        "trouser": ("checkpoints/denoiser_trouser.pt", trouser_test),
    }
    out = {"repr_sigma": REPR_SIGMA, "bit_place_values": BIT_W, "denoisers": {}}
    print(f"device={device}  representative read-out σ={REPR_SIGMA}\n")
    print("=== Step 1: error scale ε and residual Gaussianity ===")
    for name, (ck, test) in denoisers.items():
        prior = build(device)
        if os.path.isfile(ck):
            prior.load_state_dict(torch.load(ck, map_location=device))
        else:
            print(f"[WARN] missing {ck}"); continue
        st = measure_eps(prior, test, device)
        eps = st["eps255"]
        bp3 = bitplane_llr(prior, test, device, 3.0)
        bpE = bitplane_llr(prior, test, device, eps)
        out["denoisers"][name] = {"ckpt": ck, **st,
                                  "bitplane_llr_sp3": bp3.tolist(),
                                  "bitplane_llr_epsilon": bpE.tolist()}
        print(f"\n[{name}]  ckpt={ck}")
        print(f"  MSE(0..1)={st['mse']:.5f}  RMSE={st['rmse01']:.4f}  "
              f"=> ε(0..255)={eps:.2f}   (vs tuned sigma_post=3.0)")
        print(f"  residual: mean={st['resid_mean']:.4f} std={st['resid_std']:.4f} "
              f"skew={st['skew']:.2f} exKurt={st['excess_kurtosis']:.2f} "
              f">3σ tail={st['tail_gt3sigma_frac']*100:.2f}% (Gauss 0.27%)")
        print(f"  bit-plane mean|LLR|  (MSB→LSB, place-value w_m):")
        print(f"    w_m   : " + "  ".join(f"{w:>5}" for w in BIT_W))
        print(f"    sp=3.0: " + "  ".join(f"{v:5.2f}" for v in bp3))
        print(f"    sp=ε  : " + "  ".join(f"{v:5.2f}" for v in bpE))
        del prior
        if device == "cuda":
            torch.cuda.empty_cache()

    with open("results/promptC_eps_bitplane.json", "w") as f:
        json.dump(out, f, indent=2)
    _plot(out, "results/promptC_eps_bitplane.png")
    print("\nsaved: results/promptC_eps_bitplane.png/.json")


def _plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    planes = list(range(BPP))
    xt = [f"b{i}\n(w{BIT_W[i]})" for i in planes]
    for ax, name in zip(axes, ("multi", "trouser")):
        d = out["denoisers"].get(name)
        if not d:
            continue
        ax.plot(planes, d["bitplane_llr_sp3"], "s-", label="sigma_post=3.0 (tuned)")
        ax.plot(planes, d["bitplane_llr_epsilon"], "o-",
                label=f"sigma_post=ε={d['eps255']:.1f} (auto_mse)")
        ax.set_title(f"{name} denoiser — bit-plane mean|LLR|")
        ax.set_xlabel("bit plane (MSB→LSB)"); ax.set_xticks(planes)
        ax.set_xticklabels(xt, fontsize=8); ax.grid(alpha=0.3); ax.legend()
        ax.set_yscale("log")
    axes[0].set_ylabel("mean |proj LLR|  (log)")
    fig.suptitle("Prompt C step 3: ε read-out stratifies bit confidence "
                 "(MSB sharp, LSB→0)")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
