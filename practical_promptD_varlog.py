"""
Prompt D, step 9 — log the diagonal Tweedie per-pixel variance distribution on
Trouser, contrasted with multi-cat.

For each denoiser, on its own test images at a representative decode σ, compute
the diagonal Tweedie per-pixel std  √(σ²·∂D/∂x̃)  (source_prior._tweedie_pixel_std,
0..255 units, clamped [floor, cap]) and report:
  * distribution over pixels (percentiles, mean, floor/cap saturation)
  * bit-plane mean|LLR| when this per-pixel Tweedie std drives the read-out,
    vs the fixed sigma_post=3.0 — how the input-dependent variance re-weights bits.

Contrast Trouser (unimodal) vs multi-cat: is the Trouser variance smaller / more
concentrated (which would indicate a more unimodal posterior)?
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
EVAL_SEED = 1234
REPR_SIGMA = 0.30
BIT_W = [2 ** (BPP - 1 - i) for i in range(BPP)]


def load_split(train, label=None):
    ds = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=train,
                                            download=True, transform=T.ToTensor())
    idx = range(len(ds)) if label is None else np.where(ds.targets.numpy() == label)[0]
    return torch.stack([ds[i][0] for i in idx])


def build(device):
    return SourcePriorDenoiser(img_h=IMG_H, img_w=IMG_W, bits_per_pixel=BPP,
                               model_channels=64, channel_mult=(1, 2, 2),
                               num_blocks=2, attn_resolutions=(7,),
                               sigma_data=SIGMA_DATA,
                               tweedie_precision=True).to(device).eval()


@torch.no_grad()
def analyze(prior, imgs, device, n=512):
    g = torch.Generator().manual_seed(EVAL_SEED + 3)
    n = min(n, imgs.shape[0])
    idx = torch.randperm(imgs.shape[0], generator=g)[:n]
    x = imgs[idx].to(device)
    sig = torch.full((n,), REPR_SIGMA, device=device)
    mu_cav = (x + REPR_SIGMA * torch.randn(n, 1, IMG_H, IMG_W, generator=g).to(device))
    mu_proj = prior.net(mu_cav, sig).clamp(0, 1)
    std_pix = prior._tweedie_pixel_std(mu_cav, mu_proj, sig)   # [n, N_PIX], 0..255
    s = std_pix.flatten().cpu().numpy()
    pct = {f"p{q}": float(np.percentile(s, q)) for q in (1, 10, 25, 50, 75, 90, 99)}
    dist = {"mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
            "frac_at_floor": float((s <= prior.tweedie_std_floor + 1e-6).mean()),
            "frac_at_cap": float((s >= prior.tweedie_std_cap - 1e-6).mean()),
            "floor": prior.tweedie_std_floor, "cap": prior.tweedie_std_cap, **pct}
    # bit-plane mean|LLR|: Tweedie per-pixel std vs fixed 3.0
    llr_tw = prior.soft_field_to_posterior_logits(mu_proj, posterior_pixel_std=std_pix)
    old = prior.sigma_post; prior.sigma_post = 3.0
    llr_3 = prior.soft_field_to_posterior_logits(mu_proj); prior.sigma_post = old
    bp_tw = llr_tw.abs().reshape(n, N_PIX, BPP).mean(dim=(0, 1)).cpu().numpy()
    bp_3 = llr_3.abs().reshape(n, N_PIX, BPP).mean(dim=(0, 1)).cpu().numpy()
    return dist, bp_tw, bp_3


def main():
    os.makedirs("results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets = {"trouser": ("checkpoints/denoiser_trouser.pt",
                        load_split(False, TROUSER_LABEL)),
            "multi": ("checkpoints/denoiser.pt", load_split(False, None))}
    out = {"repr_sigma": REPR_SIGMA, "bit_place_values": BIT_W, "denoisers": {}}
    print(f"device={device}  representative σ={REPR_SIGMA}\n")
    for name, (ck, test) in sets.items():
        prior = build(device)
        if not os.path.isfile(ck):
            print(f"[WARN] missing {ck}"); continue
        prior.load_state_dict(torch.load(ck, map_location=device))
        dist, bp_tw, bp_3 = analyze(prior, test, device)
        out["denoisers"][name] = {"ckpt": ck, "tweedie_std_dist": dist,
                                  "bitplane_llr_tweedie": bp_tw.tolist(),
                                  "bitplane_llr_sp3": bp_3.tolist()}
        print(f"[{name}] diagonal Tweedie per-pixel std (0..255, clamp "
              f"[{dist['floor']},{dist['cap']}]):")
        print(f"  mean={dist['mean']:.1f} median={dist['p50']:.1f} "
              f"p10={dist['p10']:.1f} p90={dist['p90']:.1f} "
              f"min={dist['min']:.1f} max={dist['max']:.1f}")
        print(f"  saturation: at floor={dist['frac_at_floor']*100:.1f}%  "
              f"at cap={dist['frac_at_cap']*100:.1f}%")
        print(f"  bit-plane mean|LLR| (MSB→LSB):")
        print(f"    w_m     : " + "  ".join(f"{w:>5}" for w in BIT_W))
        print(f"    Tweedie : " + "  ".join(f"{v:5.2f}" for v in bp_tw))
        print(f"    sp=3.0  : " + "  ".join(f"{v:5.2f}" for v in bp_3))
        print()
        del prior
        if device == "cuda":
            torch.cuda.empty_cache()

    with open("results/promptD_varlog.json", "w") as f:
        json.dump(out, f, indent=2)
    _plot(out, "results/promptD_varlog.png")
    print("saved: results/promptD_varlog.png/.json")


def _plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name in ("trouser", "multi"):
        d = out["denoisers"].get(name)
        if not d:
            continue
        dd = d["tweedie_std_dist"]
        xs = [dd[k] for k in ("p1", "p10", "p25", "p50", "p75", "p90", "p99")]
        qs = [1, 10, 25, 50, 75, 90, 99]
        ax1.plot(qs, xs, "o-", label=f"{name} (mean {dd['mean']:.1f})")
        ax2.plot(range(BPP), d["bitplane_llr_tweedie"], "o-",
                 label=f"{name} Tweedie")
    ax1.set_xlabel("percentile"); ax1.set_ylabel("per-pixel Tweedie std (0..255)")
    ax1.set_title("Diagonal Tweedie variance distribution (Trouser vs multi)")
    ax1.grid(alpha=0.3); ax1.legend()
    ax2.set_xlabel("bit plane (MSB→LSB)"); ax2.set_ylabel("mean |LLR|")
    ax2.set_yscale("log"); ax2.set_xticks(range(BPP))
    ax2.set_title("Bit-plane |LLR| under Tweedie precision")
    ax2.grid(alpha=0.3); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
