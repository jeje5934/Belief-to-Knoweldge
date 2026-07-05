"""
Denoiser training-state probe.

Questions (user):
  1. What training-degree does the active checkpoint (checkpoints/denoiser.pt)
     sit at?  (no epoch metadata is stored, so we locate it by its TEST MSE.)
  2. Does training longer (up to 100 epochs) push the TEST denoising MSE below
     0.1 — i.e. is the current ~0.1 due to (a) under-training or (b) intrinsic
     data difficulty / EDM-DSM floor?
  3. Plot train/test curves vs epoch: still decreasing (under-trained) or
     plateaued (difficulty floor)?

Metric — held-out FashionMNIST TEST split (never seen in training):
  * test_mse   : UNWEIGHTED per-pixel denoising MSE  mean ||D(x+σn;σ) − x||²
  * test_wloss : the EDM-DSM WEIGHTED loss (same objective as training's "loss=")
  Both averaged over a FIXED (image, σ, noise) eval set (fixed seed) so the
  per-epoch curve is directly comparable (no eval-noise jitter).

Safety: trains a FRESH model; the probe checkpoint is saved to --ckpt-out
(default checkpoints/denoiser_probe.pt) — the working checkpoints/denoiser.pt
is NEVER overwritten (only read, as the reference line).

Usage:
  CUDA_VISIBLE_DEVICES=0 python denoiser_training_probe.py \
      --epochs 100 --eval-every 1 --eval-n 2000
"""
import argparse
import os
import time
import json
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T

from source_prior import SourcePriorDenoiser

IMG_H, IMG_W = 28, 28
CKPT_DIR = "checkpoints"
SIGMA_DATA = 0.5
P_MEAN, P_STD = -1.2, 1.2
EVAL_SEED = 1234


def load_split(train: bool):
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=train, download=True, transform=T.ToTensor())
    return torch.stack([ds[i][0] for i in range(len(ds))])   # [N,1,28,28] in [0,1]


def make_eval_set(test_imgs, n, device):
    """Fixed (image, sigma, noise) eval triples — identical every epoch."""
    g = torch.Generator().manual_seed(EVAL_SEED)
    idx = torch.randperm(test_imgs.shape[0], generator=g)[:n]
    x = test_imgs[idx].clone()
    ln_sigma = torch.randn(n, generator=g) * P_STD + P_MEAN
    sigma = ln_sigma.exp()
    noise = torch.randn(n, 1, IMG_H, IMG_W, generator=g)
    return (x.to(device), sigma.to(device), noise.to(device))


@torch.no_grad()
def evaluate(model, eval_set, batch=512):
    x, sigma, noise = eval_set
    model.eval()
    n = x.shape[0]
    sd = SIGMA_DATA
    se_sum = 0.0          # unweighted squared error sum (over all pixels)
    wl_sum = 0.0          # weighted loss sum (per-sample mean, summed)
    npix = 0
    for i in range(0, n, batch):
        xb = x[i:i + batch]
        sb = sigma[i:i + batch]
        nb = noise[i:i + batch]
        noisy = xb + sb.reshape(-1, 1, 1, 1) * nb
        den = model.net(noisy, sb)
        se = (den - xb) ** 2
        se_sum += se.sum().item()
        npix += se.numel()
        w = (sb ** 2 + sd ** 2) / (sb * sd) ** 2
        wl_sum += (w.reshape(-1, 1, 1, 1) * se).mean(dim=(1, 2, 3)).sum().item()
    model.train()
    return se_sum / npix, wl_sum / n   # (unweighted MSE, weighted DSM loss)


def build_model(device):
    return SourcePriorDenoiser(
        img_h=IMG_H, img_w=IMG_W, bits_per_pixel=8,
        model_channels=64, channel_mult=(1, 2, 2),
        num_blocks=2, attn_resolutions=(7,),
        sigma_data=SIGMA_DATA).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--ref-ckpt", default="checkpoints/denoiser.pt")
    ap.add_argument("--ckpt-out", default="checkpoints/denoiser_probe.pt")
    ap.add_argument("--save-at", type=int, nargs="*", default=[5, 20, 50, 100])
    ap.add_argument("--out-prefix", default="results/denoiser_training_probe")
    args = ap.parse_args()

    assert os.path.abspath(args.ckpt_out) != os.path.abspath(args.ref_ckpt), \
        "refuse to overwrite the working denoiser.pt"
    os.makedirs("results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device={device}  epochs={args.epochs}  eval_n={args.eval_n}")
    train_imgs = load_split(train=True)
    test_imgs = load_split(train=False)
    print(f"train={tuple(train_imgs.shape)}  test={tuple(test_imgs.shape)}")
    eval_set = make_eval_set(test_imgs, args.eval_n, device)

    loader = DataLoader(TensorDataset(train_imgs), batch_size=args.batch,
                        shuffle=True, drop_last=True, num_workers=0)

    # ── Reference: the active working checkpoint, located by its test MSE ──
    ref = build_model(device)
    ref_mse = ref_wl = None
    if os.path.isfile(args.ref_ckpt):
        ref.load_state_dict(torch.load(args.ref_ckpt, map_location=device))
        ref_mse, ref_wl = evaluate(ref, eval_set)
        print(f"[REF] {args.ref_ckpt}: test_mse={ref_mse:.4f}  "
              f"test_wloss={ref_wl:.4f}")
    del ref
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Fresh training run with per-epoch test eval ──
    model = build_model(device)
    model.train()
    opt = torch.optim.Adam(model.net.parameters(), lr=args.lr)
    sd = SIGMA_DATA

    rows = []
    # epoch 0 = randomly initialized baseline
    e0_mse, e0_wl = evaluate(model, eval_set)
    print(f"Epoch  0  (init)  test_mse={e0_mse:.4f}  test_wloss={e0_wl:.4f}")
    rows.append({"epoch": 0, "train_loss": None,
                 "test_mse": e0_mse, "test_wloss": e0_wl})

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        for (x,) in loader:
            x = x.to(device)
            B = x.shape[0]
            ln_sigma = torch.randn(B, device=device) * P_STD + P_MEAN
            sigma = ln_sigma.exp()
            noise = torch.randn_like(x)
            noisy = x + sigma.reshape(-1, 1, 1, 1) * noise
            den = model.net(noisy, sigma)
            w = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
            loss = (w.reshape(-1, 1, 1, 1) * (den - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        dt = time.time() - t0

        row = {"epoch": epoch, "train_loss": train_loss,
               "test_mse": None, "test_wloss": None}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            mse, wl = evaluate(model, eval_set)
            row["test_mse"], row["test_wloss"] = mse, wl
            print(f"Epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.4f}  "
                  f"test_mse={mse:.4f}  test_wloss={wl:.4f}  ({dt:.0f}s)")
        else:
            print(f"Epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.4f}  "
                  f"({dt:.0f}s)")
        rows.append(row)

        if epoch in args.save_at:
            p = args.ckpt_out.replace(".pt", f"_ep{epoch}.pt")
            torch.save(model.state_dict(), p)
            print(f"  saved probe checkpoint → {p}")

        # incremental CSV
        with open(args.out_prefix + ".csv", "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["epoch", "train_loss", "test_mse", "test_wloss"])
            for r in rows:
                wtr.writerow([r["epoch"], r["train_loss"],
                              r["test_mse"], r["test_wloss"]])

    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"epochs": args.epochs, "eval_n": args.eval_n,
                   "ref_ckpt": args.ref_ckpt, "ref_test_mse": ref_mse,
                   "ref_test_wloss": ref_wl, "rows": rows}, f, indent=2)

    _plot(rows, ref_mse, args.out_prefix + ".png")
    print(f"\nsaved: {args.out_prefix}.csv/.png/.json")


def _plot(rows, ref_mse, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = [r for r in rows if r["test_mse"] is not None]
    ep = [r["epoch"] for r in ev]
    mse = [r["test_mse"] for r in ev]
    tl = [(r["epoch"], r["train_loss"]) for r in rows if r["train_loss"] is not None]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(ep, mse, "o-", label="test denoising MSE (unweighted)")
    ax1.axhline(0.1, color="r", ls="--", alpha=0.6, label="MSE = 0.1 ref")
    if ref_mse is not None:
        ax1.axhline(ref_mse, color="g", ls=":", alpha=0.8,
                    label=f"active denoiser.pt ({ref_mse:.3f})")
    ax1.axvline(5, color="gray", ls=":", alpha=0.5, label="README default (5 ep)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("test MSE")
    ax1.set_title("Test denoising MSE vs epoch"); ax1.grid(alpha=0.3); ax1.legend()
    twl = [(r["epoch"], r["test_wloss"]) for r in ev]
    ax2.plot([e for e, _ in tl], [v for _, v in tl], "s-", alpha=0.7,
             label="train weighted loss")
    ax2.plot([e for e, _ in twl], [v for _, v in twl], "o-",
             label="test weighted loss (DSM obj)")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("weighted DSM loss")
    ax2.set_title("EDM-DSM objective vs epoch"); ax2.grid(alpha=0.3); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
