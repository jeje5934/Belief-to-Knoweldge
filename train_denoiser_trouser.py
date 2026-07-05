"""
Prompt A — single-category (Trouser) denoiser retraining.

Goal: a UNIMODAL source (no cross-category multimodality) so later full-EP
diagnostics aren't confounded by the garment-class multimodality of the
multi-category denoiser.  Trouser (FashionMNIST label 1) has the most consistent
shape → closest to unimodal.

Design:
  * Filter FashionMNIST to label==1 only, KEEP the train/test split
    (~6000 train / 1000 test).
  * Retrain a FRESH denoiser on Trouser-train.  Single category = ~1/10 the
    data, so more EPOCHS are needed to reach the same # gradient steps; we train
    with patience-based early stop until the Trouser-test MSE plateaus, so the
    "unimodal effect" is NOT confounded with under-training.
  * Save to a SEPARATE checkpoint (default checkpoints/denoiser_trouser.pt);
    the working multi-category checkpoints/denoiser.pt is never touched.
  * Report train/test denoising-MSE curve, final test MSE, and compare against
    the multi-category denoiser evaluated (a) on the SAME Trouser-test set and
    (b) its full-test reference (~0.0105).

Crash-safety (see docs/COMPUTE_LESSONS.md): incremental CSV per eval,
intermediate checkpoints, and --resume to continue after a GPU fault.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_denoiser_trouser.py \
      --max-epochs 400 --eval-every 5 --patience 8
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
TROUSER_LABEL = 1
SIGMA_DATA = 0.5
P_MEAN, P_STD = -1.2, 1.2
EVAL_SEED = 1234
REF_FULLTEST_MSE = 0.0105     # multi-cat denoiser.pt on the FULL FMNIST test set


def load_split(train, label=None):
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=train, download=True, transform=T.ToTensor())
    if label is None:
        idx = range(len(ds))
    else:
        targets = ds.targets.numpy()
        idx = np.where(targets == label)[0]
    return torch.stack([ds[i][0] for i in idx])   # [N,1,28,28] in [0,1]


def make_eval_set(imgs, n, device):
    g = torch.Generator().manual_seed(EVAL_SEED)
    n = min(n, imgs.shape[0])
    idx = torch.randperm(imgs.shape[0], generator=g)[:n]
    x = imgs[idx].clone()
    ln_sigma = torch.randn(n, generator=g) * P_STD + P_MEAN
    sigma = ln_sigma.exp()
    noise = torch.randn(n, 1, IMG_H, IMG_W, generator=g)
    return (x.to(device), sigma.to(device), noise.to(device))


@torch.no_grad()
def evaluate(net, eval_set, batch=512):
    x, sigma, noise = eval_set
    was_training = net.training
    net.eval()
    n = x.shape[0]
    sd = SIGMA_DATA
    se_sum = wl_sum = 0.0
    npix = 0
    for i in range(0, n, batch):
        xb, sb, nb = x[i:i+batch], sigma[i:i+batch], noise[i:i+batch]
        den = net(xb + sb.reshape(-1, 1, 1, 1) * nb, sb)
        se = (den - xb) ** 2
        se_sum += se.sum().item(); npix += se.numel()
        w = (sb ** 2 + sd ** 2) / (sb * sd) ** 2
        wl_sum += (w.reshape(-1, 1, 1, 1) * se).mean(dim=(1, 2, 3)).sum().item()
    if was_training:
        net.train()
    return se_sum / npix, wl_sum / n


def build_model(device):
    return SourcePriorDenoiser(
        img_h=IMG_H, img_w=IMG_W, bits_per_pixel=8,
        model_channels=64, channel_mult=(1, 2, 2),
        num_blocks=2, attn_resolutions=(7,),
        sigma_data=SIGMA_DATA).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=8,
                    help="stop if test MSE has not improved > min-delta for this "
                         "many consecutive evals")
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--ref-ckpt", default="checkpoints/denoiser.pt")
    ap.add_argument("--ckpt-out", default="checkpoints/denoiser_trouser.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-prefix", default="results/denoiser_trouser")
    args = ap.parse_args()

    assert os.path.abspath(args.ckpt_out) != os.path.abspath(args.ref_ckpt), \
        "refuse to overwrite the working denoiser.pt"
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = load_split(train=True, label=TROUSER_LABEL)
    te = load_split(train=False, label=TROUSER_LABEL)
    print(f"device={device}  Trouser train={tuple(tr.shape)}  test={tuple(te.shape)}")
    steps_per_epoch = tr.shape[0] // args.batch
    print(f"steps/epoch={steps_per_epoch} (multi-cat had ~468 => need ~{468//max(steps_per_epoch,1)}× "
          f"more epochs for equal gradient steps)")

    train_eval = make_eval_set(tr, 1000, device)     # fixed train-subset eval
    test_eval = make_eval_set(te, te.shape[0], device)  # all Trouser test

    # ── multi-category reference on the SAME Trouser test set ──
    ref_on_trouser = None
    if os.path.isfile(args.ref_ckpt):
        ref = build_model(device)
        ref.load_state_dict(torch.load(args.ref_ckpt, map_location=device))
        rmse, rwl = evaluate(ref.net, test_eval)
        ref_on_trouser = rmse
        print(f"[REF] multi-cat denoiser.pt on Trouser-test: mse={rmse:.4f} "
              f"wloss={rwl:.4f}  (full-test ref mse={REF_FULLTEST_MSE})")
        del ref
        if device == "cuda":
            torch.cuda.empty_cache()

    loader = DataLoader(TensorDataset(tr), batch_size=args.batch, shuffle=True,
                        drop_last=True, num_workers=0)
    model = build_model(device)
    opt = torch.optim.Adam(model.net.parameters(), lr=args.lr)

    rows = []
    start_epoch = 1
    best_mse = float("inf")
    if args.resume and os.path.isfile(args.ckpt_out):
        model.load_state_dict(torch.load(args.ckpt_out, map_location=device))
        if os.path.isfile(args.out_prefix + ".csv"):
            for r in csv.DictReader(open(args.out_prefix + ".csv")):
                rows.append({k: (float(v) if v not in ("", "None") else None)
                             for k, v in r.items()})
                rows[-1]["epoch"] = int(rows[-1]["epoch"])
                if rows[-1]["test_mse"] is not None:
                    best_mse = min(best_mse, rows[-1]["test_mse"])
            start_epoch = rows[-1]["epoch"] + 1
        print(f"[RESUME] from epoch {start_epoch}, best_mse={best_mse:.4f}")

    model.train()
    sd = SIGMA_DATA
    no_improve = 0

    def dump_csv():
        with open(args.out_prefix + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_mse", "test_mse", "test_wloss"])
            for r in rows:
                w.writerow([r["epoch"], r.get("train_loss"), r.get("train_mse"),
                            r.get("test_mse"), r.get("test_wloss")])

    if start_epoch == 1:
        m0, w0 = evaluate(model.net, test_eval)
        tm0, _ = evaluate(model.net, train_eval)
        rows.append({"epoch": 0, "train_loss": None, "train_mse": tm0,
                     "test_mse": m0, "test_wloss": w0})
        print(f"Epoch   0 (init)  train_mse={tm0:.4f}  test_mse={m0:.4f}")
        dump_csv()

    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        losses = []
        for (x,) in loader:
            x = x.to(device); B = x.shape[0]
            sigma = (torch.randn(B, device=device) * P_STD + P_MEAN).exp()
            noisy = x + sigma.reshape(-1, 1, 1, 1) * torch.randn_like(x)
            den = model.net(noisy, sigma)
            w = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
            loss = (w.reshape(-1, 1, 1, 1) * (den - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses)); dt = time.time() - t0

        row = {"epoch": epoch, "train_loss": train_loss, "train_mse": None,
               "test_mse": None, "test_wloss": None}
        if epoch % args.eval_every == 0 or epoch == args.max_epochs:
            tmse, _ = evaluate(model.net, train_eval)
            mse, wl = evaluate(model.net, test_eval)
            row.update(train_mse=tmse, test_mse=mse, test_wloss=wl)
            improved = mse < best_mse - args.min_delta
            print(f"Epoch {epoch:>3}  train_loss={train_loss:.4f}  "
                  f"train_mse={tmse:.4f}  test_mse={mse:.4f}  "
                  f"({dt:.1f}s){'  *best' if improved else ''}")
            if improved:
                best_mse = mse; no_improve = 0
                torch.save(model.state_dict(), args.ckpt_out)
            else:
                no_improve += 1
            rows.append(row); dump_csv()
            # intermediate snapshot
            torch.save(model.state_dict(),
                       args.ckpt_out.replace(".pt", "_last.pt"))
            if no_improve >= args.patience:
                print(f"early stop: no test-MSE improvement for {args.patience} "
                      f"evals (best={best_mse:.4f})")
                break
        else:
            rows.append(row)

    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"trouser_label": TROUSER_LABEL, "best_test_mse": best_mse,
                   "ref_on_trouser_mse": ref_on_trouser,
                   "ref_fulltest_mse": REF_FULLTEST_MSE, "rows": rows}, f, indent=2)
    _plot(rows, best_mse, ref_on_trouser, args.out_prefix + ".png")
    print(f"\nbest Trouser-test MSE={best_mse:.4f}  "
          f"(multi-cat on Trouser-test={ref_on_trouser}, "
          f"multi-cat full-test={REF_FULLTEST_MSE})")
    print(f"saved best → {args.ckpt_out}; curves → {args.out_prefix}.*")


def _plot(rows, best_mse, ref_on_trouser, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = [r for r in rows if r.get("test_mse") is not None]
    ep = [r["epoch"] for r in ev]
    te = [r["test_mse"] for r in ev]
    trn = [r["train_mse"] for r in ev if r.get("train_mse") is not None]
    ep_tr = [r["epoch"] for r in ev if r.get("train_mse") is not None]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, te, "o-", ms=4, label="Trouser TEST denoising MSE")
    if ep_tr:
        ax.plot(ep_tr, trn, "s--", ms=3, alpha=0.6, label="Trouser TRAIN MSE")
    ax.axhline(REF_FULLTEST_MSE, color="r", ls="--", alpha=0.7,
               label=f"multi-cat full-test MSE ({REF_FULLTEST_MSE})")
    if ref_on_trouser is not None:
        ax.axhline(ref_on_trouser, color="g", ls=":", alpha=0.8,
                   label=f"multi-cat on Trouser-test ({ref_on_trouser:.4f})")
    ax.set_xlabel("epoch"); ax.set_ylabel("denoising MSE")
    ax.set_title("Trouser (unimodal) denoiser retraining vs multi-category")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
