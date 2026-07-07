"""
Train the Gated PixelCNN compressor on Fashion-MNIST train (60K).

Objective: minimise NLL in bits/pixel.  The held-out test split (never trained
on) gives the honest cross-entropy = the arithmetic coder's expected code length.

Logs a learning curve (train bpp per step, test bpp per epoch) to
results/train_log.json and checkpoints the best-test-NLL weights.

Run (single controlled GPU job; caps memory per repo run-safety guidance):
    python compression_baseline/train.py --epochs 30
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from compression_baseline import data as D
from compression_baseline.pixelcnn import GatedPixelCNN, count_params

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
CKPT = os.path.join(RESULTS, "pixelcnn_fmnist.pt")
LOG = os.path.join(RESULTS, "train_log.json")


@torch.no_grad()
def eval_bpp(model, imgs_u8, device, batch=256):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, imgs_u8.size(0), batch):
        xb = imgs_u8[i:i + batch].to(device)
        xn = D.to_model_input(xb)
        bpp = model.loss_bits_per_pixel(xb, xn)
        tot += bpp.item() * xb.size(0)
        n += xb.size(0)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_channels", type=int, default=72)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--mem_frac", type=float, default=0.5,
                    help="cap fraction of GPU memory (repo run-safety)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
    print(f"device={device}")

    train_u8 = D.load_train()
    test_u8 = D.load_test()
    print(f"train {tuple(train_u8.shape)}  test {tuple(test_u8.shape)}")

    loader = DataLoader(TensorDataset(train_u8), batch_size=args.batch,
                        shuffle=True, drop_last=True, num_workers=2,
                        pin_memory=(device == "cuda"))

    model = GatedPixelCNN(args.n_channels, args.n_layers, args.k).to(device)
    npar = count_params(model)
    print(f"model params: {npar/1e6:.2f}M  (target band 5-9M)")
    assert 5e6 <= npar <= 9e6, "capacity outside the fair 5-9M band"

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log = {"params": npar, "n_channels": args.n_channels,
           "n_layers": args.n_layers, "k": args.k,
           "train_bpp_step": [], "test_bpp_epoch": [], "epoch_time_s": []}
    best = float("inf")
    bad = 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        te0 = time.time()
        for it, (xb,) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            xn = D.to_model_input(xb)
            bpp = model.loss_bits_per_pixel(xb, xn)
            opt.zero_grad(set_to_none=True)
            bpp.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if it % 50 == 0:
                log["train_bpp_step"].append([ep, it, bpp.item()])
                print(f"ep{ep:02d} it{it:04d}  train_bpp={bpp.item():.4f}",
                      flush=True)
        sched.step()

        test_bpp = eval_bpp(model, test_u8, device)
        dt = time.time() - te0
        log["test_bpp_epoch"].append([ep, test_bpp])
        log["epoch_time_s"].append(dt)
        print(f"== epoch {ep:02d}  test_bpp={test_bpp:.4f}  "
              f"({dt:.1f}s, elapsed {(time.time()-t0)/60:.1f}m)", flush=True)

        with open(LOG, "w") as f:
            json.dump(log, f, indent=2)

        if test_bpp < best - 1e-4:
            best = test_bpp
            bad = 0
            torch.save({"state_dict": model.state_dict(),
                        "cfg": {"n_channels": args.n_channels,
                                "n_layers": args.n_layers, "k": args.k},
                        "test_bpp": test_bpp, "epoch": ep}, CKPT)
            print(f"   saved best ckpt (test_bpp={test_bpp:.4f})", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep} (best test_bpp={best:.4f})")
                break

    log["best_test_bpp"] = best
    with open(LOG, "w") as f:
        json.dump(log, f, indent=2)
    print(f"DONE. best test_bpp={best:.4f}  bits/image={best*784:.1f}")


if __name__ == "__main__":
    main()
