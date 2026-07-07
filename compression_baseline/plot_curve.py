"""Plot the training learning curve (train bpp per step, test bpp per epoch)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")

with open(os.path.join(RESULTS, "train_log.json")) as f:
    log = json.load(f)

steps = log["train_bpp_step"]
# global step index = (epoch-1)*steps_per_epoch + it ; approximate with running order
xs = list(range(len(steps)))
train_bpp = [s[2] for s in steps]

test = log["test_bpp_epoch"]
# map each test point to its position on the train-step x-axis
epochs = [t[0] for t in test]
test_bpp = [t[1] for t in test]
per_ep = max(1, len(steps) // max(1, len(test)))
test_x = [min(len(steps) - 1, e * per_ep) for e in epochs]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xs, train_bpp, lw=0.8, alpha=0.6, label="train bpp (per logged step)")
ax.plot(test_x, test_bpp, "o-", color="crimson", label="test bpp (per epoch)")
ax.axhspan(1.5, 2.5, color="green", alpha=0.08, label="prompt-expected 1.5-2.5")
best = log.get("best_test_bpp", min(test_bpp))
ax.axhline(best, ls="--", color="k", lw=0.8, label=f"best test bpp={best:.3f}")
ax.set_xlabel("logged training step")
ax.set_ylabel("bits / pixel (NLL)")
ax.set_title(f"Gated PixelCNN on Fashion-MNIST  "
             f"({log['params']/1e6:.2f}M params, ch={log['n_channels']}, "
             f"L={log['n_layers']})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(RESULTS, "learning_curve.png")
fig.savefig(out, dpi=120)
print("saved", out)
print(f"best test bpp = {best:.4f}  ({best*784:.1f} bits/image)")
