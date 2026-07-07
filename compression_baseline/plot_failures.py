"""
Failure-quality montage: for CRC-FAILED blocks, show (original, reconstruction)
pairs for both systems and summarise PSNR.

ours   : payload IS the raw image, so a decode failure = scattered pixel errors
         -> the picture survives (graceful).
baseline: arithmetic coding is a sequential dependent code, so any surviving bit
         error desyncs the decoder -> the picture is destroyed (catastrophic).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.dirname(__file__) + "/results"


def load(tag):
    d = np.load(os.path.join(RESULTS, f"fail_{tag}.npz"))
    return d["recon"], d["orig"], d["psnr"], float(d["esno"])


def main(ncols=8):
    o_rec, o_org, o_ps, o_es = load("ours")
    b_rec, b_org, b_ps, b_es = load("base")

    fig, axes = plt.subplots(4, ncols, figsize=(ncols * 1.3, 5.6))
    rows = [(o_org, "ours orig", o_es), (o_rec, f"ours recon (PSNR)", o_es),
            (b_org, "base orig", b_es), (b_rec, "base recon (PSNR)", b_es)]
    ps_rows = [None, o_ps, None, b_ps]
    for r, (imgs, label, es) in enumerate(rows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if ps_rows[r] is not None:
                ax.set_title(f"{ps_rows[r][c]:.1f}dB", fontsize=7)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle(
        f"Failure quality on CRC-failed blocks — "
        f"ours(legacy)@Es/N0={o_es}dB PSNR {o_ps.mean():.1f}dB (graceful)  vs  "
        f"base(MAX)@Es/N0={b_es}dB PSNR {b_ps.mean():.1f}dB (catastrophic)",
        fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESULTS, "failure_montage.png")
    fig.savefig(out, dpi=120)
    print("saved", out)
    print(f"ours  PSNR: mean={o_ps.mean():.2f} median={np.median(o_ps):.2f} "
          f"min={o_ps.min():.2f} max={o_ps.max():.2f} (n={len(o_ps)}, Es/N0={o_es})")
    print(f"base  PSNR: mean={b_ps.mean():.2f} median={np.median(b_ps):.2f} "
          f"min={b_ps.min():.2f} max={b_ps.max():.2f} (n={len(b_ps)}, Es/N0={b_es})")


if __name__ == "__main__":
    main()
