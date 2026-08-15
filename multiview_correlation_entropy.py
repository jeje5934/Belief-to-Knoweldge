#!/usr/bin/env python3
"""Empirical correlation budget for the synthetic Fashion-MNIST views.

This is an interpretation diagnostic, not a source-coding theorem.  It reports
the pixelwise plug-in H(X|Y) and I(X;Y), a residual-code upper-bound proxy, and
bit-plane conditional quantities.  Oracle registration is used to isolate the
available source correlation; receiver registration loss is evaluated in the
separate decoding experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import FashionMNIST

from multiview_gate import BPP, PIXELS, STRENGTHS, make_views


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    probabilities = counts[counts > 0] / np.sum(counts)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def joint_metrics(x_u8, y_u8):
    x = x_u8.reshape(-1).astype(np.int64)
    y = y_u8.reshape(-1).astype(np.int64)
    joint = np.bincount(x * 256 + y, minlength=256 * 256).reshape(256, 256)
    hx = entropy_from_counts(np.sum(joint, axis=1))
    hy = entropy_from_counts(np.sum(joint, axis=0))
    hxy = entropy_from_counts(joint)
    h_x_given_y = hxy - hy
    mutual = hx + hy - hxy
    residual = x - y
    residual_counts = np.bincount(residual + 255, minlength=511)
    bit_rows = []
    x_bits = np.unpackbits(x_u8[..., None], axis=-1)
    y_bits = np.unpackbits(y_u8[..., None], axis=-1)
    for plane in range(BPP):
        xb = x_bits[..., plane].reshape(-1).astype(np.int64)
        yb = y_bits[..., plane].reshape(-1).astype(np.int64)
        bit_joint = np.bincount(xb * 2 + yb, minlength=4).reshape(2, 2)
        hxb = entropy_from_counts(np.sum(bit_joint, axis=1))
        hyb = entropy_from_counts(np.sum(bit_joint, axis=0))
        hxyb = entropy_from_counts(bit_joint)
        bit_rows.append({
            "plane_msb0": plane,
            "agreement": float(np.mean(xb == yb)),
            "h_x": hxb,
            "h_x_given_y": hxyb - hyb,
            "mutual_information": hxb + hyb - hxyb,
        })
    return {
        "h_x_empirical_pixel_bpp": hx,
        "h_y_empirical_pixel_bpp": hy,
        "h_x_given_y_pixelwise_bpp": h_x_given_y,
        "i_x_y_pixelwise_bpp": mutual,
        "fraction_of_empirical_hx": mutual / hx,
        "residual_entropy_bpp_upper_bound_proxy": entropy_from_counts(residual_counts),
        "byte_exact_rate": float(np.mean(x == y)),
        "bit_agreement": float(np.mean(x_bits == y_bits)),
        "bit_planes": bit_rows,
        "sum_bitplane_mutual_information": float(
            sum(row["mutual_information"] for row in bit_rows)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=20263201)
    parser.add_argument("--stream-root", default=(
        "/home/LJH/onlyextrinsic_ada_sigma/compression_baseline/results"
    ))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = FashionMNIST(root="/tmp/fmnist", train=False, download=False)
    x_u8 = dataset.data.numpy()[:args.blocks].astype(np.uint8)
    x = x_u8.astype(np.float32) / 255.0
    pixelcnn_path = Path(args.stream_root) / "channel_streams.npz"
    with np.load(pixelcnn_path) as archive:
        lengths = np.asarray(archive["lengths"][:args.blocks], dtype=np.float64)
    pixelcnn_bpp = float(np.mean(lengths) / PIXELS)

    results = {}
    for strength_index, strength in enumerate(STRENGTHS[:3]):
        _, aligned, _ = make_views(
            x, strength, args.seed + strength_index * 100000
        )
        y_u8 = np.rint(np.clip(aligned, 0.0, 1.0) * 255.0).astype(np.uint8)
        results[strength.name] = joint_metrics(x_u8, y_u8)

    payload = {
        "kind": "synthetic_multiview_correlation_entropy",
        "blocks": args.blocks,
        "pixels_per_block": PIXELS,
        "registration": "oracle inverse affine/brightness for an intrinsic-correlation upper bound",
        "estimator": (
            "plug-in 256x256 pixel joint histogram; spatial dependence is ignored, "
            "so this is not an operational whole-image entropy rate"
        ),
        "pixelcnn_operational_hx_proxy_bpp": pixelcnn_bpp,
        "pixelcnn_note": (
            "mean arithmetic-stream length / 784; reported beside, not subtracted "
            "from, the pixelwise conditional entropy because the estimators differ"
        ),
        "strengths": results,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
