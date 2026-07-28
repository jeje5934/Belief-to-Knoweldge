"""Reproducible architectural depth and rough work profile.

The work comparison is intentionally approximate.  It counts dense learned
multiply-accumulates (MACs) in one SongUNet forward and compares them with the
number of LDPC edge-message updates in one flooding BP iteration.  The latency
model itself uses critical-path depth, not this work ratio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from denoiser_sigma_conditional_diag import make_system
from source_prior import SourcePriorDenoiser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    denoiser = SourcePriorDenoiser().eval()
    model = denoiser.net.model
    macs = {"conv": 0, "linear": 0}
    handles = []

    def hook(module, inputs, output):
        if not hasattr(module, "weight") or module.weight is None:
            return
        weight = module.weight
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            return
        if weight.ndim == 4 and tensor.ndim == 4:
            # Output elements times input-channel/kernel products.
            per_output = int(np.prod(weight.shape[1:]))
            macs["conv"] += int(tensor.numel() * per_output)
        elif weight.ndim == 2 and tensor.ndim >= 2:
            macs["linear"] += int(tensor.numel() * weight.shape[1])

    for module in model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim in (2, 4):
            handles.append(module.register_forward_hook(hook))

    with torch.inference_mode():
        denoiser.net(
            torch.zeros(1, 1, 28, 28),
            torch.tensor([0.3]),
        )
    for handle in handles:
        handle.remove()

    unet_blocks = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "UNetBlock"
    ]
    attention_blocks = [
        module for module in unet_blocks if int(module.num_heads) > 0
    ]
    # Noise embedding's two linear layers and the image input convolution run
    # in parallel before the first block.  Each UNetBlock has two serial spatial
    # convolutions; attention blocks add qkv/projection stages.  The final image
    # projection contributes one more stage.
    critical_depth = (
        max(2, 1)
        + 2 * len(unet_blocks)
        + 2 * len(attention_blocks)
        + 1
    )

    _, _, ldpc, _, _, _ = make_system()
    pcm = ldpc.pcm
    edges = int(pcm.nnz if hasattr(pcm, "nnz") else np.count_nonzero(pcm))
    edge_updates_per_iteration = 2 * edges
    total_macs = int(macs["conv"] + macs["linear"])

    pixelcnn_layers = 12
    pixelcnn_depth_per_symbol = 1 + 3 * pixelcnn_layers + 2
    result = {
        "songunet": {
            "model_channels": 64,
            "channel_mult": [1, 2, 2],
            "num_blocks": 2,
            "unet_blocks": len(unet_blocks),
            "attention_blocks": len(attention_blocks),
            "critical_path_stage_estimate": critical_depth,
            "macs_per_forward_batch1": total_macs,
            "conv_macs": macs["conv"],
            "linear_macs": macs["linear"],
        },
        "ldpc": {
            "pcm_shape": list(pcm.shape),
            "edges": edges,
            "edge_message_updates_per_bp_iteration": edge_updates_per_iteration,
            "mac_to_edge_update_iteration_ratio": (
                total_macs / edge_updates_per_iteration
            ),
            "warning": (
                "work-equivalent lower-bound proxy only; a BP edge update is "
                "not exactly one dense MAC"
            ),
        },
        "pixelcnn": {
            "pixels_sequential": 784,
            "gated_layers": pixelcnn_layers,
            "critical_path_stages_per_pixel": pixelcnn_depth_per_symbol,
            "encoder_depth_units": 784 * pixelcnn_depth_per_symbol,
            "optimistic_symbol_only_lower_bound": 784,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
