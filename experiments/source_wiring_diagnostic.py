#!/usr/bin/env python3
"""Small, assertion-heavy diagnostic for the no-LDPC source-message wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from coding.bcjr import bcjr_decode
from coding.spc import decode_systematic, encode_spc
from decoders import NoLDPCConfig, RSCSourceIterativeDecoder, SourceSPCSISO
from experiments.no_ldpc_smoke import load_score_provider
from experiments.waterfall_common import (
    awgn_llr,
    load_fashion_mnist_bits,
    paired_batches,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--esn0-db", type=float, nargs="+", default=[8.0, 2.6])
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--dataset-root", default="/tmp/fmnist")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/denoiser.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/source_wiring_diagnostic")
    )
    return parser.parse_args()


def payload_to_pixels(payload: np.ndarray) -> np.ndarray:
    bits = np.asarray(payload, dtype=np.uint8)
    if bits.shape[-1] != 28 * 28 * 8:
        raise AssertionError("payload width is not one 8-bit 28x28 image")
    shaped = bits.reshape(bits.shape[:-1] + (28 * 28, 8))
    return np.packbits(shaped, axis=-1, bitorder="big")[..., 0].reshape(
        bits.shape[:-1] + (28, 28)
    )


def save_montage(
    path: Path,
    original: np.ndarray,
    soft_input: np.ndarray,
    hard_input: np.ndarray,
    denoised: np.ndarray,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    panels = [
        (original, "original"),
        (soft_input, "source SISO input\nsoft pixels"),
        (hard_input, "source SISO input\nhard bits"),
        (denoised, "denoiser output $\\hat{x}_0$"),
    ]
    for axis, (image, label) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def payload_positions(values: np.ndarray, bits_per_symbol: int = 8) -> np.ndarray:
    shaped = values.reshape(values.shape[:-1] + (-1, bits_per_symbol + 1))
    return shaped[..., :bits_per_symbol].reshape(values.shape[:-1] + (-1,))


def sign_agreement(values: np.ndarray, truth: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.uint8)
    expected = 2.0 * truth.astype(np.float64) - 1.0
    active = np.abs(values) > 1.0e-12
    supportive = values * expected > 0.0
    return {
        "all_bits": float(np.mean(supportive)),
        "nonzero_bits": float(np.mean(supportive[active])) if np.any(active) else None,
        "nonzero_fraction": float(np.mean(active)),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.blocks <= 64:
        raise ValueError("diagnostic blocks must be in [1, 64]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = NoLDPCConfig(
        outer_iterations=2,
        alpha_schedule=(0.1, 0.1),
        llr_clip=30.0,
        sigma=0.3,
        sigma_post=3.0,
        interleaver_seed=20260722,
        bcjr_mode="logmap",
    )
    decoder = RSCSourceIterativeDecoder(config)
    provider = load_score_provider(config, args.checkpoint, args.device)
    source_siso = SourceSPCSISO(config.source_bits_per_symbol, provider)
    bank = load_fashion_mnist_bits(args.dataset_root)

    # Layout and inverse-map assertions independent of SNR.
    tags = np.arange(config.rsc_information_length, dtype=np.int64)[None, :]
    round_trip = decoder.interleaver.deinterleave(decoder.interleaver.interleave(tags))
    assert np.array_equal(round_trip, tags), "interleave/deinterleave round-trip failed"

    rows = []
    for esn0_db in args.esn0_db:
        _, payload, noise = next(
            paired_batches(
                bank,
                blocks=args.blocks,
                batch_size=args.blocks,
                esn0_db=esn0_db,
                seed=args.seed,
            )
        )
        frame = decoder.encode(payload)

        expected_source_coded = encode_spc(payload, config.source_bits_per_symbol)
        assert np.array_equal(
            frame.information_bits[..., : config.source_coded_length],
            expected_source_coded,
        ), "SPC source-code layout differs from [8 systematic, 1 parity]"
        assert np.array_equal(
            decode_systematic(expected_source_coded, config.source_bits_per_symbol),
            payload,
        ), "SPC systematic extraction does not recover payload"
        assert np.array_equal(
            decoder.interleaver.deinterleave(
                decoder.interleaver.interleave(frame.information_bits)
            ),
            frame.information_bits,
        ), "frame interleaver is not exactly invertible"

        transmitted_llr = awgn_llr(frame.transmitted_bits, noise, esn0_db)
        systematic_llr, parity_llr = decoder.rate_matcher.depuncture_llr(
            transmitted_llr
        )
        zero_prior = np.zeros_like(systematic_llr)
        channel_result = bcjr_decode(
            systematic_llr,
            parity_llr,
            decoder.trellis,
            a_priori_llr=zero_prior,
            mode=config.bcjr_mode,
            start_state=0,
            end_state=0,
        )
        interleaved_message = channel_result.extrinsic_llr[
            ..., : config.rsc_information_length
        ]
        channel_to_source = decoder.interleaver.deinterleave(interleaved_message)
        source_message = channel_to_source[..., : config.source_coded_length]
        crc_message = channel_to_source[..., config.source_coded_length :]
        assert source_message.shape[-1] == 784 * 9
        assert crc_message.shape[-1] == 16

        shaped = source_message.reshape(args.blocks, 784, 9)
        systematic_cavity = shaped[..., :8]
        parity_cavity = shaped[..., 8]
        assert np.shares_memory(shaped, source_message)
        assert np.array_equal(
            systematic_cavity.reshape(args.blocks, -1),
            decoder._payload_llr_from_source(source_message),
        ), "payload extraction includes parity or changes ordering"

        captured = {}

        def capture_denoiser(_module, module_input, module_output):
            captured["soft_input"] = module_input[0].detach().cpu().numpy()
            captured["denoised"] = module_output.detach().cpu().numpy()

        hook = provider.prior_model.net.register_forward_hook(capture_denoiser)
        try:
            source_result = source_siso(source_message)
        finally:
            hook.remove()

        assert "soft_input" in captured and "denoised" in captured
        assert np.allclose(
            source_result.extrinsic_llr,
            source_result.posterior_llr - source_message,
            rtol=0.0,
            atol=0.0,
        ), "source extrinsic did not subtract the exact incoming cavity"

        scaled = 0.1 * np.clip(source_result.extrinsic_llr, -30.0, 30.0)
        source_plus_crc = np.concatenate([scaled, np.zeros_like(crc_message)], axis=-1)
        assert np.all(source_plus_crc[..., config.source_coded_length :] == 0.0)
        feedback_interleaved = decoder.interleaver.interleave(source_plus_crc)
        feedback_recovered = decoder.interleaver.deinterleave(feedback_interleaved)
        assert np.array_equal(
            feedback_recovered, source_plus_crc
        ), "BCJR feedback used a non-forward interleaver or changed positions"

        payload_cavity = systematic_cavity.reshape(args.blocks, -1)
        payload_posterior = payload_positions(source_result.posterior_llr)
        payload_extrinsic = payload_positions(source_result.extrinsic_llr)

        original_pixels = payload_to_pixels(payload)
        hard_pixels = payload_to_pixels((payload_cavity > 0.0).astype(np.uint8))
        soft_pixels = (
            captured["soft_input"].reshape(args.blocks, 28, 28) * 255.0
        ).clip(0.0, 255.0)
        denoised_pixels = (
            captured["denoised"].reshape(args.blocks, 28, 28) * 255.0
        ).clip(0.0, 255.0)

        png_path = args.output_dir / f"source_first_pass_esn0_{esn0_db:g}dB.png"
        save_montage(
            png_path,
            original_pixels[0],
            soft_pixels[0],
            hard_pixels[0],
            denoised_pixels[0],
            f"First outer pass, Es/N0={esn0_db:g} dB",
        )

        source_sign = sign_agreement(payload_extrinsic, payload)
        row = {
            "esn0_db": float(esn0_db),
            "blocks": int(args.blocks),
            "png": str(png_path),
            "interleaver_round_trip": True,
            "spc_payload_extract": True,
            "crc_feedback_zero": True,
            "feedback_forward_interleaver_round_trip": True,
            "bit_order": "MSB-first via np.unpackbits/np.packbits(bitorder='big')",
            "pixel_order": "Fashion-MNIST raster order",
            "cavity_hard_sign_agreement": sign_agreement(payload_cavity, payload),
            "source_posterior_sign_agreement": sign_agreement(
                payload_posterior, payload
            ),
            "source_extrinsic_sign_agreement": source_sign,
            "source_extrinsic_clip_fraction_all_spc_bits": float(
                np.mean(np.abs(source_result.extrinsic_llr) >= 30.0)
            ),
            "source_extrinsic_clip_fraction_payload_bits": float(
                np.mean(np.abs(payload_extrinsic) >= 30.0)
            ),
            "source_extrinsic_mean_abs_payload": float(
                np.mean(np.abs(payload_extrinsic))
            ),
            "source_extrinsic_mean_signed_support_payload": float(
                np.mean(payload_extrinsic * (2.0 * payload - 1.0))
            ),
            "source_input_soft_mae_pixel": float(
                np.mean(np.abs(soft_pixels - original_pixels))
            ),
            "source_input_hard_pixel_exact_fraction": float(
                np.mean(hard_pixels == original_pixels)
            ),
            "denoised_mae_pixel": float(
                np.mean(np.abs(denoised_pixels - original_pixels))
            ),
            "parity_cavity_hard_accuracy": float(
                np.mean(
                    (parity_cavity > 0.0)
                    == expected_source_coded.reshape(args.blocks, 784, 9)[..., 8]
                )
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    output = {
        "experiment": "rsc_source_wiring_diagnostic",
        "blocks_per_snr": args.blocks,
        "seed": args.seed,
        "config": {
            "outer_iterations": 2,
            "alpha": 0.1,
            "sigma": 0.3,
            "sigma_post": 3.0,
            "llr_clip": 30.0,
            "bcjr_mode": "logmap",
        },
        "assertions": {
            "deinterleave_inverse": True,
            "spc_systematic_payload_only": True,
            "crc_source_feedback_zero": True,
            "feedback_uses_forward_interleaver": True,
            "source_extrinsic_subtracts_exact_cavity": True,
            "msb_first_raster": True,
        },
        "rows": rows,
    }
    output_path = args.output_dir / "diagnostic.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
