#!/usr/bin/env python3
"""Retraining-free per-pixel cavity/source variance study.

Production decoder.py, denoiser.py and source_prior.py stay untouched.  A
temporary denoiser wrapper evaluates:

* cavity variance: Var[pixel]=sum w_i^2 p_i(1-p_i), then a spatial sigma map;
  the scalar-conditioned EDM is evaluated on a small sigma grid and its output
  is interpolated pixel-by-pixel (a retraining-free upper-bound adapter);
* source variance: diagonal Tweedie covariance sigma^2 diag(dD/dx), estimated
  with deterministic Hutchinson finite differences, then used as the existing
  per-pixel posterior_pixel_std readout slot.

All candidates share payloads/noise and are judged by hard CRC.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import torch

from crc_utils import hard_crc_decode
from denoiser_sigma_alignment_diag import K_PAYLOAD, load_fashion_mnist
from denoiser_sigma_conditional_diag import make_channel_batch, make_system, wilson
from denoiser_sigma_three_scheme import DEFAULT_CHECKPOINT, build_decoder, configure_candidate
from low_budget_source_study import exact_paired_binomial, geom_scheduler


SCHEDULE = [5] * 20
SOURCE_CALLS = len(SCHEDULE) - 1
SIGMA_ENDPOINT = 0.05
DEFAULT_SIGMA_GRID = (0.001, 0.003, 0.01, 0.03, 0.10, 0.30)


def dump_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")


def describe(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q10": float(np.quantile(x, 0.10)),
        "median": float(np.median(x)),
        "q90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
    }


def paired_summary(success, reference=None):
    success = np.asarray(success, dtype=bool)
    failures = int(np.sum(~success))
    row = {
        "blocks": int(len(success)),
        "failures": failures,
        "crc_bler": float(failures / len(success)),
        "wilson_95": wilson(failures, len(success)),
        "success_mask": success.tolist(),
    }
    if reference is not None:
        reference = np.asarray(reference, dtype=bool)
        broken = int(np.sum(reference & ~success))
        rescued = int(np.sum(~reference & success))
        row["paired_vs_constant"] = {
            "broken": broken,
            "rescued": rescued,
            "net_failure_change": broken - rescued,
            "exact_two_sided_p": exact_paired_binomial(broken, rescued),
        }
    return row


def parse_candidate(spec):
    fields = spec.split(":")
    mode = fields[0]
    if mode == "const" and len(fields) == 1:
        return {"name": "const", "mode": "constant", "c": 1.0, "cp": 1.0}
    if mode == "cav" and len(fields) == 2:
        c = float(fields[1])
        return {"name": f"cav_c{c:g}", "mode": "cavity", "c": c, "cp": 1.0}
    if mode == "src" and len(fields) == 2:
        cp = float(fields[1])
        return {"name": f"src_cp{cp:g}", "mode": "source", "c": 1.0, "cp": cp}
    if mode == "both" and len(fields) == 3:
        c, cp = float(fields[1]), float(fields[2])
        return {"name": f"both_c{c:g}_cp{cp:g}", "mode": "both", "c": c, "cp": cp}
    raise ValueError(f"invalid candidate spec: {spec}")


class SiteVarianceProxy(tf.keras.layers.Layer):
    """Experiment-only per-pixel moment adapter around an unchanged denoiser."""

    def __init__(self, inner, sigma_grid, probes=4, eps=0.02):
        super().__init__(name="site_variance_proxy")
        self.inner = inner
        self.sigma_grid = tuple(float(x) for x in sigma_grid)
        self.probes = int(probes)
        self.eps = float(eps)
        self.override = False
        self.path = ()
        self.call_index = 0
        self.mode = "constant"
        self.c = 1.0
        self.cp = 1.0
        self.probe_seed = 0
        self.diag_rows = []
        self.network_forwards = 0

    @property
    def sigma_post(self):
        return self.inner.sigma_post

    def reset(self, override, path):
        self.override = bool(override)
        self.path = tuple(float(x) for x in path)
        self.call_index = 0

    def set_variance(self, mode, c, cp, probe_seed):
        self.mode = mode
        self.c = float(c)
        self.cp = float(cp)
        self.probe_seed = int(probe_seed)
        self.diag_rows = []
        self.network_forwards = 0

    def _sigma_vector(self, sigma, batch, device):
        if sigma is None:
            value = self.inner.sigma
        elif isinstance(sigma, tf.Tensor):
            value = sigma.numpy()
        else:
            value = sigma
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(batch)
        if tensor.numel() != batch:
            raise ValueError("sigma must be scalar or one value per block")
        return tensor

    def _net(self, mu, sigma):
        self.network_forwards += 1
        return self.inner.prior_model.net(mu, sigma).clamp(0.0, 1.0)

    def _spatial_projection(self, mu, target_sigma):
        """Interpolate scalar-conditioned EDM outputs at each pixel's sigma."""
        device, dtype = mu.device, mu.dtype
        grid = torch.as_tensor(self.sigma_grid, device=device, dtype=dtype)
        outputs = []
        for value in grid:
            sigma = torch.full((mu.shape[0],), float(value), device=device, dtype=dtype)
            outputs.append(self._net(mu, sigma))
        target = target_sigma.clamp(float(grid[0]), float(grid[-1]))
        log_grid = torch.log(grid)
        log_target = torch.log(target)
        result = outputs[-1].clone()
        for index in range(len(outputs) - 1):
            lo, hi = log_grid[index], log_grid[index + 1]
            if index == len(outputs) - 2:
                mask = (log_target >= lo) & (log_target <= hi)
            else:
                mask = (log_target >= lo) & (log_target < hi)
            weight = ((log_target - lo) / (hi - lo)).clamp(0.0, 1.0)
            blended = (1.0 - weight) * outputs[index] + weight * outputs[index + 1]
            result = torch.where(mask, blended, result)
        result = torch.where(target <= grid[0], outputs[0], result)
        return result, target

    def _tweedie_std(self, mu, mu_proj, effective_sigma, spatial):
        """Diagonal Tweedie std, matching the prior sibling-branch estimator."""
        generator = torch.Generator(device=mu.device)
        generator.manual_seed(self.probe_seed + 1009 * self.call_index)
        diag = torch.zeros_like(mu)
        for _ in range(self.probes):
            bits = torch.randint(
                0, 2, mu.shape, generator=generator, device=mu.device,
                dtype=torch.int64,
            )
            direction = bits.to(mu.dtype) * 2.0 - 1.0
            perturbed = mu + self.eps * direction
            if spatial:
                projected, _ = self._spatial_projection(perturbed, effective_sigma)
            else:
                sigma_vec = effective_sigma.reshape(mu.shape[0])
                projected = self._net(perturbed, sigma_vec)
            diag += direction * (projected - mu_proj) / self.eps
        diag = (diag / self.probes).clamp(0.0, 1.0)
        variance = effective_sigma.pow(2) * diag
        raw_std = 255.0 * torch.sqrt(variance.clamp(min=1.0e-12))
        raw_std = raw_std.reshape(mu.shape[0], -1).clamp(0.5, 128.0)
        scaled = (self.cp * raw_std).clamp(0.25, 128.0)
        return scaled, raw_std, diag

    def call(self, input_llr, sigma=None):
        if self.override:
            if self.call_index >= len(self.path):
                raise RuntimeError("altproj consumed too many source calls")
            sigma = self.path[self.call_index]

        if self.mode == "constant":
            self.call_index += 1
            self.network_forwards += 1
            return self.inner(input_llr, sigma=sigma)

        llr = torch.from_numpy(input_llr.numpy()).float().to(self.inner._device)
        prior = self.inner.prior_model
        sigma_vec = self._sigma_vector(sigma, llr.shape[0], llr.device)
        with torch.no_grad():
            mu = prior.llr_to_soft_field(llr)
            use_cavity = self.mode in ("cavity", "both")
            use_source = self.mode in ("source", "both")
            if use_cavity:
                cavity_std = torch.sqrt(prior.pixel_variance(llr).clamp(min=1.0e-8))
                target_sigma = (self.c * cavity_std / 255.0).reshape(
                    llr.shape[0], 1, prior.img_h, prior.img_w
                )
                mu_proj, effective_sigma = self._spatial_projection(mu, target_sigma)
            else:
                mu_proj = self._net(mu, sigma_vec)
                effective_sigma = sigma_vec.reshape(-1, 1, 1, 1)

            posterior_std = None
            raw_std = None
            jac_diag = None
            if use_source:
                posterior_std, raw_std, jac_diag = self._tweedie_std(
                    mu, mu_proj, effective_sigma, use_cavity
                )
            projected_llr = prior.soft_field_to_posterior_logits(
                mu_proj, posterior_pixel_std=posterior_std
            )
            extrinsic = projected_llr - llr

            row = {"source_call": self.call_index + 1, "mode": self.mode}
            if use_cavity:
                target_np = effective_sigma.detach().cpu().numpy()
                row["cavity_sigma"] = describe(target_np)
                row["cavity_sigma_clipped_low_fraction"] = float(
                    np.mean(target_np <= self.sigma_grid[0] * (1.0 + 1e-6))
                )
                row["cavity_sigma_clipped_high_fraction"] = float(
                    np.mean(target_np >= self.sigma_grid[-1] * (1.0 - 1e-6))
                )
            if use_source:
                row["source_std_raw_255"] = describe(raw_std.detach().cpu().numpy())
                row["source_std_scaled_255"] = describe(
                    posterior_std.detach().cpu().numpy()
                )
                row["jacobian_floor_fraction"] = float(
                    torch.mean((jac_diag <= 0.0).float()).item()
                )
            self.diag_rows.append(row)

        self.call_index += 1
        return tf.constant(extrinsic.cpu().numpy(), dtype=input_llr.dtype)


def configure_decoder(decoder, proxy, scheduler, path, scheme, candidate, probe_seed, ep_alpha):
    if scheme == "altproj":
        base = {"name": "altproj", "scheme": "altproj", "delta": 0.02, "rho": 0.95}
    elif scheme == "ep":
        base = {"name": "ep", "scheme": "ep", "alpha_ep": float(ep_alpha)}
    else:
        raise ValueError(scheme)
    configure_candidate(decoder, proxy, scheduler, path, base)
    decoder.altproj_early_stop = False
    decoder.track_u_hat = False
    proxy.set_variance(
        candidate["mode"], candidate["c"], candidate["cp"], probe_seed
    )


def average_diag(diag_batches):
    if not diag_batches:
        return []
    calls = max((len(rows) for rows in diag_batches), default=0)
    result = []
    for call in range(calls):
        rows = [batch[call] for batch in diag_batches if call < len(batch)]
        merged = {"source_call": call + 1, "batches": len(rows)}
        for prefix in ("cavity_sigma", "source_std_raw_255", "source_std_scaled_255"):
            available = [row[prefix] for row in rows if prefix in row]
            if available:
                merged[prefix] = {
                    key: float(np.mean([item[key] for item in available]))
                    for key in available[0]
                }
        for key in (
            "cavity_sigma_clipped_low_fraction",
            "cavity_sigma_clipped_high_fraction",
            "jacobian_floor_fraction",
        ):
            values = [row[key] for row in rows if key in row]
            if values:
                merged[key] = float(np.mean(values))
        result.append(merged)
    return result


def run_point(args, snr, blocks, point_seed, candidates):
    if blocks % args.batch:
        raise ValueError("blocks must be divisible by batch")
    crc_encoder, crc_decoder, ldpc, mapper, demapper, awgn = make_system()
    images, bit_bank = load_fashion_mnist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder, old_proxy, _, _ = build_decoder(ldpc, args.checkpoint, SCHEDULE, device)
    scheduler, path = geom_scheduler(SCHEDULE, SIGMA_ENDPOINT)
    proxy = SiteVarianceProxy(
        old_proxy.inner,
        sigma_grid=args.sigma_grid,
        probes=args.probes,
        eps=args.probe_eps,
    )
    decoder._denoiser = proxy
    decoder._ep_config_logged = True

    masks = {candidate["name"]: [] for candidate in candidates}
    bit_errors = {candidate["name"]: 0 for candidate in candidates}
    diag = {candidate["name"]: [] for candidate in candidates}
    forwards = {candidate["name"]: 0 for candidate in candidates}
    runtimes = {candidate["name"]: 0.0 for candidate in candidates}

    rounds = blocks // args.batch
    for round_index in range(rounds):
        _, payload, _, llr = make_channel_batch(
            round_index, args.batch, point_seed, snr, images, bit_bank,
            crc_encoder, ldpc, mapper, demapper, awgn,
        )
        truth = np.asarray(payload.numpy()).astype(bool)
        for candidate_index, candidate in enumerate(candidates):
            configure_decoder(
                decoder, proxy, scheduler, path, args.scheme, candidate,
                point_seed + round_index * 10000, args.ep_alpha,
            )
            started = time.perf_counter()
            logits = decoder(llr)
            runtimes[candidate["name"]] += time.perf_counter() - started
            _, valid = hard_crc_decode(crc_decoder, logits)
            valid_np = np.asarray(valid.numpy()).reshape(-1).astype(bool)
            masks[candidate["name"]].extend(valid_np.tolist())
            hard = np.asarray(logits.numpy())[:, :K_PAYLOAD] > 0.0
            bit_errors[candidate["name"]] += int(np.sum(hard != truth))
            diag[candidate["name"]].append(list(proxy.diag_rows))
            forwards[candidate["name"]] += int(proxy.network_forwards)
            if proxy.call_index != SOURCE_CALLS:
                raise RuntimeError(
                    f"{candidate['name']} source calls={proxy.call_index}, expected={SOURCE_CALLS}"
                )
        print(
            f"variance {args.scheme} Es/N0={snr:+.2f} "
            f"round {round_index + 1}/{rounds}", flush=True,
        )

    reference = np.asarray(masks[candidates[0]["name"]], dtype=bool)
    rows = {}
    for index, candidate in enumerate(candidates):
        name = candidate["name"]
        row = paired_summary(
            masks[name], None if index == 0 else reference
        )
        row.update({
            "candidate": candidate,
            "payload_ber": float(bit_errors[name] / (blocks * K_PAYLOAD)),
            "runtime_seconds": runtimes[name],
            "network_forwards": forwards[name],
            "mean_network_forwards_per_decode": forwards[name] / rounds,
            "variance_trajectory": average_diag(diag[name]),
        })
        rows[name] = row
    return rows, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=("altproj", "ep"), default="altproj")
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--snrs", type=float, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, nargs="+", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20263301)
    parser.add_argument("--ep-alpha", type=float, default=0.01)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--probe-eps", type=float, default=0.02)
    parser.add_argument("--sigma-grid", type=float, nargs="+", default=DEFAULT_SIGMA_GRID)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.snrs) != len(args.blocks):
        raise ValueError("--snrs and --blocks lengths differ")
    candidates = [parse_candidate(value) for value in args.candidate]
    if candidates[0]["mode"] != "constant":
        raise ValueError("first candidate must be const for paired accounting")

    result = {
        "kind": "single_terminal_pixelwise_site_variance",
        "configuration": {
            "branch": "practical_sigma",
            "scheme": args.scheme,
            "schedule": SCHEDULE,
            "source_calls": SOURCE_CALLS,
            "altproj": {"delta": 0.02, "rho": 0.95},
            "ep": {"update": "damped_ep", "alpha_ep": args.ep_alpha, "beta_ep": 1.0},
            "global_sigma_schedule": [0.3, SIGMA_ENDPOINT, "geometric"],
            "fixed_sigma_post": 3.0,
            "sigma_grid": args.sigma_grid,
            "tweedie_probes": args.probes,
            "tweedie_eps": args.probe_eps,
            "hard_crc": True,
            "channel": "BPSK/AWGN/perfect_CSI",
            "payload_bits": K_PAYLOAD,
            "n": 12600,
            "production_decoder_modified": False,
            "spatial_sigma_adapter": (
                "pixelwise interpolation of scalar-conditioned EDM outputs; "
                "retraining-free upper-bound approximation"
            ),
        },
        "candidates": candidates,
        "points": {},
    }
    for index, (snr, blocks) in enumerate(zip(args.snrs, args.blocks)):
        point_seed = args.seed + index * 10000
        rows, path = run_point(args, snr, blocks, point_seed, candidates)
        result["sigma_path"] = list(path)
        result["points"][f"{snr:.3f}"] = {
            "esn0_db": snr,
            "blocks": blocks,
            "seed": point_seed,
            "rows": rows,
        }
        dump_json(args.output, result)
        print(f"saved {snr:+.2f} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
