"""Source-decoder points for the abstract latency-matching study.

This is an experiment harness.  Production ``decoder.py`` is imported without
modification.  It records both first-CRC BP iterations and the number of source
calls that lie on the corresponding serial critical path.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

import numpy as np

from denoiser_sigma_three_scheme import (
    DEFAULT_CHECKPOINT,
    EP_BETA_CODE_SITE,
    LEGACY_BETA_BP_EXTRINSIC_BOOST,
    dump_json,
)
from low_budget_source_study import (
    make_result_row,
    readout_from_a,
    run_masks_for_configs,
)


def named_config(name, scheme, sigma_end, **kwargs):
    candidate = {"name": name, "scheme": scheme, **kwargs}
    return {
        "name": f"{name}_send{sigma_end:g}",
        "candidate": candidate,
        "sigma_end": float(sigma_end),
    }


def experiment_groups():
    """Ordered latency points, including a focused low-latency EP search."""
    groups = OrderedDict()
    groups["bp10"] = {
        "schedule": [10],
        "configs": [
            named_config(
                "bp10", "altproj", 0.3, delta=0.0, rho=0.9
            )
        ],
    }

    ten_by_two = []
    for endpoint in (0.05, 0.02):
        for delta in (0.05, 0.1, 0.2):
            ten_by_two.append(
                named_config(
                    f"altproj_10x2_d{delta:g}_r0.9",
                    "altproj",
                    endpoint,
                    delta=delta,
                    rho=0.9,
                )
            )
        for alpha in (0.05, 0.1, 0.2):
            ten_by_two.append(
                named_config(
                    f"ep_10x2_a{alpha:g}",
                    "ep",
                    endpoint,
                    alpha_ep=alpha,
                )
            )
    groups["10x2"] = {"schedule": [10, 10], "configs": ten_by_two}

    five_by_four = [
        named_config(
            "altproj_5x4_d0.1_r0.9",
            "altproj",
            0.02,
            delta=0.1,
            rho=0.9,
        )
    ]
    for endpoint in (0.05, 0.02):
        five_by_four.extend(
            [
                named_config(
                    f"altproj_5x4_d0.2_r0.9",
                    "altproj",
                    endpoint,
                    delta=0.2,
                    rho=0.9,
                ),
                named_config(
                    f"ep_5x4_a0.1",
                    "ep",
                    endpoint,
                    alpha_ep=0.1,
                ),
                named_config(
                    f"ep_5x4_a0.2",
                    "ep",
                    endpoint,
                    alpha_ep=0.2,
                ),
            ]
        )
    groups["5x4"] = {"schedule": [5] * 4, "configs": five_by_four}

    groups["5x6"] = {
        "schedule": [5] * 6,
        "configs": [
            named_config(
                "altproj_5x6_d0.1_r0.9",
                "altproj",
                0.05,
                delta=0.1,
                rho=0.9,
            )
        ],
    }
    groups["5x10"] = {
        "schedule": [5] * 10,
        "configs": [
            named_config(
                "altproj_5x10_d0.05_r0.9",
                "altproj",
                0.02,
                delta=0.05,
                rho=0.9,
            )
        ],
    }
    groups["5x20"] = {
        "schedule": [5] * 20,
        "configs": [
            named_config(
                "altproj_5x20_d0.02_r0.95",
                "altproj",
                0.02,
                delta=0.02,
                rho=0.95,
            )
        ],
    }
    return groups


def source_calls_from_iterations(iterations, schedule):
    """Calls completed before the first CRC pass (or final failure)."""
    cumulative = np.cumsum(schedule)
    values = np.asarray(iterations, dtype=np.float64)
    return np.asarray(
        [np.searchsorted(cumulative, value, side="left") for value in values],
        dtype=np.float64,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esn0", type=float, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, default=512)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--a-result")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.blocks % args.batch:
        raise ValueError("blocks must be divisible by batch")

    groups = experiment_groups()
    readout = readout_from_a(args.a_result)
    result = {
        "kind": "latency_matching_source_sweep",
        "configuration": {
            "channel": "BPSK/AWGN/perfect_CSI",
            "payload_bits": 6272,
            "n": 12600,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "readout": readout,
            "legacy_beta": LEGACY_BETA_BP_EXTRINSIC_BOOST,
            "ep_beta_ep": EP_BETA_CODE_SITE,
            "crc_early_stop": "first CRC pass from existing chunk history",
            "source_call_count": (
                "number of inter-chunk denoiser calls completed before the "
                "first CRC pass; a final failed block uses outer_count-1"
            ),
        },
        "points": {},
    }

    for point_index, esn0_db in enumerate(args.esn0):
        point = {"esn0_db": float(esn0_db), "groups": {}}
        for group_index, (group_name, group) in enumerate(groups.items()):
            schedule = group["schedule"]
            configs = group["configs"]
            masks, iterations = run_masks_for_configs(
                configs=configs,
                schedule=schedule,
                readout=readout,
                esn0_db=esn0_db,
                blocks=args.blocks,
                batch=args.batch,
                # Same seed across groups preserves image/noise pairing.
                seed=args.seed + point_index * 10000,
                checkpoint=args.checkpoint,
            )
            rows = {}
            for config in configs:
                name = config["name"]
                row = make_result_row(
                    masks[name], iterations[name], config
                )
                calls = source_calls_from_iterations(
                    iterations[name], schedule
                )
                row["actual_mean_source_calls"] = float(np.mean(calls))
                row["actual_source_call_distribution"] = {
                    "median": float(np.median(calls)),
                    "q10": float(np.quantile(calls, 0.1)),
                    "q90": float(np.quantile(calls, 0.9)),
                }
                row.pop("success_mask")
                rows[name] = row
            best_name = min(
                rows,
                key=lambda name: (
                    rows[name]["failures"],
                    rows[name]["actual_mean_source_calls"],
                    rows[name]["actual_mean_bp_iters"],
                ),
            )
            point["groups"][group_name] = {
                "schedule": schedule,
                "nominal_bp_budget": int(sum(schedule)),
                "rows": rows,
                "selected": best_name,
            }
            dump_json(args.output, result | {
                "points": result["points"] | {
                    f"{esn0_db:.3f}": point
                }
            })
        result["points"][f"{esn0_db:.3f}"] = point
        dump_json(args.output, result)


if __name__ == "__main__":
    main()
