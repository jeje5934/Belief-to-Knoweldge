"""Hard-CRC robustness checks for the budget-aware AltProj LUT.

This is an experiment harness only; production ``decoder.py`` is imported and
left unchanged.  The SNR screen is deliberately one-factor-at-a-time around
the hard-CRC LUT so each neighboring comparison has an interpretable cause and
the tuning audit does not grow into an unnecessary 3-D retuning campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from denoiser_sigma_three_scheme import DEFAULT_CHECKPOINT, dump_json
from low_budget_source_study import (
    SCHEDULES,
    make_result_row,
    readout_from_a,
    run_masks_for_configs,
)


HARD_CRC_LUT = {
    100: {"schedule": "5x20", "delta": 0.02, "rho": 0.95, "sigma_end": 0.05},
    50: {"schedule": "5x10", "delta": 0.05, "rho": 0.90, "sigma_end": 0.05},
    20: {"schedule": "5x4", "delta": 0.10, "rho": 0.90, "sigma_end": 0.05},
}


def overlap(left, right):
    return max(left[0], right[0]) <= min(left[1], right[1])


def runner_config(name, delta, rho, sigma_end, variation):
    return {
        "name": name,
        "candidate": {
            "name": f"altproj_d{delta:g}_r{rho:g}",
            "scheme": "altproj",
            "delta": float(delta),
            "rho": float(rho),
        },
        "sigma_end": float(sigma_end),
        "variation": variation,
    }


def base_config(budget, rho_override=None):
    item = HARD_CRC_LUT[budget]
    rho = item["rho"] if rho_override is None else float(rho_override)
    return runner_config(
        "base",
        item["delta"],
        rho,
        item["sigma_end"],
        {"axis": "base", "value": None},
    )


def oat_configs(budget):
    item = HARD_CRC_LUT[budget]
    delta = item["delta"]
    rho = item["rho"]
    endpoint = item["sigma_end"]
    configs = [base_config(budget)]
    for value in (delta / 2.0, delta * 2.0):
        configs.append(
            runner_config(
                f"delta_{value:g}",
                value,
                rho,
                endpoint,
                {"axis": "delta", "value": value},
            )
        )
    for value in (0.02, 0.10):
        configs.append(
            runner_config(
                f"sigma_end_{value:g}",
                delta,
                rho,
                value,
                {"axis": "sigma_end", "value": value},
            )
        )
    for value in (rho - 0.05, rho + 0.05):
        configs.append(
            runner_config(
                f"rho_{value:g}",
                delta,
                value,
                endpoint,
                {"axis": "rho", "value": value},
            )
        )
    return configs


def summarize(rows):
    best_name = min(
        rows,
        key=lambda name: (
            rows[name]["failures"],
            rows[name]["actual_mean_bp_iters"],
        ),
    )
    base = rows["base"]
    best = rows[best_name]
    axis_best = {}
    for axis in ("delta", "sigma_end", "rho"):
        names = [
            "base",
            *[
                name
                for name, row in rows.items()
                if row["config"]["variation"]["axis"] == axis
            ],
        ]
        chosen = min(
            names,
            key=lambda name: (
                rows[name]["failures"],
                rows[name]["actual_mean_bp_iters"],
            ),
        )
        axis_best[axis] = chosen
    return {
        "best": best_name,
        "base_is_failure_minimum": base["failures"]
        == min(row["failures"] for row in rows.values()),
        "base_ci_overlaps_best": overlap(base["wilson_95"], best["wilson_95"]),
        "axis_best": axis_best,
    }


def run(args):
    budgets = [int(value) for value in args.budgets.split(",") if value]
    unknown = set(budgets) - set(HARD_CRC_LUT)
    if unknown:
        raise ValueError(f"unsupported budgets: {sorted(unknown)}")
    if args.blocks % args.batch:
        raise ValueError("blocks must be divisible by batch")
    readout = readout_from_a(args.a_result)
    initial = {
        "kind": (
            "hard_crc_lut_oat_snr_robustness"
            if args.mode == "screen"
            else "hard_crc_lut_independent_seed_reproduction"
        ),
        "configuration": {
            "channel": "BPSK/AWGN/perfect_CSI",
            "crc_input": "hard decision logit>0 -> bit 1",
            "budgets": budgets,
            "esn0_db": args.esn0,
            "blocks": args.blocks,
            "batch": args.batch,
            "seed": args.seed,
            "readout": readout,
            "hard_crc_lut": HARD_CRC_LUT,
            "rho_override": args.rho_override,
            "screen_design": (
                "one-factor-at-a-time adjacent settings"
                if args.mode == "screen"
                else "fixed hard-CRC LUT only"
            ),
        },
        "budgets": {},
    }
    result = (
        json.loads(Path(args.output).read_text(encoding="utf-8"))
        if args.resume and Path(args.output).exists()
        else initial
    )
    for budget in budgets:
        budget_key = str(budget)
        budget_result = result["budgets"].setdefault(budget_key, {"points": {}})
        lut = HARD_CRC_LUT[budget]
        schedule = SCHEDULES[budget][lut["schedule"]]
        configs = (
            oat_configs(budget)
            if args.mode == "screen"
            else [base_config(budget, args.rho_override)]
        )
        for point_index, esn0_db in enumerate(args.esn0):
            point_key = f"{esn0_db:.3f}"
            if point_key in budget_result["points"]:
                continue
            masks, iterations = run_masks_for_configs(
                configs=configs,
                schedule=schedule,
                readout=readout,
                esn0_db=esn0_db,
                blocks=args.blocks,
                batch=args.batch,
                seed=args.seed + point_index * 10000,
                checkpoint=args.checkpoint,
            )
            base_mask = masks["base"]
            rows = {}
            for config in configs:
                name = config["name"]
                row = make_result_row(
                    masks[name],
                    iterations[name],
                    config,
                    None if name == "base" else base_mask,
                )
                row["config"]["variation"] = config["variation"]
                rows[name] = row
            budget_result["points"][point_key] = {
                "esn0_db": float(esn0_db),
                "schedule": schedule,
                "rows": rows,
                "summary": summarize(rows),
            }
            dump_json(args.output, result)
    dump_json(args.output, result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("screen", "reproduce"))
    parser.add_argument("--budgets", default="100,50")
    parser.add_argument("--esn0", type=float, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, default=512)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument(
        "--rho-override",
        type=float,
        help="reproduce mode only: evaluate the robustness-selected rho",
    )
    parser.add_argument("--a-result")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
