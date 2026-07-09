"""
practical_sigma [C] — head-to-head of denoiser-σ strategies on EP [5]×20.

Three σ families, all on the SAME EP structure (α_ep=0.01, damped_ep):
  (i)  fixed σ            : baseline (σ=0.3 anchor; σ=0.15 small control)
  (ii) handcrafted adapt. : final-table EP-best config (DEFAULT lookup,
                            thresholds (0.03,0.10,0.20) → σ (0.15,0.25,0.35,0.45))
  (iii) 2-stage LUT adapt.: [B] (σ ∈ [0.15,0.35], SNR-mediated)
  annealing (open-loop)   : σ_s ∈ {0.6,1.2,2.4} → σ_e, linear/geom
                            — probes whether a LARGE early σ (mode-smearing) helps
  combo                   : min(LUT-adaptive, annealing)

Primary point 0.5 dB (EP ~1e-2 there → 3200 cw resolves CI; 0.6 dB floors ~6e-4).
Records BLER (Wilson CI), per-round payload-BER trajectory, per-chunk σ trajectory.
Writes results incrementally so partial runs are usable.
GPU recipe: TF on CPU, torch denoiser on GPU.
"""
import argparse, json, os
import numpy as np

from sigma_experiment import build_decoder, run as ep_run, K
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder

LUT = "results/sigma_2stage_lookup.json"
CONFIGS = [
    ("fixed_0.30",       "fixed",       dict(fixed_sigma=0.30)),
    ("fixed_0.15",       "fixed",       dict(fixed_sigma=0.15)),
    ("handcrafted",      "handcrafted", {}),
    ("lut2stage",        "adaptive",    dict(json_path=LUT)),
    ("anneal_geom_0.35", "annealing",   dict(anneal=("geom", 0.35, 0.15))),
    ("anneal_geom_0.6",  "annealing",   dict(anneal=("geom", 0.6, 0.15))),
    ("anneal_geom_1.2",  "annealing",   dict(anneal=("geom", 1.2, 0.15))),
    ("anneal_geom_2.4",  "annealing",   dict(anneal=("geom", 2.4, 0.15))),
    ("anneal_lin_1.2",   "annealing",   dict(anneal=("linear", 1.2, 0.15))),
    ("combo_lut_g1.2",   "combo",       dict(json_path=LUT, anneal=("geom", 1.2, 0.15))),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebno", type=float, nargs="+", default=[0.5])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=50)   # 3200 cw
    ap.add_argument("--only", nargs="+", default=None, help="subset of config labels")
    ap.add_argument("--out", default="results/sigma_compare.json")
    a = ap.parse_args()

    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=1)
    configs = [c for c in CONFIGS if (a.only is None or c[0] in a.only)]

    results = {}
    if os.path.exists(a.out):
        results = json.load(open(a.out))
    for label, strat, params in configs:
        dec = build_decoder(ldpc, strat, **params)
        for eb in a.ebno:
            key = f"{label}@{eb}"
            if key in results:
                print(f"skip {key} (cached)"); continue
            r = ep_run(dec, ldpc, eb, a.batch, a.rounds, track_traj=True)
            results[key] = {"label": label, "strategy": strat, "ebno": eb,
                            "bler": r["bler"], "ci": r["ci"], "nack": r["nack"],
                            "total": r["total"], "ber": r["ber"],
                            "sigma_traj": r["sigma_traj"], "ber_traj": r["ber_traj"]}
            json.dump(results, open(a.out, "w"), indent=2)
            print(f"{key:26} BLER={r['bler']:.4f} [{r['ci'][0]:.4f},{r['ci'][1]:.4f}] "
                  f"BER={r['ber']:.5f} ({r['nack']}/{r['total']})", flush=True)
    print("done", a.out)


if __name__ == "__main__":
    main()
