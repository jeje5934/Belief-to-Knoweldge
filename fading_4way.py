"""
[4] minus-mode probe (§F D<C mechanism).  4-way BP / legacy / minus / EP on the
SAME method-B LLR, at two points (perfect+fading and σ_e²=0.1), 3200 cw.

  legacy : source_input_mode="bp_post"           (SELF-ANCHORED cavity)
  minus  : source_input_mode="minus_source_feedback" (SOURCE-FREE / accurate
           cavity, like EP — but in the legacy α/β frame; same robust α as legacy)
  EP     : exact cavity (ep_mode)

If minus tracks EP (fragile) rather than legacy (robust) under fading, the
self-anchoring of the incomplete cavity IS the robustness source → closes §F.
"""
import argparse, csv, os
import tensorflow as tf
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.utils import ebnodb2no

from sigma_experiment import _bank, K, wilson, DEV, SCHEDULE
from decoder import LDPC5GDecoder_soft
from channel_models import ChannelModel, num_bps_for
from fading_3way import build_legacy, build_ep, _COMMON, SEED, run_point


def build_minus(ldpc, sigma=0.3, alpha=0.15, beta=0.15):
    d = LDPC5GDecoder_soft(ldpc, k_payload=K, num_iter=sum(SCHEDULE), bp_schedule=SCHEDULE,
                           ep_mode=False, source_input_mode="minus_source_feedback",
                           adaptive_sigma=False, denoiser_kwargs=dict(device=DEV), **_COMMON)
    d.denoiser.load_weights_pt("checkpoints/denoiser.pt")
    d.denoiser.sigma = float(sigma); d.alpha = float(alpha); d.beta = float(beta)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="qpsk_fast_fading_imperfect_csi")
    ap.add_argument("--points", nargs="+", default=["0.0:2", "0.1:3"],
                    help="se2:ebno pairs")
    ap.add_argument("--alpha", type=float, default=0.15, help="robust α for legacy & minus")
    ap.add_argument("--alpha-ep", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=50)     # 3200 cw
    ap.add_argument("--out", default="results/fading_4way.csv")
    a = ap.parse_args()

    nb = num_bps_for(a.channel)
    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=nb)
    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    decs = {"bp": LDPC5GDecoder(ldpc, num_iter=100, **_COMMON),
            "legacy": build_legacy(ldpc, a.sigma, a.alpha, a.alpha),
            "minus": build_minus(ldpc, a.sigma, a.alpha, a.alpha),
            "ep": build_ep(ldpc, a.sigma, a.alpha_ep)}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    fields = ["channel", "sigma_e2", "perfect_csi", "ebno_db", "alpha", "alpha_ep",
              "bler_bp", "bler_legacy", "bler_minus", "bler_ep",
              "legacy_ci_lo", "legacy_ci_hi", "minus_ci_lo", "minus_ci_hi",
              "ep_ci_lo", "ep_ci_hi", "nack_bp", "nack_legacy", "nack_minus",
              "nack_ep", "total"]
    f = open(a.out, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    for pt in a.points:
        se2, ebno = float(pt.split(":")[0]), float(pt.split(":")[1])
        perfect = se2 == 0.0
        ch = ChannelModel(a.channel, sigma_e2=se2, perfect_csi=perfect, llr_method="B")
        r = run_point(ch, ldpc, crc, crcd, decs, ebno, a.batch, a.rounds)
        row = {"channel": a.channel, "sigma_e2": se2, "perfect_csi": perfect,
               "ebno_db": ebno, "alpha": a.alpha, "alpha_ep": a.alpha_ep,
               "bler_bp": r["bp"]["bler"], "bler_legacy": r["legacy"]["bler"],
               "bler_minus": r["minus"]["bler"], "bler_ep": r["ep"]["bler"],
               "legacy_ci_lo": r["legacy"]["ci"][0], "legacy_ci_hi": r["legacy"]["ci"][1],
               "minus_ci_lo": r["minus"]["ci"][0], "minus_ci_hi": r["minus"]["ci"][1],
               "ep_ci_lo": r["ep"]["ci"][0], "ep_ci_hi": r["ep"]["ci"][1],
               "nack_bp": r["bp"]["nack"], "nack_legacy": r["legacy"]["nack"],
               "nack_minus": r["minus"]["nack"], "nack_ep": r["ep"]["nack"],
               "total": r["total"]}
        w.writerow(row); f.flush()
        print(f"[se2={se2} ebno={ebno}] BP={r['bp']['bler']:.4f} | "
              f"legacy={r['legacy']['bler']:.4f}[{r['legacy']['ci'][0]:.4f},{r['legacy']['ci'][1]:.4f}] | "
              f"minus={r['minus']['bler']:.4f}[{r['minus']['ci'][0]:.4f},{r['minus']['ci'][1]:.4f}] | "
              f"EP={r['ep']['bler']:.4f}[{r['ep']['ci'][0]:.4f},{r['ep']['ci'][1]:.4f}]", flush=True)
    f.close(); print("wrote", a.out)


if __name__ == "__main__":
    main()
