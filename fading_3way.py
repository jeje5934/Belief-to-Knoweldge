"""
3-way rough-channel diagnostic: BP-only vs legacy (warm-start turbo) vs EP,
all fed the SAME (method-B) channel LLR.  Channel/LLR generated once per round
outside the decoders.

  BP     : LDPC5GDecoder, 100 iters, no denoiser.
  legacy : LDPC5GDecoder_soft, ep_mode=False, source_input_mode="bp_post",
           α=β=0.1, fixed σ=0.3, [5]×20   (AWGN-best warm-start turbo, §F).
  EP     : LDPC5GDecoder_soft, ep_mode=True, damped_ep, ep_source_power=α_ep,
           fixed σ=0.3, [5]×20            (accurate cavity; §F EP-best is fixed σ).

CSV: channel,num_bps,sigma_e2,perfect_csi,ebno_db,llr_method,
     bler_bp,ber_bp,bler_legacy,ber_legacy,bler_ep,ber_ep,(+CI),nack_*,total,
     alpha_legacy,beta_legacy,alpha_ep

GPU recipe: TF on CPU, torch denoiser on GPU (via sigma_experiment import).
"""
import argparse, csv, os
import numpy as np
import tensorflow as tf
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.utils import ebnodb2no

from sigma_experiment import _bank, K, wilson, DEV, SCHEDULE
from decoder import LDPC5GDecoder_soft
from channel_models import ChannelModel, num_bps_for

SEED = 42
_COMMON = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
               hard_out=False, return_infobits=True, llr_max=30.0)


def build_legacy(ldpc, sigma=0.3, alpha=0.1, beta=0.1):
    d = LDPC5GDecoder_soft(ldpc, k_payload=K, num_iter=sum(SCHEDULE),
                           bp_schedule=SCHEDULE, ep_mode=False,
                           source_input_mode="bp_post", adaptive_sigma=False,
                           denoiser_kwargs=dict(device=DEV), **_COMMON)
    d.denoiser.load_weights_pt("checkpoints/denoiser.pt")
    d.denoiser.sigma = float(sigma); d.alpha = float(alpha); d.beta = float(beta)
    return d


def build_ep(ldpc, sigma=0.3, alpha_ep=0.01):
    d = LDPC5GDecoder_soft(ldpc, k_payload=K, num_iter=sum(SCHEDULE),
                           bp_schedule=SCHEDULE, ep_mode=True, ep_update="damped_ep",
                           ep_source_power=float(alpha_ep), ep_code_power=1.0,
                           adaptive_sigma=False, denoiser_kwargs=dict(device=DEV),
                           **_COMMON)
    d.denoiser.load_weights_pt("checkpoints/denoiser.pt")
    d.denoiser.sigma = float(sigma)
    return d


def run_point(channel, ldpc, crc, crcd, decs, ebno, batch, rounds):
    bank = _bank()
    no = ebnodb2no(ebno, channel.num_bps, ldpc.coderate)
    nack = {k: 0 for k in decs}; berr = {k: 0 for k in decs}
    for r in range(rounds):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        idx = tf.random.uniform([batch], 0, tf.shape(bank)[0], dtype=tf.int32)
        u = tf.gather(bank, idx)
        c = ldpc(crc(tf.cast(u, ldpc.rdtype)))
        llr = channel.transmit(c, no)                       # SAME LLR to all
        for name, dec in decs.items():
            hat = dec(llr); _, cv = hard_crc_decode(crcd, hat)
            nack[name] += batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
            berr[name] += int(tf.reduce_sum(tf.cast(
                tf.not_equal(u, tf.cast(hat[:, :K] > 0, tf.int32)), tf.int32)).numpy())
    tot = batch * rounds
    out = {}
    for name in decs:
        p, lo, hi = wilson(nack[name], tot)
        out[name] = {"bler": p, "ci": [lo, hi], "ber": berr[name] / (tot * K),
                     "nack": nack[name]}
    out["total"] = tot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="qpsk_fast_fading_imperfect_csi")
    ap.add_argument("--sigma-e2", type=float, nargs="+", default=[0.0])
    ap.add_argument("--perfect-csi", action="store_true")
    ap.add_argument("--llr-method", default="B", choices=["A", "B"])
    ap.add_argument("--ebno", type=float, nargs="+", default=[8.0])
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--alpha-legacy", type=float, default=0.1)
    ap.add_argument("--beta-legacy", type=float, default=0.1)
    ap.add_argument("--alpha-ep", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", default="results/fading_3way.csv")
    a = ap.parse_args()

    nb = num_bps_for(a.channel)
    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=nb)
    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    decs = {"bp": LDPC5GDecoder(ldpc, num_iter=100, **_COMMON),
            "legacy": build_legacy(ldpc, a.sigma, a.alpha_legacy, a.beta_legacy),
            "ep": build_ep(ldpc, a.sigma, a.alpha_ep)}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    fields = ["channel", "num_bps", "sigma_e2", "perfect_csi", "ebno_db", "llr_method",
              "bler_bp", "ber_bp", "bler_legacy", "ber_legacy", "bler_ep", "ber_ep",
              "bp_ci_lo", "bp_ci_hi", "legacy_ci_lo", "legacy_ci_hi",
              "ep_ci_lo", "ep_ci_hi", "nack_bp", "nack_legacy", "nack_ep", "total",
              "alpha_legacy", "beta_legacy", "alpha_ep"]
    f = open(a.out, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    for se2 in a.sigma_e2:               # decoders reused; only the channel changes
        perfect = a.perfect_csi or se2 == 0.0
        channel = ChannelModel(a.channel, sigma_e2=se2, perfect_csi=perfect,
                               llr_method=a.llr_method)
        for ebno in a.ebno:
            r = run_point(channel, ldpc, crc, crcd, decs, ebno, a.batch, a.rounds)
            row = {"channel": a.channel, "num_bps": nb, "sigma_e2": se2,
                   "perfect_csi": perfect, "ebno_db": ebno, "llr_method": a.llr_method,
                   "bler_bp": r["bp"]["bler"], "ber_bp": r["bp"]["ber"],
                   "bler_legacy": r["legacy"]["bler"], "ber_legacy": r["legacy"]["ber"],
                   "bler_ep": r["ep"]["bler"], "ber_ep": r["ep"]["ber"],
                   "bp_ci_lo": r["bp"]["ci"][0], "bp_ci_hi": r["bp"]["ci"][1],
                   "legacy_ci_lo": r["legacy"]["ci"][0], "legacy_ci_hi": r["legacy"]["ci"][1],
                   "ep_ci_lo": r["ep"]["ci"][0], "ep_ci_hi": r["ep"]["ci"][1],
                   "nack_bp": r["bp"]["nack"], "nack_legacy": r["legacy"]["nack"],
                   "nack_ep": r["ep"]["nack"], "total": r["total"],
                   "alpha_legacy": a.alpha_legacy, "beta_legacy": a.beta_legacy,
                   "alpha_ep": a.alpha_ep}
            w.writerow(row); f.flush()
            print(f"[se2={se2} pcsi={perfect}] ebno={ebno} | BP={r['bp']['bler']:.4f} | "
                  f"legacy={r['legacy']['bler']:.4f}[{r['legacy']['ci'][0]:.4f},"
                  f"{r['legacy']['ci'][1]:.4f}] | EP={r['ep']['bler']:.4f}"
                  f"[{r['ep']['ci'][0]:.4f},{r['ep']['ci'][1]:.4f}]", flush=True)
    f.close(); print("wrote", a.out)


if __name__ == "__main__":
    main()
