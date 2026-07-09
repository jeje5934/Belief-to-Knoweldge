"""
Rough-channel robustness runner: BP-only baseline vs BP+denoiser (EP [5]×20,
fixed σ) over a selectable channel, BOTH fed the SAME (mismatched-CSI) LLR.

    --channel awgn_bpsk | qpsk_fast_fading_imperfect_csi
    --sigma-e2 FLOAT     (CSI error variance; sweep 0.01 0.05 0.1 0.2)
    --perfect-csi        (force ĥ=h)
    --ebno ...           (Eb/N0 dB points)

CSV columns: channel,num_bps,sigma_e2,perfect_csi,ebno_db,
             bler_baseline,ber_baseline,bler_fixed,ber_fixed,
             baseline_ci_lo,baseline_ci_hi,fixed_ci_lo,fixed_ci_hi,nack_*,total

GPU recipe: TF on CPU, torch denoiser on GPU (sigma_experiment sets this up).
"""
import argparse, csv, math, os
import numpy as np
import tensorflow as tf
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.utils import ebnodb2no

from sigma_experiment import build_decoder, _bank, K, wilson
from channel_models import ChannelModel, num_bps_for

SEED = 42


def run_point(channel, ldpc, crc, crcd, dec_base, dec_fixed, ebno, batch, rounds):
    bank = _bank()
    no = ebnodb2no(ebno, channel.num_bps, ldpc.coderate)
    nb = nf = eb = ef = 0
    for r in range(rounds):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        idx = tf.random.uniform([batch], 0, tf.shape(bank)[0], dtype=tf.int32)
        u = tf.gather(bank, idx)
        c = ldpc(crc(tf.cast(u, ldpc.rdtype)))
        llr = channel.transmit(c, no)                      # same LLR to both
        for dec, is_base in ((dec_base, True), (dec_fixed, False)):
            hat = dec(llr)
            _, cv = crcd(hat)
            nack = batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
            berr = int(tf.reduce_sum(tf.cast(
                tf.not_equal(u, tf.cast(hat[:, :K] > 0, tf.int32)), tf.int32)).numpy())
            if is_base:
                nb += nack; eb += berr
            else:
                nf += nack; ef += berr
    tot = batch * rounds
    pb, blo, bhi = wilson(nb, tot)
    pf, flo, fhi = wilson(nf, tot)
    return {"bler_baseline": pb, "ber_baseline": eb / (tot * K),
            "bler_fixed": pf, "ber_fixed": ef / (tot * K),
            "baseline_ci_lo": blo, "baseline_ci_hi": bhi,
            "fixed_ci_lo": flo, "fixed_ci_hi": fhi,
            "nack_baseline": nb, "nack_fixed": nf, "total": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="awgn_bpsk",
                    choices=["awgn_bpsk", "qpsk_fast_fading_imperfect_csi"])
    ap.add_argument("--sigma-e2", type=float, default=0.0)
    ap.add_argument("--perfect-csi", action="store_true")
    ap.add_argument("--ebno", type=float, nargs="+", default=[0.6])
    ap.add_argument("--sigma", type=float, default=0.3, help="denoiser fixed σ")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--out", default="results/fading_experiment.csv")
    a = ap.parse_args()

    num_bps = num_bps_for(a.channel)
    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=num_bps)
    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    dec_base = LDPC5GDecoder(ldpc, num_iter=100, **common)      # BP-only
    dec_fixed = build_decoder(ldpc, "fixed", fixed_sigma=a.sigma)  # EP [5]×20 fixed σ

    channel = ChannelModel(a.channel, sigma_e2=a.sigma_e2, perfect_csi=a.perfect_csi)
    perfect = a.perfect_csi or a.sigma_e2 == 0.0

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    fields = ["channel", "num_bps", "sigma_e2", "perfect_csi", "ebno_db",
              "bler_baseline", "ber_baseline", "bler_fixed", "ber_fixed",
              "baseline_ci_lo", "baseline_ci_hi", "fixed_ci_lo", "fixed_ci_hi",
              "nack_baseline", "nack_fixed", "total"]
    f = open(a.out, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    for ebno in a.ebno:
        r = run_point(channel, ldpc, crc, crcd, dec_base, dec_fixed,
                      ebno, a.batch, a.rounds)
        row = {"channel": a.channel, "num_bps": num_bps, "sigma_e2": a.sigma_e2,
               "perfect_csi": perfect, "ebno_db": ebno, **r}
        w.writerow(row); f.flush()
        print(f"[{a.channel} se2={a.sigma_e2} pcsi={perfect}] ebno={ebno}  "
              f"BASE BLER={r['bler_baseline']:.4f}[{r['baseline_ci_lo']:.4f},"
              f"{r['baseline_ci_hi']:.4f}] BER={r['ber_baseline']:.5f}  |  "
              f"FIXED BLER={r['bler_fixed']:.4f}[{r['fixed_ci_lo']:.4f},"
              f"{r['fixed_ci_hi']:.4f}] BER={r['ber_fixed']:.5f}", flush=True)
    f.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
