"""
[2] legacy robust-α search — legacy ONLY (EP not swept).  One legacy decoder is
built once; α (and β) are set per candidate (they are plain attributes), and the
channel is rebuilt per σ_e².  Goal: a SINGLE (α,β) that is good across the whole
σ_e² range (minimax), since a real system does not know / cannot track CSI
quality.

CSV: alpha,beta,sigma,sigma_e2,ebno_db,bler,ci_lo,ci_hi,nack,total
"""
import argparse, csv, os
import numpy as np
import tensorflow as tf
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.utils import ebnodb2no

from sigma_experiment import _bank, K, wilson
from fading_3way import build_legacy
from channel_models import ChannelModel, num_bps_for

SEED = 42


def legacy_bler(dec, ldpc, crc, crcd, channel, ebno, batch, rounds):
    bank = _bank(); no = ebnodb2no(ebno, channel.num_bps, ldpc.coderate)
    nack = 0
    for r in range(rounds):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        idx = tf.random.uniform([batch], 0, tf.shape(bank)[0], dtype=tf.int32)
        u = tf.gather(bank, idx)
        llr = channel.transmit(ldpc(crc(tf.cast(u, ldpc.rdtype))), no)
        hat = dec(llr); _, cv = crcd(hat)
        nack += batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
    return wilson(nack, batch * rounds), nack, batch * rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="qpsk_fast_fading_imperfect_csi")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2])
    ap.add_argument("--beta-mode", default="matched", choices=["matched", "zero", "half"])
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--sigma-e2", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    ap.add_argument("--ebno", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", default="results/fading_legacy_tune.csv")
    a = ap.parse_args()

    nb = num_bps_for(a.channel)
    ldpc = LDPC5GEncoder(K + 24, 12600, num_bits_per_symbol=nb)
    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    dec = build_legacy(ldpc, a.sigma, a.alphas[0], a.alphas[0])   # built once
    dec.denoiser.sigma = a.sigma

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    fields = ["alpha", "beta", "sigma", "sigma_e2", "ebno_db",
              "bler", "ci_lo", "ci_hi", "nack", "total"]
    f = open(a.out, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    results = {}   # (alpha) -> {(se2,ebno): bler}
    for alpha in a.alphas:
        beta = {"matched": alpha, "zero": 0.0, "half": alpha / 2}[a.beta_mode]
        dec.alpha = float(alpha); dec.beta = float(beta)
        for se2 in a.sigma_e2:
            ch = ChannelModel(a.channel, sigma_e2=se2, perfect_csi=(se2 == 0.0), llr_method="B")
            for ebno in a.ebno:
                (p, lo, hi), nack, tot = legacy_bler(dec, ldpc, crc, crcd, ch, ebno, a.batch, a.rounds)
                w.writerow({"alpha": alpha, "beta": beta, "sigma": a.sigma,
                            "sigma_e2": se2, "ebno_db": ebno, "bler": p,
                            "ci_lo": lo, "ci_hi": hi, "nack": nack, "total": tot})
                f.flush()
                results.setdefault(alpha, {})[(se2, ebno)] = p
                print(f"  a={alpha} b={beta} se2={se2} ebno={ebno}: BLER={p:.4f}", flush=True)
    f.close()

    # minimax: for each ebno, the α with the best worst-case over σ_e²
    print("\n=== minimax robust α (worst-case BLER over σ_e²) ===")
    for ebno in a.ebno:
        print(f" ebno={ebno}:")
        best, best_a = 1e9, None
        for alpha in a.alphas:
            wc = max(results[alpha][(se2, ebno)] for se2 in a.sigma_e2)
            print(f"   α={alpha}: worst-case BLER={wc:.4f}")
            if wc < best:
                best, best_a = wc, alpha
        print(f"   -> robust α at ebno {ebno}: {best_a} (worst-case {best:.4f})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
