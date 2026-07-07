"""
Phase-2 stage B (TensorFlow / Sionna).  NO torch in this process (torch-CUDA +
TF-CUDA in one process segfaults on this host — docs/COMPUTE_LESSONS §5).

Separation-baseline channel pipeline, per image:
    compressed bits (B_c) -> pad to fixed container K (zeros) -> CRC24A
      -> LDPC5GEncoder(k=K+24, n=12600) -> BPSK -> AWGN
      -> LDPC5GDecoder (pure BP, num_iter=100, NO denoiser) -> CRC24A check.
CRC pass = success (same criterion as the denoiser-in-the-loop system).

Two containers:
    P99 : K = ceil(p99 of B_c)  (images with B_c > K are pre-transmission
          failures — counted, not transmitted)   rate ~ 0.331
    MAX : K = max B_c            (0 overflow)                       rate ~ 0.389

Energy-equivalence (main axis): both systems use N=12600 BPSK symbols at the
SAME noise variance `no` (== same Es/N0 == same total energy).  We derive `no`
from the raw system's Eb/N0 points {0.5,0.6,0.7} at its rate 0.4997, then apply
the SAME `no` to the baseline.  The baseline's own (much higher) Eb/N0 at that
`no` is reported as a secondary axis (its lower rate spends more energy/infobit).

Output: results/channel_bler_<tag>.json
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

# GPU visibility is controlled by CUDA_VISIBLE_DEVICES at launch.  This process
# never imports torch, so TF-on-GPU is safe here (no TF+torch CUDA mix).
for _g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_g, True)

from sionna.phy.utils import ebnodb2no
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.channel import AWGN

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
STREAMS = os.path.join(RESULTS, "channel_streams.npz")

N_CODEWORD = 12600
NUM_BPS = 1
CRC_DEGREE = "CRC24A"
CRC_LEN = 24
RAW_K = 6296
RAW_RATE = RAW_K / N_CODEWORD          # 0.499682...
SEED = 42


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def run_config(bits, lengths, imgs, K, esno_points, batch, use_gpu, tag):
    """esno_points: list of dicts {label, no, esno_db, ebno_raw}."""
    k_ldpc = K + CRC_LEN
    rate = k_ldpc / N_CODEWORD
    ebno_base_shift = -10.0 * np.log10(rate)   # Eb/N0_base = Es/N0 + this

    N = bits.shape[0]
    overflow_mask = lengths > K
    n_over = int(overflow_mask.sum())
    keep = ~overflow_mask
    u = bits[keep, :K].astype(np.float32)      # [n_ok, K], zero-padded stream
    n_ok = u.shape[0]
    print(f"\n=== container {tag}: K={K} k_ldpc={k_ldpc} rate={rate:.4f} "
          f"| overflow {n_over}/{N} ({n_over/N*100:.2f}%) | transmit {n_ok} ===",
          flush=True)

    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)
    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    awgn = AWGN()
    dec = LDPC5GDecoder(ldpc_enc, cn_update="boxplus-phi", vn_update="sum",
                        cn_schedule="flooding", hard_out=True,
                        return_infobits=True, num_iter=100, llr_max=30.0)

    # pre-encode all codewords once (SNR-independent)
    u_tf = tf.constant(u, dtype=ldpc_enc.rdtype)
    results = {}
    for pt in esno_points:
        no = pt["no"]
        tf.random.set_seed(SEED + int(round(pt["esno_db"] * 1000)))
        ldpc_fail = 0
        for i in range(0, n_ok, batch):
            ub = u_tf[i:i + batch]
            u_crc = crc_enc(ub)
            c = ldpc_enc(u_crc)
            x = mapper(c)
            y = awgn(x, tf.constant(no, dtype=tf.float32))
            llr = demapper(y, tf.constant(no, dtype=tf.float32))
            hat = dec(llr)
            _, cv = crc_dec(hat)                # cv True == CRC pass
            ldpc_fail += int(tf.reduce_sum(tf.cast(~cv, tf.int32)).numpy())
        nack = n_over + ldpc_fail               # overflow always fails
        p, lo, hi = wilson(nack, N)
        _, flo, fhi = wilson(ldpc_fail, n_ok)
        ebno_base = pt["esno_db"] + ebno_base_shift
        results[pt["label"]] = {
            "esno_db": pt["esno_db"], "no": no,
            "ebno_raw_equiv": pt["ebno_raw"], "ebno_base": ebno_base,
            "N": N, "n_transmit": n_ok, "overflow": n_over,
            "ldpc_fail": ldpc_fail, "nack": nack,
            "bler": p, "bler_ci": [lo, hi],
            "ldpc_bler": ldpc_fail / n_ok if n_ok else 0.0,
            "ldpc_bler_ci": [flo, fhi],
        }
        print(f"  Es/N0={pt['esno_db']:+.3f}dB (raw Eb/N0={pt['ebno_raw']}, "
              f"base Eb/N0={ebno_base:.2f}) | overflow={n_over} ldpc_fail={ldpc_fail} "
              f"| BLER={p:.4f} [{lo:.4f},{hi:.4f}]", flush=True)
    return {"K": K, "k_ldpc": k_ldpc, "rate": rate, "overflow": n_over,
            "n_transmit": n_ok, "points": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=320)
    ap.add_argument("--ebno_raw", type=float, nargs="+", default=[0.5, 0.6, 0.7],
                    help="raw-system Eb/N0 points defining the main Es/N0 axis")
    ap.add_argument("--esno_extra", type=float, nargs="+", default=[],
                    help="extra Es/N0(dB) points for baseline waterfall")
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TF devices: GPU={gpus if gpus else 'hidden/CPU'}")

    d = np.load(STREAMS)
    bits, lengths, imgs = d["bits"], d["lengths"], d["imgs"]
    N = bits.shape[0]
    print(f"loaded {N} streams  B_c mean={lengths.mean():.1f} "
          f"p99={np.percentile(lengths,99):.1f} max={lengths.max()}")

    # main Es/N0 axis from raw Eb/N0 points (same `no` applied to baseline)
    esno_points = []
    for eb in args.ebno_raw:
        no = float(ebnodb2no(eb, NUM_BPS, RAW_RATE))
        esno_db = -10.0 * np.log10(no)          # Es/N0 (Es=1 for unit-energy BPSK)
        esno_points.append({"label": f"raw{eb}", "no": no,
                            "esno_db": float(esno_db), "ebno_raw": eb})
    for es in args.esno_extra:
        no = float(10.0 ** (-es / 10.0))
        esno_points.append({"label": f"es{es}", "no": no,
                            "esno_db": float(es), "ebno_raw": None})
    print("Es/N0 points:")
    for pt in esno_points:
        print(f"  {pt['label']}: Es/N0={pt['esno_db']:+.4f}dB  no={pt['no']:.5f}")

    # Sionna 5G LDPC feasibility at n=12600 (measured): BG2 needs k<=3824
    # (rate<=0.3035), BG1 needs rate>1/3 (k>=4200).  Rates in (0.3035, 0.3333)
    # are UNSUPPORTED (no base graph) -- this corrects the Phase-1 floor claim
    # and makes the intended p99 container (k=4171, rate 0.331) infeasible.
    # We therefore use the nearest feasible containers:
    #   MAX      : k = max B_c + 24                    (0 overflow, BG1)
    #   BG1LEAN  : k = 4200  (rate 1/3, smallest BG1)  (~1% overflow) <- p99 stand-in
    #   BG2LEAN  : k = 3824  (rate 0.3035, largest BG2)(higher overflow, more parity)
    K_max = int(lengths.max())
    configs = [("MAX", K_max), ("BG1LEAN", 4200 - CRC_LEN), ("BG2LEAN", 3824 - CRC_LEN)]
    out = {"raw_rate": RAW_RATE, "N": N,
           "Bc_mean": float(lengths.mean()),
           "Bc_p99": float(np.percentile(lengths, 99)),
           "Bc_max": int(lengths.max()),
           "ldpc_infeasible_gap": "rate in (0.3035, 0.3333) unsupported at n=12600",
           "configs": {}}
    for label, K in configs:
        out["configs"][label] = run_config(bits, lengths, imgs, K,
                                            esno_points, args.batch, bool(gpus), label)

    with open(os.path.join(RESULTS, "channel_bler.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote channel_bler.json")


if __name__ == "__main__":
    main()
