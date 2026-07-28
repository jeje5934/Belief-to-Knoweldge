"""
Pure-EP vs Baseline-BP Eb/N0 sweep (waterfall).

Curves (README §10 / docs/EP_SCHEDULING_EXPERIMENT.md):
  * bp30      — Baseline BP, 30 iterations, NO source prior.  Same total BP
                budget as the EP schedule [2]*15 (= 30 BP iters).  [version a]
  * bp100     — Baseline BP, 100 iterations.  BP-limit reference line. [version c]
  * ep        — Pure EP decoder: bp_schedule=[2]*15, ep_mode=True,
                ep_update="full_ep" (α_ep=β_ep=1), adaptive syndrome-ratio σ.

Outputs (to results/):
  * ep_snr_sweep.csv   — one row per (ebno, decoder): ber, bler, counts.
  * ep_snr_sweep.png   — two subplots (BER, BLER) vs Eb/N0, log-y.
  * ep_snr_sweep.json  — summary + EP per-SNR schedule-adequacy diagnostics.

CPU-only by default (safe; the denoiser bridge is CPU-bound anyway).
"""
import os
import sys
import csv
import json
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no

from decoder import LDPC5GDecoder_soft

IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP
N_CODEWORD = 12600
NUM_BPS = 1
CKPT = "checkpoints/denoiser.pt"
EP_SCHEDULE = [2] * 15          # confirmed sweet-spot schedule
SEED = 42


def build_test_bitbank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=False,
                                           download=True)
    images = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = images.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    return tf.constant(np.unpackbits(flat, axis=1), dtype=tf.int32)


def make_common():
    crc_enc = CRCEncoder("CRC24A")
    crc_dec = CRCDecoder(crc_enc)
    ldpc_enc = LDPC5GEncoder(K_PAYLOAD + crc_enc.crc_length, N_CODEWORD,
                             num_bits_per_symbol=NUM_BPS)
    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam",
                        num_bits_per_symbol=NUM_BPS)
    return crc_enc, crc_dec, ldpc_enc, mapper, demapper, AWGN()


def build_decoders(ldpc_enc):
    common = dict(cn_update="boxplus-phi", vn_update="sum",
                  cn_schedule="flooding", hard_out=False,
                  return_infobits=True, llr_max=30.0)
    bp30 = LDPC5GDecoder(ldpc_enc, num_iter=30, **common)
    bp100 = LDPC5GDecoder(ldpc_enc, num_iter=100, **common)
    ep = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=EP_SCHEDULE, k_payload=K_PAYLOAD,
        ep_mode=True, ep_update="full_ep", adaptive_sigma=True,
        num_iter=sum(EP_SCHEDULE), **common)
    if os.path.isfile(CKPT):
        ep.denoiser.load_weights_pt(CKPT)
    else:
        print(f"[WARN] checkpoint missing ({CKPT}); denoiser weights random.")
    return {"bp30": bp30, "bp100": bp100, "ep": ep}


def run_point(dec, name, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
              bit_bank, ebno, batch, rounds):
    """Return (bler, ber, nack, total, biterr, ep_diag) for one (dec, ebno)."""
    no = ebnodb2no(ebno, NUM_BPS, ldpc_enc.coderate)
    n_imgs = tf.shape(bit_bank)[0]
    tot_nack = tot_biterr = 0
    total = batch * rounds
    dsite_first, dsite_last = [], []
    for r in range(rounds):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        idx = tf.random.uniform([batch], 0, n_imgs, dtype=tf.int32)
        u = tf.gather(bit_bank, idx)
        u_crc = crc_enc(tf.cast(u, ldpc_enc.rdtype))
        y = awgn(mapper(ldpc_enc(u_crc)), no)
        llr = demapper(y, no)
        hat = dec(llr)
        _, cv = hard_crc_decode(crc_dec, hat)
        ack = int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())   # cv=1 => CRC valid => ACK
        tot_nack += (batch - ack)
        tot_biterr += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())
        if name == "ep" and getattr(dec, "last_chunk_diagnostics", None):
            dl = dec.last_chunk_diagnostics
            dsite_first.append(dl[0]["src_site_delta_l2"])
            dsite_last.append(dl[-1]["src_site_delta_l2"])
    bler = tot_nack / total
    ber = tot_biterr / (total * K_PAYLOAD)
    ep_diag = None
    if name == "ep" and dsite_first:
        ep_diag = {"dsite_first_mean": float(np.mean(dsite_first)),
                   "dsite_last_mean": float(np.mean(dsite_last)),
                   "dsite_plateau_ratio": float(np.mean(dsite_last) /
                                                max(np.mean(dsite_first), 1e-9))}
    return bler, ber, tot_nack, total, tot_biterr, ep_diag


def interp_ebno_at_bler(ebnos, blers, target):
    """Linear-in-(Eb/N0, log10 BLER) crossing of `target`; None if not bracketed."""
    xs = list(ebnos)
    ys = [b if b > 0 else 1e-6 for b in blers]
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            l0, l1, lt = np.log10(y0), np.log10(y1), np.log10(target)
            f = (lt - l0) / (l1 - l0)
            return xs[i] + f * (xs[i + 1] - xs[i])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ebno-list", type=float, nargs="+",
                   default=[0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--rounds", type=int, default=24)   # 768 codewords / point
    p.add_argument("--out-prefix", default="results/ep_snr_sweep")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    crc_enc, crc_dec, ldpc_enc, mapper, demapper, awgn = make_common()
    bit_bank = build_test_bitbank()
    decs = build_decoders(ldpc_enc)

    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["ebno_db", "decoder", "bler", "ber", "nack", "total", "biterr"])

    results = {name: {"ebno": [], "bler": [], "ber": []} for name in decs}
    ep_adequacy = []
    total_cw = args.batch * args.rounds
    print(f"EP SNR sweep: {total_cw} codewords/point, schedule={EP_SCHEDULE}, "
          f"Eb/N0={args.ebno_list}\n")
    for ebno in args.ebno_list:
        for name, dec in decs.items():
            bler, ber, nack, total, biterr, ep_diag = run_point(
                dec, name, ldpc_enc, crc_enc, crc_dec, mapper, demapper, awgn,
                bit_bank, ebno, args.batch, args.rounds)
            results[name]["ebno"].append(ebno)
            results[name]["bler"].append(bler)
            results[name]["ber"].append(ber)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [ebno, name, f"{bler:.6g}", f"{ber:.6g}", nack, total, biterr])
            extra = ""
            if ep_diag:
                extra = (f"  [Δsite {ep_diag['dsite_first_mean']:.0f}→"
                         f"{ep_diag['dsite_last_mean']:.0f}, "
                         f"ratio {ep_diag['dsite_plateau_ratio']:.2f}]")
                ep_adequacy.append({"ebno": ebno, **ep_diag})
            print(f"  Eb/N0={ebno:>4}  {name:5}  BLER={bler:.4f}  "
                  f"BER={ber:.2e}  ({nack}/{total} NACK){extra}")
        print()

    # waterfall shift at a couple of BLER targets (ep vs bp30)
    shifts = {}
    for tgt in (0.5, 0.1):
        e_ep = interp_ebno_at_bler(results["ep"]["ebno"], results["ep"]["bler"], tgt)
        e_bp = interp_ebno_at_bler(results["bp30"]["ebno"], results["bp30"]["bler"], tgt)
        shifts[f"bler_{tgt}"] = {
            "ep_ebno": e_ep, "bp30_ebno": e_bp,
            "shift_db": (None if (e_ep is None or e_bp is None) else e_bp - e_ep)}

    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"schedule": EP_SCHEDULE, "codewords_per_point": total_cw,
                   "ebno_list": args.ebno_list, "results": results,
                   "waterfall_shift_db_vs_bp30": shifts,
                   "ep_schedule_adequacy": ep_adequacy}, f, indent=2)

    _plot(results, args.out_prefix + ".png")
    print("waterfall shift (EP left of bp30):")
    for k, v in shifts.items():
        print(f"  {k}: {v}")
    print(f"\nsaved: {csv_path}, {args.out_prefix}.png, {args.out_prefix}.json")


def _plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"bp30": "Baseline BP (30 it, no prior)",
              "bp100": "Baseline BP (100 it, ref)",
              "ep": "Pure EP ([2]×15, full_ep, adaptive σ)"}
    styles = {"bp30": "s--", "bp100": "^:", "ep": "o-"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for name in ("bp30", "bp100", "ep"):
        e = results[name]["ebno"]
        ax1.semilogy(e, [max(b, 1e-6) for b in results[name]["ber"]],
                     styles[name], label=labels[name], linewidth=2, markersize=6)
        ax2.semilogy(e, [max(b, 1e-4) for b in results[name]["bler"]],
                     styles[name], label=labels[name], linewidth=2, markersize=6)
    ax1.set_ylabel("BER"); ax1.grid(True, which="both", alpha=0.3); ax1.legend()
    ax1.set_title("Pure EP vs Baseline BP — Fashion-MNIST / 5G-LDPC / AWGN")
    ax2.set_ylabel("BLER"); ax2.set_xlabel("Eb/N0 (dB)")
    ax2.grid(True, which="both", alpha=0.3); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
