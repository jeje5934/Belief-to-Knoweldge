"""
Prompt C, steps 4-6 — damped-EP performance with sigma_post=ε vs 3.0.

Combos (all damped EP, ep_source_power=α; baselines share the EP BP budget K=30):
  FMNIST images (multi-cat denoiser):
    A    : sp=3.0, α=0.02   (current baseline)
    B    : sp=ε,   α=0.02   (ε effect, isolated)
    B05  : sp=ε,   α=0.05   (step 5 α sweep)
    B10  : sp=ε,   α=0.10
  Trouser images:
    Atr  : multi denoiser,   sp=3.0, α=0.02  (control for C)
    C    : trouser denoiser, sp=3.0, α=0.02  (unimodal prior effect)
    D    : trouser denoiser, sp=ε,   α=0.02  (combined)
    D05  : trouser denoiser, sp=ε,   α=0.05
    D10  : trouser denoiser, sp=ε,   α=0.10
  + baseline BP-30 (same budget) and BP-100 (ceiling) per image set.

Metrics: pooled BLER + Wilson 95% CI, pooled BER, and per-round BER trajectory
(ep_track_payload_hist) for the EP combos.

CPU-only default is safe but slow; GPU ok (short-ish).  Reads ε from
results/promptC_eps_bitplane.json (run practical_promptC_eps_bitplane.py first).
"""
import os
import json
import csv
import math
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception as _e:
        print(f"[GPU] set_memory_growth skipped: {_e}")

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
TROUSER_LABEL = 1
EP_SCHEDULE = [2] * 15
BP_BUDGET = sum(EP_SCHEDULE)
CKPT_MULTI = "checkpoints/denoiser.pt"
CKPT_TROUSER = "checkpoints/denoiser_trouser.pt"
SEED = 42


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; z2 = z * z; d = 1 + z2 / n
    c = (p + z2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (p, max(0.0, c - h), min(1.0, c + h))


def load_eps():
    with open("results/promptC_eps_bitplane.json") as f:
        d = json.load(f)
    return (d["denoisers"]["multi"]["eps255"],
            d["denoisers"]["trouser"]["eps255"])


def bitbank(label=None):
    import torchvision
    ds = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=False,
                                           download=True)
    imgs = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    if label is not None:
        imgs = imgs[np.array(ds.targets) == label]
    flat = imgs.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    return tf.constant(np.unpackbits(flat, axis=1), dtype=tf.int32)


def make_common():
    crc_enc = CRCEncoder("CRC24A"); crc_dec = CRCDecoder(crc_enc)
    ldpc = LDPC5GEncoder(K_PAYLOAD + crc_enc.crc_length, N_CODEWORD,
                         num_bits_per_symbol=NUM_BPS)
    mp = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    dm = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    return crc_enc, crc_dec, ldpc, mp, dm, AWGN()


def run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank, ebno, batch, rounds,
        track=False):
    no = ebnodb2no(ebno, NUM_BPS, ldpc.coderate)
    nimg = tf.shape(bank)[0]
    tot_nack = tot_be = 0; total = batch * rounds
    pr = None
    for r in range(rounds):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        idx = tf.random.uniform([batch], 0, nimg, dtype=tf.int32)
        u = tf.gather(bank, idx)
        y = awgn(mp(ldpc(crc_enc(tf.cast(u, ldpc.rdtype)))), no)
        hat = dec(dm(y, no))
        _, cv = hard_crc_decode(crc_dec, hat)
        tot_nack += batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_be += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())
        if track and getattr(dec, "last_payload_hist", None):
            step = []
            for lg in dec.last_payload_hist:
                hard = tf.cast(lg[:, :K_PAYLOAD] > 0, tf.int32)
                step.append(float(tf.reduce_mean(
                    tf.cast(tf.not_equal(u, hard), tf.float64)).numpy()))
            pr = step if pr is None else [a + b for a, b in zip(pr, step)]
    p, lo, hi = wilson_ci(tot_nack, total)
    out = {"bler": p, "ci_lo": lo, "ci_hi": hi,
           "ber": tot_be / (total * K_PAYLOAD), "nack": tot_nack, "total": total}
    if pr is not None:
        out["per_round_ber"] = [v / rounds for v in pr]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebno-core", type=float, nargs="+", default=[0.6, 0.5])
    ap.add_argument("--ebno-full", type=float, default=0.6,
                    help="SNR at which the α-sweep + extra combos also run")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--out-prefix", default="results/promptC_sweep")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    eps_multi, eps_tr = load_eps()
    print(f"ε: multi={eps_multi:.2f}  trouser={eps_tr:.2f}\n")
    crc_enc, crc_dec, ldpc, mp, dm, awgn = make_common()
    banks = {"fmnist": bitbank(None), "trouser": bitbank(TROUSER_LABEL)}

    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    bp30 = LDPC5GDecoder(ldpc, num_iter=BP_BUDGET, **common)
    bp100 = LDPC5GDecoder(ldpc, num_iter=100, **common)
    ep_multi = LDPC5GDecoder_soft(ldpc, bp_schedule=EP_SCHEDULE, k_payload=K_PAYLOAD,
                                  ep_mode=True, ep_update="damped_ep", ep_code_power=1.0,
                                  adaptive_sigma=True, ep_track_payload_hist=True,
                                  num_iter=BP_BUDGET, **common)
    ep_tr = LDPC5GDecoder_soft(ldpc, bp_schedule=EP_SCHEDULE, k_payload=K_PAYLOAD,
                               ep_mode=True, ep_update="damped_ep", ep_code_power=1.0,
                               adaptive_sigma=True, ep_track_payload_hist=True,
                               num_iter=BP_BUDGET, **common)
    ep_multi.denoiser.load_weights_pt(CKPT_MULTI)
    ep_tr.denoiser.load_weights_pt(CKPT_TROUSER)

    # spec: (label, kind, images, dec, sigma_post, alpha, core, track)
    SP = "eps"
    specs = [
        ("bp30_fmnist",  "bp",  "fmnist",  bp30,  None, None, True,  False),
        ("bp100_fmnist", "bp",  "fmnist",  bp100, None, None, True,  False),
        ("A",   "ep", "fmnist",  ep_multi, 3.0,       0.02, True,  True),
        ("B",   "ep", "fmnist",  ep_multi, eps_multi, 0.02, True,  True),
        ("B05", "ep", "fmnist",  ep_multi, eps_multi, 0.05, False, False),
        ("B10", "ep", "fmnist",  ep_multi, eps_multi, 0.10, False, False),
        ("bp30_trouser",  "bp", "trouser", bp30,  None, None, True,  False),
        ("bp100_trouser", "bp", "trouser", bp100, None, None, True,  False),
        ("Atr", "ep", "trouser", ep_multi, 3.0,     0.02, True,  False),
        ("C",   "ep", "trouser", ep_tr,    3.0,     0.02, True,  False),
        ("D",   "ep", "trouser", ep_tr,    eps_tr,  0.02, True,  True),
        ("D05", "ep", "trouser", ep_tr,    eps_tr,  0.05, False, False),
        ("D10", "ep", "trouser", ep_tr,    eps_tr,  0.10, False, False),
    ]

    ebnos = sorted(set(args.ebno_core) | {args.ebno_full}, reverse=True)
    results = {}
    # Incremental CSV (flush per spec) so a mid-run kill keeps partial results.
    csv_f = open(args.out_prefix + ".csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["ebno", "label", "images", "sigma_post", "alpha", "bler",
                    "ci_lo", "ci_hi", "ber", "nack", "total"])
    csv_f.flush()
    for ebno in ebnos:
        print(f"===== Eb/N0 = {ebno} dB =====")
        for (label, kind, imgs, dec, sp, alpha, core, track) in specs:
            # core specs run at every SNR; extra combos (α-sweep) only at ebno_full
            if not core and ebno != args.ebno_full:
                continue
            if kind == "ep":
                dec.ep_source_power = alpha
                dec.denoiser.sigma_post = float(sp)
            r = run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, banks[imgs],
                    ebno, args.batch, args.rounds, track=track)
            key = f"{label}@{ebno}"
            results[key] = {"label": label, "ebno": ebno, "images": imgs,
                            "sigma_post": sp, "alpha": alpha, **r}
            spd = "-" if sp is None else (f"{sp:.1f}" if sp == 3.0 else f"ε{sp:.1f}")
            print(f"  {label:14} [{imgs:7}] sp={spd:6} α={str(alpha):5} "
                  f"BLER={r['bler']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
                  f"BER={r['ber']:.4f} ({r['nack']}/{r['total']})", flush=True)
            if "per_round_ber" in r:
                pr = r["per_round_ber"]
                print(f"                 per-round BER: c0={pr[0]:.4f} "
                      f"c1={pr[1]:.4f} ... c{len(pr)-1}={pr[-1]:.4f}", flush=True)
            csv_w.writerow([ebno, label, imgs, spd, alpha, f"{r['bler']:.6g}",
                            f"{r['ci_lo']:.6g}", f"{r['ci_hi']:.6g}",
                            f"{r['ber']:.6g}", r["nack"], r["total"]])
            csv_f.flush()
        print(flush=True)

    csv_f.close()
    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"eps_multi": eps_multi, "eps_trouser": eps_tr,
                   "schedule": EP_SCHEDULE, "batch": args.batch,
                   "rounds": args.rounds, "results": results}, f, indent=2)
    print(f"saved: {args.out_prefix}.csv/.json")


if __name__ == "__main__":
    main()
