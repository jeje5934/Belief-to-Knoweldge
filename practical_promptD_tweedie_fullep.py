"""
Prompt D, steps 2-4 — full EP + diagonal Tweedie on the unimodal (Trouser)
domain: does the per-round explosion vanish?

Question: restricting the source to one category (Trouser) is meant to make the
tilted posterior more unimodal, so the DIAGONAL Tweedie 2nd moment
(σ²·diag ∂D/∂x̃) might finally approximate the posterior variance well enough
for full EP (α_ep=1) to survive.  The key metric is NOT damped performance but
whether the full-EP per-round BER explosion (prompt B: Trouser 0.13→0.52)
disappears when diagonal Tweedie precision is turned on.

Decoders (Trouser denoiser, Trouser images, full_ep α_ep=1):
  bp       : source OFF (ep_mode=False, α=β=0) — per-round BER should DECREASE.
  ep_sp3   : full EP, sigma_post=3.0, tweedie OFF   (reproduce prompt B explosion)
  ep_tw    : full EP, diagonal Tweedie precision ON (the key test)

CPU-only by default (stable; Tweedie adds ~5× net forwards via Hutchinson probes).

Usage:
  CUDA_VISIBLE_DEVICES="" python practical_promptD_tweedie_fullep.py \
      --ebno 0.6 --batch 64 --rounds 8
"""
import os
import argparse
import json
import csv

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
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no

from decoder import LDPC5GDecoder_soft

IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP
N_CODEWORD = 12600
NUM_BPS = 1
TROUSER_LABEL = 1
EP_SCHEDULE = [2] * 15
CKPT_TROUSER = "checkpoints/denoiser_trouser.pt"
SEED = 42
# Prompt B reference (Trouser denoiser, full EP, sigma_post=3.0):
PROMPTB_TROUSER = {"chunk0": 0.137, "chunk1": 0.521, "bler": 1.0, "ber": 0.49}


def trouser_bank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=False,
                                           download=True)
    imgs = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    imgs = imgs[np.array(ds.targets) == TROUSER_LABEL]
    flat = imgs.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    return tf.constant(np.unpackbits(flat, axis=1), dtype=tf.int32)


def make_common():
    crc_enc = CRCEncoder("CRC24A"); crc_dec = CRCDecoder(crc_enc)
    ldpc = LDPC5GEncoder(K_PAYLOAD + crc_enc.crc_length, N_CODEWORD,
                         num_bits_per_symbol=NUM_BPS)
    mp = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    dm = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    return crc_enc, crc_dec, ldpc, mp, dm, AWGN()


def per_round(dec, u):
    out = []
    for lg in dec.last_payload_hist:
        hard = tf.cast(lg[:, :K_PAYLOAD] > 0, tf.int32)
        out.append(float(tf.reduce_mean(
            tf.cast(tf.not_equal(u, hard), tf.float64)).numpy()))
    return out


def run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank, ebno, batch, rounds):
    no = ebnodb2no(ebno, NUM_BPS, ldpc.coderate)
    nimg = tf.shape(bank)[0]
    tot_nack = tot_be = 0; total = batch * rounds; pr = None
    for r in range(rounds):
        tf.random.set_seed(SEED + r + int(ebno * 1000))
        idx = tf.random.uniform([batch], 0, nimg, dtype=tf.int32)
        u = tf.gather(bank, idx)
        y = awgn(mp(ldpc(crc_enc(tf.cast(u, ldpc.rdtype)))), no)
        hat = dec(dm(y, no))
        _, cv = crc_dec(hat)
        tot_nack += batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_be += int(tf.reduce_sum(tf.cast(
            tf.not_equal(u, tf.cast(hat[:, :K_PAYLOAD] > 0, tf.int32)),
            tf.int32)).numpy())
        step = per_round(dec, u)
        pr = step if pr is None else [a + b for a, b in zip(pr, step)]
    return {"bler": tot_nack / total, "ber": tot_be / (total * K_PAYLOAD),
            "nack": tot_nack, "total": total,
            "per_round_ber": [v / rounds for v in pr]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebno", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out-prefix", default="results/promptD_tweedie_fullep")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    crc_enc, crc_dec, ldpc, mp, dm, awgn = make_common()
    bank = trouser_bank()
    print(f"Trouser images={int(tf.shape(bank)[0])}, Eb/N0={args.ebno} dB, "
          f"{args.batch}×{args.rounds} cw\n")

    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    bp = LDPC5GDecoder_soft(ldpc, bp_schedule=EP_SCHEDULE, k_payload=K_PAYLOAD,
                            ep_mode=False, alpha=0.0, beta=0.0, adaptive_sigma=True,
                            ep_track_payload_hist=True, num_iter=sum(EP_SCHEDULE),
                            **common)
    ep = LDPC5GDecoder_soft(ldpc, bp_schedule=EP_SCHEDULE, k_payload=K_PAYLOAD,
                            ep_mode=True, ep_update="full_ep", adaptive_sigma=True,
                            ep_track_payload_hist=True, num_iter=sum(EP_SCHEDULE),
                            **common)
    for d in (bp, ep):
        d.denoiser.load_weights_pt(CKPT_TROUSER)

    results = {}
    specs = [("bp", bp, False), ("ep_sp3", ep, False), ("ep_tw", ep, True)]
    for name, dec, tw in specs:
        if name.startswith("ep"):
            dec.denoiser.tweedie_precision = bool(tw)
        r = run(dec, ldpc, crc_enc, crc_dec, mp, dm, awgn, bank,
                args.ebno, args.batch, args.rounds)
        results[name] = {"tweedie": tw, **r}
        pr = r["per_round_ber"]
        tag = ("source OFF" if name == "bp" else
               ("full EP + sigma_post=3.0 (Tweedie off)" if not tw else
                "full EP + diagonal Tweedie ON"))
        print(f"[{name}] {tag}")
        print(f"    BLER={r['bler']:.4f}  BER={r['ber']:.4f}  "
              f"({r['nack']}/{r['total']} NACK)")
        print(f"    per-round BER: c0={pr[0]:.4f}  c1={pr[1]:.4f}  "
              f"... c{len(pr)-1}={pr[-1]:.4f}  |  c1/c0={pr[1]/max(pr[0],1e-9):.2f}×")
        print(f"    trajectory: {[round(v,4) for v in pr]}\n")

    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"ebno": args.ebno, "promptB_trouser_ref": PROMPTB_TROUSER,
                   "results": results}, f, indent=2)
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["decoder", "chunk", "per_round_ber"])
        for name in results:
            for c, v in enumerate(results[name]["per_round_ber"]):
                w.writerow([name, c, f"{v:.6g}"])
    _plot(results, args.ebno, args.out_prefix + ".png")
    print(f"saved: {args.out_prefix}.png/.csv/.json")
    print(f"prompt-B Trouser ref (sigma_post=3.0): c0={PROMPTB_TROUSER['chunk0']} "
          f"-> c1={PROMPTB_TROUSER['chunk1']} (explosion)")


def _plot(results, ebno, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = {"bp": "source OFF (pure BP)",
           "ep_sp3": "full EP + sigma_post=3.0 (Tweedie off)",
           "ep_tw": "full EP + diagonal Tweedie ON"}
    sty = {"bp": "s--", "ep_sp3": "o-", "ep_tw": "^-"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("bp", "ep_sp3", "ep_tw"):
        pr = results[name]["per_round_ber"]
        ax.plot(range(len(pr)), pr, sty[name], label=lab[name], lw=2, ms=6)
    ax.set_xlabel("chunk (round); source from chunk 1")
    ax.set_ylabel("per-round payload BER")
    ax.set_title(f"Prompt D: full EP + diagonal Tweedie on Trouser ({ebno} dB)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
