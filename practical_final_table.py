"""
[Prompt-3] Final comparison table — one channel realisation, GPU, 3200 cw,
Wilson CI, 0.6 dB (top configs extended to 0.5/0.7).

Rows (each at its own best config; note σ differs: EP uses adaptive σ, the turbo
family uses fixed σ=0.3 — each is its own best, flagged in the report):
  BP-30, BP-100                       references / same-budget & ceiling
  legacy [2]×15  α=β=0.1  (budget 30) legacy turbo best
  legacy [2]×15  α=0.1 β=0            legacy, β=0
  legacy [5]×20  α=β=0.1  (budget100) FAIR budget-100 legacy (new)
  EP     [2]×15  α=0.02   (budget 30) EP budget-30 best
  EP     [5]×20  α=0.01   (budget100) EP budget-100 best  ← Prompt-2 winner
  true_turbo 33×3 α=0.05  (budget 99) true-turbo best
  minus  [2]×15  α=0.1 β=0            purity minus best

GPU recipe (TF on CPU, torch denoiser on GPU).  No commit here.
"""
import os, json, csv, math, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no
from decoder import LDPC5GDecoder_soft

K = 6272; NUM_BPS = 1; CKPT = "checkpoints/denoiser.pt"; SEED = 42


def wilson(k, n, z=1.96):
    if n == 0: return (0., 0., 0.)
    p = k/n; z2 = z*z; d = 1+z2/n; c = (p+z2/(2*n))/d
    h = (z/d)*math.sqrt(p*(1-p)/n+z2/(4*n*n)); return (p, max(0., c-h), min(1., c+h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebnos", type=float, nargs="+", default=[0.6, 0.5, 0.7])
    ap.add_argument("--top3-ebnos", type=float, nargs="+", default=[0.5, 0.7])
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--out-prefix", default="results/final_table"); args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    ldpc = LDPC5GEncoder(K+crc.crc_length, 12600, num_bits_per_symbol=NUM_BPS)
    mp = Mapper("pam", num_bits_per_symbol=NUM_BPS); dm = Demapper("app", "pam", num_bits_per_symbol=NUM_BPS); aw = AWGN()
    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    imgs = np.array([np.array(im) for im, _ in ds], dtype=np.uint8)
    bank = tf.constant(np.unpackbits(imgs.reshape(-1, 784).astype(np.uint8), axis=1), tf.int32)
    common = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
                  hard_out=False, return_infobits=True, llr_max=30.0)
    bp30 = LDPC5GDecoder(ldpc, num_iter=30, **common)
    bp100 = LDPC5GDecoder(ldpc, num_iter=100, **common)

    def soft(**kw):
        d = LDPC5GDecoder_soft(ldpc, k_payload=K, num_iter=100,
                               denoiser_kwargs=dict(device=DEV), **kw, **common)
        d.denoiser.load_weights_pt(CKPT); d.denoiser.sigma = 0.3
        return d
    # EP decoders (adaptive σ)
    ep30 = soft(bp_schedule=[2]*15, ep_mode=True, ep_update="damped_ep", ep_source_power=0.02, ep_code_power=1.0, adaptive_sigma=True)
    ep100 = soft(bp_schedule=[5]*20, ep_mode=True, ep_update="damped_ep", ep_source_power=0.01, ep_code_power=1.0, adaptive_sigma=True)
    # legacy / turbo family (fixed σ=0.3)
    lg30 = soft(bp_schedule=[2]*15, ep_mode=False, source_input_mode="bp_post", adaptive_sigma=False)
    lg100 = soft(bp_schedule=[5]*20, ep_mode=False, source_input_mode="bp_post", adaptive_sigma=False)
    mn30 = soft(bp_schedule=[2]*15, ep_mode=False, source_input_mode="minus_source_feedback", adaptive_sigma=False)
    tt = soft(bp_schedule=[33]*3, true_turbo=True, adaptive_sigma=False)

    # (label, decoder, setter, top3)
    def setab(a, b):
        return lambda d: (setattr(d, "alpha", a), setattr(d, "beta", b))
    ROWS = [
        ("BP-30",              bp30,  None, False),
        ("BP-100",             bp100, None, True),
        ("legacy_[2]x15_a.1b.1", lg30,  setab(0.1, 0.1), True),
        ("legacy_[2]x15_a.1b0",  lg30,  setab(0.1, 0.0), False),
        ("legacy_[5]x20_a.1b.1", lg100, setab(0.1, 0.1), True),
        ("EP_[2]x15_a.02",       ep30,  None, False),
        ("EP_[5]x20_a.01",       ep100, None, True),
        ("true_turbo_33x3_a.05", tt,    setab(0.05, 0.0), False),
        ("minus_[2]x15_a.1b0",   mn30,  setab(0.1, 0.0), False),
    ]

    def run(dec, ebno):
        no = ebnodb2no(ebno, NUM_BPS, ldpc.coderate); tn = tb = 0; tot = args.batch*args.rounds
        for r in range(args.rounds):
            tf.random.set_seed(SEED + r + int(ebno*1000))
            idx = tf.random.uniform([args.batch], 0, tf.shape(bank)[0], dtype=tf.int32)
            u = tf.gather(bank, idx); y = aw(mp(ldpc(crc(tf.cast(u, ldpc.rdtype)))), no)
            hat = dec(dm(y, no)); _, cv = hard_crc_decode(crcd, hat)
            tn += args.batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
            tb += int(tf.reduce_sum(tf.cast(tf.not_equal(u, tf.cast(hat[:, :K] > 0, tf.int32)), tf.int32)).numpy())
        p, lo, hi = wilson(tn, tot); return {"bler": p, "ci_lo": lo, "ci_hi": hi, "ber": tb/(tot*K), "nack": tn, "total": tot}

    results = {}
    csv_f = open(args.out_prefix+".csv", "w", newline=""); cw = csv.writer(csv_f)
    cw.writerow(["ebno", "label", "bler", "ci_lo", "ci_hi", "ber", "nack", "total"]); csv_f.flush()
    for ebno in args.ebnos:
        print(f"===== {ebno} dB =====", flush=True)
        for label, dec, setter, top3 in ROWS:
            if ebno in args.top3_ebnos and not top3:
                continue
            if setter: setter(dec)
            r = run(dec, ebno); results[f"{label}@{ebno}"] = {"label": label, "ebno": ebno, **r}
            cw.writerow([ebno, label, f"{r['bler']:.6g}", f"{r['ci_lo']:.6g}", f"{r['ci_hi']:.6g}", f"{r['ber']:.6g}", r["nack"], r["total"]]); csv_f.flush()
            print(f"  {label:22} BLER={r['bler']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] BER={r['ber']:.4f} ({r['nack']}/{r['total']})", flush=True)
        print(flush=True)
    csv_f.close(); json.dump(results, open(args.out_prefix+".json", "w"), indent=2)
    print(f"saved: {args.out_prefix}.csv/.json")


if __name__ == "__main__":
    main()
