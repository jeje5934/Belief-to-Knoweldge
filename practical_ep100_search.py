"""
[Prompt-2] EP budget-100 best search — give EP the same 100-BP budget the legacy
and true-turbo searches got.  ep_mode=True, damped, sp=3.0, adaptive σ (EP's
established best config).  Also probes ep_code_power (β_ep) ∈ {1,0} on [2]×15.

Schedules (total BP ≤ 100), favouring EP's natural rhythm (many small chunks):
  [2]×15 (=budget-30 control / EP-30 best), [2]×50, [4]×25, [5]×20.
Then α_ep sweep {0.01,0.02,0.04} on the best schedule.
0.6 dB, 1024 cw, Wilson CI.  GPU recipe.  No commit.
"""
import os, json, csv, math
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
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no
from decoder import LDPC5GDecoder_soft

K = 6272; NUM_BPS = 1; CKPT = "checkpoints/denoiser.pt"; SEED = 42


def wilson(k, n, z=1.96):
    if n == 0: return (0., 0., 0.)
    p = k/n; z2 = z*z; d = 1+z2/n; c = (p+z2/(2*n))/d
    h = (z/d)*math.sqrt(p*(1-p)/n+z2/(4*n*n)); return (p, max(0., c-h), min(1., c+h))


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--ebno", type=float, default=0.6)
    ap.add_argument("--out-prefix", default="results/ep100_search"); args = ap.parse_args()
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
    ep = LDPC5GDecoder_soft(ldpc, bp_schedule=[2]*15, k_payload=K, ep_mode=True,
                            ep_update="damped_ep", adaptive_sigma=True, num_iter=100,
                            denoiser_kwargs=dict(device=DEV), **common)
    ep.denoiser.load_weights_pt(CKPT)
    E = args.ebno; no = ebnodb2no(E, NUM_BPS, ldpc.coderate)

    def run(sched, alpha, beta):
        ep.bp_schedule = sched; ep.ep_source_power = alpha; ep.ep_code_power = beta
        tn = tb = 0; tot = args.batch*args.rounds
        for r in range(args.rounds):
            tf.random.set_seed(SEED + r + int(E*1000))
            idx = tf.random.uniform([args.batch], 0, tf.shape(bank)[0], dtype=tf.int32)
            u = tf.gather(bank, idx)
            y = aw(mp(ldpc(crc(tf.cast(u, ldpc.rdtype)))), no); hat = ep(dm(y, no))
            _, cv = hard_crc_decode(crcd, hat); tn += args.batch - int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
            tb += int(tf.reduce_sum(tf.cast(tf.not_equal(u, tf.cast(hat[:, :K] > 0, tf.int32)), tf.int32)).numpy())
        p, lo, hi = wilson(tn, tot); return {"bler": p, "ci_lo": lo, "ci_hi": hi, "ber": tb/(tot*K), "nack": tn, "total": tot}

    results = {}; csv_f = open(args.out_prefix+".csv", "w", newline=""); cw = csv.writer(csv_f)
    cw.writerow(["label", "schedule", "bp_total", "alpha", "beta_ep", "bler", "ci_lo", "ci_hi", "ber", "nack", "total"]); csv_f.flush()
    def rec(label, sched, a, b):
        r = run(sched, a, b); results[label] = {"schedule": sched, "alpha": a, "beta_ep": b, **r}
        cw.writerow([label, sched, sum(sched), a, b, f"{r['bler']:.6g}", f"{r['ci_lo']:.6g}", f"{r['ci_hi']:.6g}", f"{r['ber']:.6g}", r["nack"], r["total"]]); csv_f.flush()
        print(f"  {label:22} BLER={r['bler']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] BER={r['ber']:.4f} ({r['nack']}/{r['total']})", flush=True)
        return r["bler"]
    print(f"EP-100 search @ {E}dB, {args.batch}×{args.rounds} cw, denoiser={DEV}\n", flush=True)
    print("-- β_ep probe on [2]×15 --", flush=True)
    rec("ep_[2]x15_b1_a.02", [2]*15, 0.02, 1.0)
    rec("ep_[2]x15_b0_a.02", [2]*15, 0.02, 0.0)
    print("-- schedule sweep (β_ep=1, α=0.02, budget≤100) --", flush=True)
    for sched, nm in [([2]*50, "[2]x50"), ([4]*25, "[4]x25"), ([5]*20, "[5]x20")]:
        rec(f"ep_{nm}_b1_a.02", sched, 0.02, 1.0)
    # top schedule among the budget-100 ones + [2]x15
    cands = ["ep_[2]x15_b1_a.02", "ep_[2]x50_b1_a.02", "ep_[4]x25_b1_a.02", "ep_[5]x20_b1_a.02"]
    top = min(cands, key=lambda k: results[k]["bler"]); tsched = results[top]["schedule"]
    print(f"-- α_ep sweep on best schedule {top} --", flush=True)
    for a in [0.01, 0.04]:
        nm = f"{tsched[0]}x{len(tsched)}"
        rec(f"ep_{nm}_b1_a{a}", tsched, a, 1.0)
    csv_f.close(); json.dump({"ebno": E, "results": results}, open(args.out_prefix+".json", "w"), indent=2)
    best = min(results.items(), key=lambda kv: kv[1]["bler"])
    print(f"\nEP-100 BEST: {best[0]} sched={best[1]['schedule']} α={best[1]['alpha']} β_ep={best[1]['beta_ep']} "
          f"BLER={best[1]['bler']:.4f} [{best[1]['ci_lo']:.4f},{best[1]['ci_hi']:.4f}]")
    print(f"saved: {args.out_prefix}.csv/.json")


if __name__ == "__main__":
    main()
