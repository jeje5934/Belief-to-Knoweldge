"""
altproj (interpretation 1) knob screening — staged.  Implementation is verified;
this SCREENS knobs (δ, ρ, sigma_post, schedule, warm) at AWGN 0.6 dB.

TF on CPU (safe), torch denoiser on GPU (fast).  One decoder is built once
(weights loaded once); knobs are flipped per config.

Metrics per config (batch-averaged over cw):
  BLER (CRC nack / cw), and per-round trajectories syn_wt, mean|C|, mean|P|.
4-type classification (auditable — raw syn stats printed alongside):
  converge : syn_final low & stays low (syndrome satisfied at the end)
  collapse : syn dips to a low min then rises back (V3 shape) — min round recorded
  stuck    : syn stays high / flat (never resolves)
  runaway  : mean|C| reaches the ±25 clip

Usage:  python altproj_screen.py --stage 1 --cw 256 [--out results/stage1.json]
        python altproj_screen.py --configs '<json list>' --cw 1024 --out ...
"""
import os, json, time, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")          # TF -> CPU (safe)

from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no
from decoder import LDPC5GDecoder_soft

IMG_H = IMG_W = 28; BPP = 8
K = IMG_H * IMG_W * BPP; N = 12600; NBPS = 1
CKPT = "checkpoints/denoiser.pt"
DEV = "cuda"                                        # denoiser on GPU
EBNO = 0.6
BATCH = 128                                         # per-decode batch; cw = BATCH*rounds
SEED = 4242
CLIP = 25.0
COMMON = dict(cn_update="boxplus-phi", vn_update="sum", cn_schedule="flooding",
              hard_out=False, return_infobits=True, llr_max=30.0)

# classification thresholds (on batch-averaged syn_wt; n_checks ~ 6304)
SYN_CONVERGE = 100.0        # syn_final below this ⇒ resolved
SYN_DIP = 700.0            # syn_min below this ⇒ "got a low point"
COLLAPSE_RISE = 2.0        # syn_final > COLLAPSE_RISE * syn_min ⇒ rose back
CLIP_NEAR = 24.0           # mean|C| above ⇒ runaway


def bank():
    import torchvision
    ds = torchvision.datasets.FashionMNIST("/tmp/fmnist", train=False, download=True)
    flat = ds.data.numpy().astype(np.uint8).reshape(-1, IMG_H * IMG_W)
    return tf.constant(np.unpackbits(flat, axis=1), tf.int32)


def make():
    crc = CRCEncoder("CRC24A"); crcd = CRCDecoder(crc)
    ldpc = LDPC5GEncoder(K + crc.crc_length, N, num_bits_per_symbol=NBPS)
    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NBPS)
    demapper = Demapper("app", constellation_type="pam", num_bits_per_symbol=NBPS)
    return crc, crcd, ldpc, mapper, demapper, AWGN()


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (p, max(0.0, c - h), min(1.0, c + h))


def classify(syn, cmax):
    """syn: list of batch-averaged syn_wt per round. cmax: max mean|C|."""
    syn = list(syn)
    syn_min = min(syn); syn_min_round = int(np.argmin(syn))
    syn_final = syn[-1]; syn_start = syn[0]
    if cmax >= CLIP_NEAR:
        t = "runaway"
    elif syn_final <= SYN_CONVERGE:
        t = "converge"
    elif syn_min <= SYN_DIP and syn_final > COLLAPSE_RISE * max(syn_min, 1.0):
        t = "collapse"
    else:
        t = "stuck"
    return dict(type=t, syn_min=syn_min, syn_min_round=syn_min_round,
                syn_final=syn_final, syn_start=syn_start)


def run_config(dec, cfg, ldpc, crc, crcd, mapper, demapper, awgn, bk, cw):
    """Apply knobs, decode cw codewords (BATCH per round), return metrics."""
    dec.altproj = True
    dec.altproj_delta = cfg["delta"]
    dec.altproj_rho = cfg["rho"]
    dec.altproj_sigma_den = cfg.get("sigma_den", 0.3)
    dec.altproj_warm_start = cfg.get("warm", True)
    dec.denoiser.sigma_post = cfg.get("sigma_post", 3.0)
    dec.bp_schedule = cfg["schedule"]
    # structural improvements (always set — decoder state persists across configs)
    # early-stop criterion: "off" | "syn" (syndrome — NOT sufficient) | "crc"
    es_mode = cfg.get("es_mode", "off")
    dec.altproj_early_stop = es_mode != "off"
    dec.altproj_es_patience = cfg.get("es_patience", 0)
    dec.altproj_crc_check = (
        (lambda ul: hard_crc_decode(crcd, tf.reshape(ul, [-1, ldpc.k]))[1]) if es_mode == "crc"
        else None)
    dec.altproj_delta_schedule = cfg.get("delta_sched")
    dec.altproj_rho_schedule = cfg.get("rho_sched")

    no = ebnodb2no(EBNO, NBPS, ldpc.coderate)
    n_imgs = tf.shape(bk)[0]
    rounds = max(1, cw // BATCH)
    tot_nack = 0; tot = 0
    # ragged-safe: early-stop may break at a different round per batch.
    acc = {"syn": {}, "C": {}, "P": {}}
    rounds_used = []
    for r in range(rounds):
        tf.random.set_seed(SEED + r)
        idx = tf.random.uniform([BATCH], 0, n_imgs, dtype=tf.int32)
        u = tf.gather(bk, idx)
        y = awgn(mapper(ldpc(crc(tf.cast(u, ldpc.rdtype)))), no)
        llr = demapper(y, no)
        hat = dec(llr)
        _, cv = hard_crc_decode(crcd, hat)
        ack = int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_nack += BATCH - ack; tot += BATCH
        diag = dec.last_altproj_diag
        rounds_used.append(len(diag))
        for i, d in enumerate(diag):
            acc["syn"].setdefault(i, []).append(d["syndrome_weight"])
            acc["C"].setdefault(i, []).append(d["mean_abs_C"])
            acc["P"].setdefault(i, []).append(d["mean_abs_P"])
    avg = lambda k: [float(np.mean(acc[k][i])) for i in sorted(acc[k])]
    traj_syn = np.array(avg("syn")); traj_C = np.array(avg("C")); traj_P = np.array(avg("P"))
    bler, lo, hi = wilson_ci(tot_nack, tot)
    cls = classify(traj_syn.tolist(), float(traj_C.max()))
    # BP iters actually consumed (early-stop compute saving): schedule[:rounds_used]
    sch = cfg["schedule"]
    bp_iters = float(np.mean([sum(sch[:n]) for n in rounds_used]))
    return dict(cfg=cfg, cw=tot, bler=bler, ci_lo=lo, ci_hi=hi, n_nack=tot_nack,
                traj_syn=[round(x, 1) for x in traj_syn.tolist()],
                traj_C=[round(x, 3) for x in traj_C.tolist()],
                traj_P=[round(x, 2) for x in traj_P.tolist()],
                rounds_used=float(np.mean(rounds_used)), bp_iters=bp_iters,
                c_max=float(traj_C.max()), **cls)


def stage1_configs():
    cfgs = []
    for d in (0.02, 0.05, 0.1, 0.3):
        for rho in (1.0, 0.9, 0.7, 0.5):
            cfgs.append(dict(delta=d, rho=rho, sigma_post=3.0,
                             schedule=[5] * 20, warm=True, tag=f"d{d}_r{rho}"))
    return cfgs


# top (δ,ρ) from stage 1 (256cw): converge, lowest BLER.
TOP_DR = [(0.02, 0.9), (0.05, 0.7), (0.02, 0.5)]


def stage2_configs():
    cfgs = []
    for d, rho in TOP_DR:
        for sp in (3.0, 6.0, 12.0):
            cfgs.append(dict(delta=d, rho=rho, sigma_post=sp,
                             schedule=[5] * 20, warm=True,
                             tag=f"d{d}_r{rho}_sp{sp:g}"))
    return cfgs


def lin(a, b, T):
    return [float(a + (b - a) * i / max(T - 1, 1)) for i in range(T)]


def geom(a, b, T):
    return [float(a * (b / a) ** (i / max(T - 1, 1))) for i in range(T)]


def stage5_configs():
    """[improvement 1] early-stop across the sweetspot, the collapse region (ρ=1,
    δ≥0.05) and larger δ.  Patience sweep: 0 = syn=0 freeze only (strictly-safe
    subset), 2/3/5 = confirmed-degradation freeze.  Does early-stop rescue collapse
    and make a bigger step safe?"""
    S20 = [5] * 20
    pts = [(0.02, 0.9), (0.05, 1.0), (0.1, 1.0), (0.1, 0.9), (0.2, 1.0), (0.3, 1.0)]
    cfgs = []
    for d, rho in pts:
        for es in ("off", "syn", "crc"):
            cfgs.append(dict(delta=d, rho=rho, sigma_post=3.0, schedule=S20,
                             warm=True, es_mode=es, tag=f"d{d}_r{rho}_ES-{es}"))
    return cfgs


def stage6_configs():
    """[improvement 2] time-varying δ/ρ: aggressive early (big δ, ρ=1 → fast
    resolve) → conservative late (small δ, ρ<1 → hold + forget).  linear + geometric."""
    S20 = [5] * 20; T = 20
    variants = [
        ("lin_d.1-.02_r1-.9", lin(0.1, 0.02, T), lin(1.0, 0.9, T)),
        ("geo_d.1-.02_r1-.9", geom(0.1, 0.02, T), geom(1.0, 0.9, T)),
        ("lin_d.3-.02_r1-.7", lin(0.3, 0.02, T), lin(1.0, 0.7, T)),
        ("geo_d.3-.02_r1-.7", geom(0.3, 0.02, T), geom(1.0, 0.7, T)),
    ]
    cfgs = []
    for name, ds, rs in variants:
        for es in ("off", "crc"):
            cfgs.append(dict(delta=0.02, rho=0.9, sigma_post=3.0, schedule=S20,
                             warm=True, es_mode=es,
                             delta_sched=ds, rho_sched=rs,
                             tag=f"{name}_ES-{es}"))
    # sweetspot reference under the same run
    for es in ("off", "crc"):
        cfgs.append(dict(delta=0.02, rho=0.9, sigma_post=3.0, schedule=S20,
                         warm=True, es_mode=es, tag=f"REF_sweetspot_ES-{es}"))
    return cfgs


def run_anchor(decoders, cfg, ldpc, crc, crcd, mapper, demapper, awgn, bk, cw):
    """Anchor decoders (bp100 / legacy) under identical seeds — no altproj traj."""
    dec = decoders[cfg["mode"]]
    no = ebnodb2no(EBNO, NBPS, ldpc.coderate)
    n_imgs = tf.shape(bk)[0]
    rounds = max(1, cw // BATCH)
    tot_nack = 0; tot = 0
    for r in range(rounds):
        tf.random.set_seed(SEED + r)
        idx = tf.random.uniform([BATCH], 0, n_imgs, dtype=tf.int32)
        u = tf.gather(bk, idx)
        y = awgn(mapper(ldpc(crc(tf.cast(u, ldpc.rdtype)))), no)
        llr = demapper(y, no)
        hat = dec(llr)
        _, cv = hard_crc_decode(crcd, hat)
        ack = int(tf.reduce_sum(tf.cast(cv, tf.int32)).numpy())
        tot_nack += BATCH - ack; tot += BATCH
    bler, lo, hi = wilson_ci(tot_nack, tot)
    return dict(cfg=cfg, cw=tot, bler=bler, ci_lo=lo, ci_hi=hi, n_nack=tot_nack,
                type="anchor", syn_min=0, syn_min_round=0, syn_final=0,
                syn_start=0, c_max=0.0, traj_syn=[], traj_C=[], traj_P=[])


def stage4_configs():
    S20 = [5] * 20; S10 = [10] * 10
    A = [
        dict(delta=0.02, rho=0.9, sigma_post=3.0, schedule=S20, warm=True, tag="d0.02_r0.9_[5]x20"),
        dict(delta=0.02, rho=0.9, sigma_post=3.0, schedule=S10, warm=True, tag="d0.02_r0.9_[10]x10"),
        dict(delta=0.05, rho=0.7, sigma_post=3.0, schedule=S20, warm=True, tag="d0.05_r0.7_[5]x20"),
        dict(delta=0.05, rho=0.7, sigma_post=3.0, schedule=S10, warm=True, tag="d0.05_r0.7_[10]x10"),
        dict(delta=0.02, rho=0.5, sigma_post=3.0, schedule=S20, warm=True, tag="d0.02_r0.5_[5]x20"),
    ]
    anchors = [
        dict(mode="bp100", tag="ANCHOR_bp100"),
        dict(mode="legacy", tag="ANCHOR_legacy[5]x20"),
    ]
    return A + anchors


def stage3_configs():
    # top (δ,ρ,sigma_post) from stage 2 filled in after that stage runs.
    top = [(0.02, 0.9, 3.0), (0.05, 0.7, 3.0)]
    scheds = {"[5]x20": [5] * 20, "[10]x10": [10] * 10, "[2]x50": [2] * 50}
    cfgs = []
    for d, rho, sp in top:
        for sname, sch in scheds.items():
            for warm in (True, False):
                cfgs.append(dict(delta=d, rho=rho, sigma_post=sp, schedule=sch,
                                 warm=warm,
                                 tag=f"d{d}_r{rho}_{sname}_{'on' if warm else 'off'}"))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--configs", type=str, default=None,
                    help="JSON list of config dicts (overrides --stage)")
    ap.add_argument("--cw", type=int, default=256)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    crc, crcd, ldpc, mapper, demapper, awgn = make()
    bk = bank()
    dec = LDPC5GDecoder_soft(ldpc, k_payload=K, num_iter=100,
                             bp_schedule=[5] * 20, adaptive_sigma=False,
                             denoiser_kwargs=dict(device=DEV), **COMMON)
    dec.denoiser.load_weights_pt(CKPT)
    dec.denoiser.sigma = 0.3

    # anchor decoders (built lazily only if a config needs them).
    decoders = {}

    def ensure_anchors():
        if decoders:
            return
        bp100 = LDPC5GDecoder(ldpc, num_iter=100, **COMMON)
        legacy = LDPC5GDecoder_soft(
            ldpc, k_payload=K, num_iter=100, bp_schedule=[5] * 20,
            ep_mode=False, source_input_mode="bp_post", adaptive_sigma=False,
            denoiser_kwargs=dict(device=DEV), **COMMON)
        legacy.denoiser.load_weights_pt(CKPT)
        legacy.denoiser.sigma = 0.3; legacy.alpha = 0.15; legacy.beta = 0.15
        decoders["bp100"] = bp100; decoders["legacy"] = legacy

    if a.configs:
        cfgs = json.loads(a.configs)
    elif a.stage == 1:
        cfgs = stage1_configs()
    elif a.stage == 2:
        cfgs = stage2_configs()
    elif a.stage == 3:
        cfgs = stage3_configs()
    elif a.stage == 4:
        cfgs = stage4_configs()
    elif a.stage == 5:
        cfgs = stage5_configs()
    elif a.stage == 6:
        cfgs = stage6_configs()
    else:
        raise SystemExit("give --stage 1|2|3|4|5|6 or --configs")

    print(f"# altproj screen: {len(cfgs)} configs, cw={a.cw}, EbN0={EBNO}dB, DEV={DEV}")
    print(f"# {'tag':<22} {'BLER':>8} {'[95% CI]':>17} {'type':>9} "
          f"{'synMin@rnd':>11} {'synFin':>7} {'|C|max':>7} {'rnds':>5} {'bpIt':>5}  {'sec':>6}")
    results = []
    for cfg in cfgs:
        t0 = time.time()
        if cfg.get("mode") in ("bp100", "legacy"):
            ensure_anchors()
            r = run_anchor(decoders, cfg, ldpc, crc, crcd, mapper, demapper, awgn, bk, a.cw)
        else:
            r = run_config(dec, cfg, ldpc, crc, crcd, mapper, demapper, awgn, bk, a.cw)
        dt = time.time() - t0
        r["sec"] = round(dt, 1)
        results.append(r)
        tag = cfg.get("tag", "")
        print(f"  {tag:<22} {r['bler']:>8.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
              f"{r['type']:>9} {r['syn_min']:>6.0f}@{r['syn_min_round']:<3d} "
              f"{r['syn_final']:>7.0f} {r['c_max']:>7.2f} "
              f"{r.get('rounds_used', 0):>5.1f} {r.get('bp_iters', 0):>5.0f}  "
              f"{dt:>6.1f}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"# wrote {a.out}")


if __name__ == "__main__":
    main()
