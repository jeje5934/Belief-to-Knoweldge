"""
Offline calibration of the adaptive sigma lookup table.

Calibration objectives
----------------------
* ``tail_final_nack`` — *recommended* for final decoding.  After applying
  candidate σ at the calibration chunk, the remaining BP chunks are run
  with denoiser σ = ``--trace-tail-sigma``, and final CRC NACK is the score.
* ``tail_final_ber``  — same surrogate but optimises bit error rate of the
  final decoder output.
* ``source_posterior_ber`` — proxy: BER on payload bits after a single
  denoiser step on BP_post, before remaining BP chunks.  Cheap and
  illustrative but not a faithful surrogate of final BLER.

Lookup format produced
----------------------
* By default the calibrator now produces a **per-chunk** lookup, because
  the syndrome-ratio distribution differs across BP chunks.  Disable with
  ``--no-per-chunk`` to fall back to a single global lookup.
* If ``--monotonic-sigma`` is passed, the per-bin σ is post-processed so
  larger ratio bins do not map to a smaller σ.
* The default sigma range is the narrow (0.20, 0.40) band that surrounds
  the strong fixed σ=0.3 baseline.  Wider ranges (e.g. 0.05–0.50) are
  diagnostic-only.

Output JSON schema (v2)
-----------------------
    schema_version, method, objective, alpha, beta,
    syndrome_hard_decision,
    schedule, ebno_list, batch, rounds,
    collection_sigma, trace_tail_sigma,
    candidate_sigmas, sigma_min, sigma_max,
    binning_type, monotonic_sigma, per_chunk,
    samples_per_bin, candidate_objective_values, selected_sigma,
    thresholds, sigma_levels                       (global lookup),
    per_chunk_lookup                                (per-chunk lookup),
    objective_notes
"""
from __future__ import annotations

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from cli_common import early_gpu_mb_argv

_mb = early_gpu_mb_argv(sys.argv)
if _mb is not None and _mb > 0:
    os.environ["FMNIST_GPU_MEM_MB"] = str(_mb)

import argparse
import gc
import json

import numpy as np
import tensorflow as tf
from gpu_limits import setup_tensorflow_gpu

setup_tensorflow_gpu()

import torchvision

from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from crc_utils import hard_crc_decode
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPCBPDecoder
from sionna.phy.channel.awgn import AWGN
from sionna.phy.utils import ebnodb2no

from decoder import LDPC5GDecoder_soft
from syndrome_sigma_schedule import (
    DEFAULT_MAIN_CANDIDATE_SIGMAS,
    DEFAULT_MAIN_SIGMA_MAX,
    DEFAULT_MAIN_SIGMA_MIN,
    DEFAULT_SIGMA_LEVELS,
    DEFAULT_SYNDROME_THRESHOLDS,
    SIGMA_LOOKUP_SCHEMA_VERSION,
    compute_syndrome_weight_and_ratio,
    enforce_monotonic_sigma,
    sigma_from_syndrome_ratio,
)


IMG_H, IMG_W, BPP = 28, 28, 8
K_PAYLOAD = IMG_H * IMG_W * BPP
CRC_DEGREE = "CRC24A"
N_CODEWORD = 12600
NUM_BPS = 1
SCHEDULE = [10, 10, 10]
SEED = 42
CKPT = "checkpoints/denoiser.pt"


def build_test_bitbank():
    ds = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist", train=False, download=True)
    images = np.array([np.array(img) for img, _ in ds], dtype=np.uint8)
    flat = images.reshape(-1, IMG_H * IMG_W).astype(np.uint8)
    bits = np.unpackbits(flat, axis=1)
    return tf.constant(bits, dtype=tf.int32)


def _copy_state(msg_v2c):
    return tf.convert_to_tensor(msg_v2c.numpy())


def run_tail_from_chunk(
    dec, crc_dec,
    payload0, crc_and_rest, z_short, x2_par,
    payload_intr, curr_msg_v2c, chunk_after,
    post_payload, bp_ext, u_bits,
    a, b, cand_sigma, tail_sigma,
):
    """Apply denoiser at chunk_after-1 with cand_sigma, then finish BP using
    tail_sigma on later denoiser chunks. Returns (bit_err_vec[B], nack_vec[B]).
    """
    src_ext = dec._denoiser(post_payload, sigma=float(cand_sigma))
    payload_intr_n = payload0 + b * bp_ext + a * src_ext
    msg = _copy_state(curr_msg_v2c)

    k_pl = int(u_bits.shape[1])
    for ci in range(chunk_after, len(SCHEDULE)):
        x1_stage = tf.concat([payload_intr_n, crc_and_rest], axis=1)
        llr_bp = tf.concat([x1_stage, z_short, x2_par], axis=1)
        x_hat, msg = LDPCBPDecoder.call(
            dec, llr_bp, num_iter=int(SCHEDULE[ci]), msg_v2c=msg)
        if ci < len(SCHEDULE) - 1:
            post2 = x_hat[:, :k_pl]
            bp_ext2 = post2 - payload_intr_n
            src2 = dec._denoiser(post2, sigma=float(tail_sigma))
            payload_intr_n = payload0 + b * bp_ext2 + a * src2

    hat_info = x_hat[:, :int(dec.encoder.k)]
    _, cv = hard_crc_decode(crc_dec, hat_info)
    nack = tf.cast(tf.logical_not(cv), tf.float32)
    hard = tf.cast(hat_info[:, :u_bits.shape[1]] > 0.0, dec.rdtype)
    bit_err = tf.reduce_sum(tf.cast(tf.not_equal(hard, u_bits), tf.float32), axis=1)
    return bit_err, nack


def collect_samples(args):
    """Run the calibration loop, yielding per-chunk per-sample candidate metrics."""
    crc_enc = CRCEncoder(CRC_DEGREE)
    crc_dec = CRCDecoder(crc_enc)
    k_ldpc = K_PAYLOAD + crc_enc.crc_length
    ldpc_enc = LDPC5GEncoder(k_ldpc, N_CODEWORD, num_bits_per_symbol=NUM_BPS)
    mapper = Mapper(constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    demapper = Demapper("app", constellation_type="pam", num_bits_per_symbol=NUM_BPS)
    awgn = AWGN()
    bit_bank = build_test_bitbank()

    dec = LDPC5GDecoder_soft(
        ldpc_enc, bp_schedule=SCHEDULE,
        alpha=args.alpha, beta=args.beta, k_payload=K_PAYLOAD,
        cn_update="boxplus-phi", vn_update="sum",
        cn_schedule="flooding", hard_out=False, return_infobits=True,
        num_iter=sum(SCHEDULE), llr_max=30.0,
        adaptive_sigma=False,
        syndrome_thresholds=tuple(args.hand_thresholds),
        sigma_levels=tuple(args.hand_sigmas),
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        syndrome_hard_decision=args.syndrome_hard_decision,
        syndrome_sign_debug=False,
    )
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    dec.denoiser.load_weights_pt(args.ckpt)
    dec.denoiser.sigma = args.collection_sigma

    sample_cache = []

    for eb in args.ebno:
        no = ebnodb2no(eb, NUM_BPS, ldpc_enc.coderate)
        for r in range(args.rounds):
            tf.random.set_seed(SEED + r + int(eb * 1000))
            n_imgs = tf.shape(bit_bank)[0]
            idx = tf.random.uniform([args.batch], 0, n_imgs, dtype=tf.int32)
            u = tf.gather(bit_bank, idx)
            u_crc = crc_enc(tf.cast(u, ldpc_enc.rdtype))
            c = ldpc_enc(u_crc)
            x = mapper(c)
            y = awgn(x, no)
            llr_ch = demapper(y, no)

            llr = tf.reshape(llr_ch, [-1, dec.encoder.n])
            B = tf.shape(llr)[0]
            if dec.encoder.num_bits_per_symbol is not None:
                llr = tf.gather(llr, dec.encoder.out_int_inv, axis=-1)
            llr_5g = tf.concat(
                [tf.zeros([B, 2 * dec.encoder.z], dec.rdtype), llr], axis=1)
            k_filler = dec.encoder.k_ldpc - dec.encoder.k
            nb_punc_bits = (
                (dec.encoder.n_ldpc - k_filler) - dec.encoder.n - 2 * dec.encoder.z)
            llr_5g = tf.concat(
                [llr_5g,
                 tf.zeros([B, nb_punc_bits - dec._nb_pruned_nodes], dec.rdtype)],
                axis=1)

            x1_sys = llr_5g[:, :dec.encoder.k]
            nb_par_bits = (
                dec.encoder.n_ldpc - k_filler - dec.encoder.k - dec._nb_pruned_nodes)
            x2_par = llr_5g[:, dec.encoder.k:dec.encoder.k + nb_par_bits]
            z_short = -tf.cast(dec._llr_max, dec.rdtype) * tf.ones(
                [B, k_filler], dec.rdtype)

            k_payload = min(K_PAYLOAD, int(dec.encoder.k))
            payload0 = x1_sys[:, :k_payload]
            crc_and_rest = x1_sys[:, k_payload:]
            payload_intr = payload0
            curr_msg_v2c = None

            a = tf.cast(args.alpha, dec.rdtype)
            b = tf.cast(args.beta, dec.rdtype)
            u_bits = tf.cast(u[:, :k_payload], dec.rdtype)

            prev_return_state = getattr(dec, "_return_state", False)
            prev_hard_out = getattr(dec, "_hard_out", False)
            try:
                dec._return_state = True
                dec._hard_out = False
                for chunk_idx, iters in enumerate(SCHEDULE):
                    x1_stage = tf.concat([payload_intr, crc_and_rest], axis=1)
                    llr_bp = tf.concat([x1_stage, z_short, x2_par], axis=1)
                    x_hat, curr_msg_v2c = LDPCBPDecoder.call(
                        dec, llr_bp, num_iter=int(iters), msg_v2c=curr_msg_v2c)
                    if chunk_idx >= len(SCHEDULE) - 1:
                        continue

                    post_payload = x_hat[:, :k_payload]
                    bp_ext = post_payload - payload_intr
                    counts, ratios, _ = compute_syndrome_weight_and_ratio(
                        x_hat, dec._h_sparse,
                        decision_rule=args.syndrome_hard_decision,
                        compare_both_signs=False,
                    )
                    ratio_np = tf.cast(ratios, tf.float32).numpy().reshape(-1)

                    cand_metrics = {}
                    for sig in args.candidate_sigmas:
                        if args.calibration_objective == "source_posterior_ber":
                            src_ext = dec._denoiser(post_payload, sigma=float(sig))
                            src_post = post_payload + src_ext
                            hard = tf.cast(src_post > 0.0, dec.rdtype)
                            bit_err = tf.reduce_sum(
                                tf.cast(tf.not_equal(hard, u_bits), tf.float32),
                                axis=1)
                            cand_metrics[float(sig)] = {
                                "bit_err": bit_err.numpy().reshape(-1),
                                "nack": None,
                            }
                        else:
                            bit_err, nack = run_tail_from_chunk(
                                dec, crc_dec,
                                payload0, crc_and_rest, z_short, x2_par,
                                payload_intr, curr_msg_v2c, chunk_idx + 1,
                                post_payload, bp_ext, u_bits, a, b,
                                float(sig), float(args.trace_tail_sigma),
                            )
                            cand_metrics[float(sig)] = {
                                "bit_err": bit_err.numpy().reshape(-1),
                                "nack": nack.numpy().reshape(-1),
                            }

                    for i in range(int(ratio_np.shape[0])):
                        row = {
                            "chunk_idx": int(chunk_idx),
                            "ratio": float(ratio_np[i]),
                            "n_bits": int(k_payload),
                            "candidates": {},
                        }
                        for sk, m in cand_metrics.items():
                            entry = {"bit_err": float(m["bit_err"][i])}
                            if m["nack"] is not None:
                                entry["nack"] = float(m["nack"][i])
                            row["candidates"][str(sk)] = entry
                        sample_cache.append(row)

                    if args.collection_mode == "handcrafted":
                        coll_sigma = sigma_from_syndrome_ratio(
                            ratios,
                            tuple(args.hand_thresholds),
                            tuple(args.hand_sigmas),
                            sigma_min=args.sigma_min,
                            sigma_max=args.sigma_max,
                        )
                        src_ext_collect = dec._denoiser(post_payload, sigma=coll_sigma)
                    else:
                        src_ext_collect = dec._denoiser(
                            post_payload, sigma=float(args.collection_sigma))
                    payload_intr = payload0 + b * bp_ext + a * src_ext_collect
            finally:
                dec._return_state = prev_return_state
                dec._hard_out = prev_hard_out

    return sample_cache


def _bin_thresholds(ratios: np.ndarray, args) -> list[float]:
    """Compute bin thresholds for one set of (per-chunk) ratios."""
    if args.bin_thresholds:
        return [float(x) for x in args.bin_thresholds]
    q = np.linspace(0.0, 1.0, args.num_bins + 1)[1:-1]
    return np.unique(np.round(np.quantile(ratios, q), 6)).tolist()


def _select_sigma_per_bin(rows, candidate_sigmas, thresholds, objective):
    """Given a subset of sample rows, compute per-bin best σ.

    Returns
    -------
    sigma_levels       : list[float]
    per_bin_meta       : list[dict]   (bin_idx, edges, num_samples,
                                       candidate_objective_values, selected_sigma,
                                       selected_objective_value)
    """
    num_bins = len(thresholds) + 1
    thr_np = np.asarray(thresholds, dtype=np.float32)

    def bin_index(ratio: float) -> int:
        return int(np.searchsorted(thr_np, np.float32(ratio), side="right"))

    acc = {bi: {float(sk): {"bit_err": 0.0, "bits": 0.0,
                            "nack": 0.0, "blocks": 0.0}
                for sk in candidate_sigmas}
           for bi in range(num_bins)}
    samples_per_bin = [0 for _ in range(num_bins)]
    for row in rows:
        bi = bin_index(row["ratio"])
        samples_per_bin[bi] += 1
        for sk, ent in row["candidates"].items():
            sig = float(sk)
            acc[bi][sig]["bit_err"] += ent["bit_err"]
            acc[bi][sig]["bits"] += float(row["n_bits"])
            if ent.get("nack") is not None:
                acc[bi][sig]["nack"] += ent["nack"]
                acc[bi][sig]["blocks"] += 1.0

    sigma_levels = []
    per_bin_meta = []
    for bi in range(num_bins):
        cand_summary = {}
        best_sigma = None
        best_score = float("inf")
        for sig in candidate_sigmas:
            s = float(sig)
            o = acc[bi][s]
            ber = o["bit_err"] / max(o["bits"], 1.0)
            bler = o["nack"] / max(o["blocks"], 1.0) if o["blocks"] > 0 else None
            if objective == "source_posterior_ber":
                score = ber
                cand_summary[str(s)] = {
                    "mean_payload_ber": ber,
                    "note": "proxy on post_payload",
                }
            elif objective == "tail_final_ber":
                score = ber
                cand_summary[str(s)] = {"mean_final_payload_ber": ber}
            else:
                score = bler if bler is not None else ber
                cand_summary[str(s)] = {
                    "mean_final_bler": bler,
                    "mean_final_payload_ber": ber,
                }

            if score < best_score:
                best_score = score
                best_sigma = s

        if best_sigma is None:
            best_sigma = float(candidate_sigmas[0])
            best_score = None

        sigma_levels.append(best_sigma)
        lo = float("-inf") if bi == 0 else float(thresholds[bi - 1])
        hi = float("inf") if bi == num_bins - 1 else float(thresholds[bi])
        per_bin_meta.append({
            "bin_idx": bi,
            "bin_edges_open_closed": {"low_exclusive": lo, "high_inclusive": hi},
            "num_samples": samples_per_bin[bi],
            "candidate_objective_values": cand_summary,
            "selected_sigma": best_sigma,
            "selected_objective_value": best_score,
        })
    return sigma_levels, per_bin_meta


def calibrate(args) -> dict:
    sample_cache = collect_samples(args)
    if not sample_cache:
        raise RuntimeError("No calibration samples were collected.")

    # Mode A: per-chunk thresholds + per-chunk sigma_levels (default).
    # Mode B: --no-per-chunk → single global lookup over all chunks.
    chunks_present = sorted({int(r["chunk_idx"]) for r in sample_cache})

    objective_notes = {
        "source_posterior_ber": (
            "Proxy: BER on payload bits after one denoiser step on BP_post, "
            "before remaining BP chunks.  Cheap but not a faithful surrogate "
            "of final BLER — use mainly for diagnostics."
        ),
        "tail_final_ber": (
            "Surrogate: after applying candidate σ at the calibration chunk, "
            "run remaining BP chunks; denoiser on later chunks uses "
            "trace_tail_sigma."
        ),
        "tail_final_nack": (
            "Recommended.  Surrogate BLER: CRC NACK on final decoder output "
            "after tail trace (same tail schedule as tail_final_ber)."
        ),
    }

    base = {
        "schema_version": SIGMA_LOOKUP_SCHEMA_VERSION,
        "method": "offline_sigma_lookup_calibration",
        "objective": args.calibration_objective,
        "objective_notes": objective_notes,
        "alpha": args.alpha,
        "beta": args.beta,
        "schedule": SCHEDULE,
        "ebno_list": list(args.ebno),
        "batch": args.batch,
        "rounds": args.rounds,
        "syndrome_hard_decision": args.syndrome_hard_decision,
        "collection_mode": args.collection_mode,
        "collection_sigma": args.collection_sigma,
        "trace_tail_sigma": args.trace_tail_sigma,
        "candidate_sigmas": [float(x) for x in args.candidate_sigmas],
        "binning_type": "manual" if args.bin_thresholds else "quantile",
        "monotonic_sigma": bool(args.monotonic_sigma),
        "num_samples_total": len(sample_cache),
    }
    if args.sigma_min is not None:
        base["sigma_min"] = float(args.sigma_min)
    if args.sigma_max is not None:
        base["sigma_max"] = float(args.sigma_max)

    if args.per_chunk:
        per_chunk_lookup = {}
        for ci in chunks_present:
            rows = [r for r in sample_cache if int(r["chunk_idx"]) == ci]
            ratios = np.asarray([r["ratio"] for r in rows], dtype=np.float32)
            thresholds = _bin_thresholds(ratios, args)
            sigma_levels, per_bin_meta = _select_sigma_per_bin(
                rows, args.candidate_sigmas, thresholds,
                args.calibration_objective,
            )
            if args.monotonic_sigma:
                sigma_levels = list(enforce_monotonic_sigma(sigma_levels))
            per_chunk_lookup[str(ci)] = {
                "thresholds": thresholds,
                "sigma_levels": sigma_levels,
                "num_bins": len(sigma_levels),
                "per_bin": per_bin_meta,
            }
        out = {
            **base,
            "per_chunk": True,
            "per_chunk_lookup": per_chunk_lookup,
        }
    else:
        ratios_arr = np.asarray([r["ratio"] for r in sample_cache], dtype=np.float32)
        thresholds = _bin_thresholds(ratios_arr, args)
        sigma_levels, per_bin_meta = _select_sigma_per_bin(
            sample_cache, args.candidate_sigmas, thresholds,
            args.calibration_objective,
        )
        if args.monotonic_sigma:
            sigma_levels = list(enforce_monotonic_sigma(sigma_levels))
        out = {
            **base,
            "per_chunk": False,
            "thresholds": thresholds,
            "sigma_levels": sigma_levels,
            "samples_per_bin": {str(i): m["num_samples"]
                                for i, m in enumerate(per_bin_meta)},
            "per_bin": per_bin_meta,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--ebno", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--batch", type=int, default=80)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--collection-mode", default="fixed",
                   choices=["fixed", "handcrafted"])
    p.add_argument("--collection-sigma", type=float, default=0.3)
    p.add_argument("--trace-tail-sigma", type=float, default=None,
                   help="σ for denoiser on chunks after the calibration chunk "
                        "(tail_final_* objectives). Default: collection-sigma.")
    p.add_argument(
        "--calibration-objective", default="tail_final_nack",
        choices=["source_posterior_ber", "tail_final_ber", "tail_final_nack"],
        help="Recommended: tail_final_nack.  source_posterior_ber is a proxy "
             "and should not be used as the main calibrated lookup.",
    )
    p.add_argument("--hand-thresholds", type=float, nargs="+",
                   default=list(DEFAULT_SYNDROME_THRESHOLDS))
    p.add_argument("--hand-sigmas", type=float, nargs="+",
                   default=list(DEFAULT_SIGMA_LEVELS))
    p.add_argument(
        "--candidate-sigmas", type=float, nargs="+",
        default=list(DEFAULT_MAIN_CANDIDATE_SIGMAS),
        help="Candidate σ values.  Default 0.20…0.40 surrounds the strong "
             "fixed σ=0.3 baseline; wider ranges are diagnostic-only.",
    )
    p.add_argument("--num-bins", type=int, default=4)
    p.add_argument("--bin-thresholds", type=float, nargs="+", default=None,
                   help="If set, use these as fixed bin thresholds; "
                        "otherwise quantile bins are chosen automatically.")
    p.add_argument("--sigma-min", type=float, default=DEFAULT_MAIN_SIGMA_MIN)
    p.add_argument("--sigma-max", type=float, default=DEFAULT_MAIN_SIGMA_MAX)
    p.add_argument("--monotonic-sigma", action="store_true",
                   help="Enforce non-decreasing σ across larger ratio bins.")
    p.add_argument("--per-chunk", dest="per_chunk", action="store_true",
                   default=True,
                   help="(Default.) Calibrate a separate lookup per BP chunk.")
    p.add_argument("--no-per-chunk", dest="per_chunk", action="store_false",
                   help="Pool all chunks together into one global lookup.")
    p.add_argument("--syndrome-hard-decision", default="xhat_gt0",
                   choices=["xhat_gt0", "xhat_lt0"])
    p.add_argument("--output-json", default="results/sigma_lookup_calibrated.json")
    p.add_argument("--gpu-memory-mb", type=int, default=None,
                   help="GPU memory cap (MB).  Also picked up before TF "
                        "import via FMNIST_GPU_MEM_MB.")
    args = p.parse_args()
    if args.trace_tail_sigma is None:
        args.trace_tail_sigma = args.collection_sigma

    result = calibrate(args)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved calibrated lookup -> {args.output_json}")
    print(f"  schema_version={result['schema_version']}  "
          f"objective={result['objective']}  "
          f"per_chunk={result['per_chunk']}  "
          f"monotonic_sigma={result['monotonic_sigma']}")
    if result["per_chunk"]:
        for ci, pc in result["per_chunk_lookup"].items():
            print(f"  chunk {ci}: thresholds={pc['thresholds']} "
                  f"sigma_levels={pc['sigma_levels']}")
    else:
        print(f"  thresholds={result['thresholds']}")
        print(f"  sigma_levels={result['sigma_levels']}")
        print(f"  samples_per_bin={result['samples_per_bin']}")

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    main()
