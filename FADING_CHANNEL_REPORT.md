# QPSK + fast Rayleigh fading + imperfect CSI — channel option (stage 1)

Branch `practical_sigma`.  Adds a selectable channel front-end; the decoder
comparison structure (BP-only baseline vs BP+denoiser) is unchanged — only the
LLR the decoders receive changes.  Stage 1 = implement + verify LLR/sign;
interference is deferred to stage 2.

## Files
- **`channel_models.py`** — `ChannelModel(kind, sigma_e2, perfect_csi)` with
  `.transmit(c, no) → LLR [B,N]`.  Kinds: `awgn_bpsk` (original real PAM+AWGN,
  preserved) and `qpsk_fast_fading_imperfect_csi`.
- **`fading_experiment.py`** — CLI runner (`--channel`, `--sigma-e2`,
  `--perfect-csi`, `--ebno`), BP-only vs BP+denoiser(EP [5]×20, fixed σ) on the
  SAME LLR, CSV output.
- **`channel_sign_check.py`** — sign/LLR verification (requirement 9).

## Channel model (`qpsk_fast_fading_imperfect_csi`)
- Modulation **4-QAM/QPSK**, `NUM_BPS=2`; N=12600 → 6300 symbols (12600/2 exact).
- Symbol-wise **fast flat Rayleigh**: `y_i = h_i x_i + n_i`, `h_i ~ CN(0,1)` i.i.d.
  per QPSK symbol, `n_i ~ CN(0, N0)`.  No interleaving (stage 1).
- **Imperfect CSI**: `ĥ_i = h_i + e_i`, `e_i ~ CN(0, sigma_e2)` (`--sigma-e2`,
  default 0.0; sweep 0.01/0.05/0.1/0.2).  `--perfect-csi` forces `ĥ=h`.
- **Mismatched LLR uses ĥ, not h.**  We equalise by ĥ and pass the per-symbol
  effective noise variance to Sionna's QAM APP demapper:
  `ỹ_i = y_i/ĥ_i`, `Ñ0_i = N0/|ĥ_i|²`.  This is *identically* the direct
  likelihood `exp(-|y_i - ĥ_i x|²/N0)` (method A ≡ method B), so it is exact, and
  reusing the same `Demapper("app", …)` as the PAM path keeps the **same LLR sign
  convention** (LLR>0 ⇔ bit 1).  Both decoders receive this one LLR.

## Sign / LLR verification (requirement 9) — `channel_sign_check.py`
| check | awgn_bpsk | qpsk_fast_fading |
|---|---|---|
| (1) pre-decode coded-bit BER, `llr>0→1` vs tx `c`, hi-SNR perfect CSI | **0.00000** | **0.00160** |
| (2) end-to-end CRC24A pass / payload BER | **1.000 / 0** | **1.000 / 0** |
| (3) imperfect-CSI monotone degradation (CRC pass @ebno 6) | — | σ_e² 0/.05/.1 → 1.0, **.2 → 0.19** |

- Pre-decode BER ≈ 0 (not ≈ 1) ⇒ sign convention is `llr>0 ⇔ bit 1`, matching the
  decoder + CRC.  The QPSK 0.0016 is Rayleigh deep-fade raw error that LDPC+CRC
  fully correct (CRC pass 1.0, payload bit-exact).
- End-to-end CRC passes and recovers the payload bit-exact ⇒ the decoder and CRC
  work **exactly as before**; only the front-end changed.
- Imperfect CSI degrades monotonically ⇒ channel physics correct.

## Demo sweep (stability; `results/fading_demo.csv`, 512–640 cw, Wilson CI)

| channel | σ_e² | Eb/N0 | BP-only BLER | BP+denoiser BLER |
|---|---|---|---|---|
| awgn_bpsk (perfect) | — | 0.6 | 0.239 | **0.003** |
| qpsk fading (perfect) | 0.0 | 6 / 10 | 0 / 0 | 0 / 0 |
| qpsk fading | 0.05 | 6 / 10 | 0 / 0 | 0.002 / 0.002 |
| qpsk fading | 0.10 | 6 / 10 | 0 / 0 | 0.006 / **0.057** |
| qpsk fading | 0.20 | 6 / 10 | 0.766 / 0.190 | **0.815 / 0.574** |

- **AWGN-BPSK path preserved:** BP-only 0.239 ≈ the §F BP-100 ceiling; the
  denoiser recovers it to 0.003 — original behaviour intact.
- **Runs stably across the full σ_e² sweep** (no NaN/overflow; monotone physics).
- **Preliminary (demo-scale) finding:** under QPSK fading + imperfect CSI the
  BP+denoiser is **neutral-to-worse** than BP-only — the source prior, tuned
  against the *AWGN* posterior (fixed σ=0.3, sigma_post=3.0), does not transfer to
  the mismatched-CSI posterior and can inject harmful confidence (e.g. σ_e²=0.1 at
  Eb/N0 10: BP-only 0 vs denoiser 0.057).  This is exactly the rough-channel
  robustness question to quantify next; **not** conclusive at 512–640 cw.

## CSV schema
`channel, num_bps, sigma_e2, perfect_csi, ebno_db, bler_baseline, ber_baseline,
bler_fixed, ber_fixed, baseline_ci_lo, baseline_ci_hi, fixed_ci_lo, fixed_ci_hi,
nack_baseline, nack_fixed, total`.

## Notes / next
- Stage 2 (deferred): co-channel / multi-user interference — the channel is an
  isolated module, so this slots into `channel_models.py`.
- The AWGN-BPSK option reproduces the original path (preserved for regression).
- Preliminary: under heavy CSI mismatch the denoiser's source prior may not help
  (it was trained against the AWGN posterior) — to be quantified in the full
  robustness sweep.
