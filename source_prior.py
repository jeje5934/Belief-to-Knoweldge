"""
SourcePriorDenoiser — PyTorch adapter that connects an EDM-style score
denoiser to the existing iterative BP pipeline.

EP interpretation (docs/EP_THEORY.md §4.1 step 3)
-------------------------------------------------
This module implements the **source-factor projection** of Expectation
Propagation, amortized by a score/EDM denoiser.  One call performs the
tilted → projection → site-update steps of the EP cycle:

  input  llr  ──►  interpret as the CAVITY over the payload bits
                   (channel+code belief, source site removed; §4.1 step 1)
  (2) tilted   ──►  read the cavity as a GAUSSIAN OBSERVATION of the clean
                    image:  mean = cavity pixel mean,  variance = sigma^2
  (3a) project ──►  1st moment: D(mean; sigma) = E[clean | obs] = posterior
                    mean  (Tweedie's formula; EXACT given the score net)
  (3b) project ──►  2nd moment: APPROXIMATED by a fixed projected-posterior
                    pixel std (`sigma_post`), used to turn the denoised pixel
                    mean back into bit-domain LLRs  [Approx C — see below]
  (4) site     ──►  src_site = projected_posterior_llr − cavity_llr
                    (EP division = LLR subtraction, §3)

Named approximations (also annotated inline)
---------------------------------------------
  * Approx A (mean-field cavity): `llr_to_soft_field` propagates only the
    FIRST moment of the cavity to the pixel domain; the cavity variance is not
    derived from the per-bit LLR magnitudes but supplied separately as `sigma`.
  * Approx C (mean-only site / fixed 2nd moment): the projected posterior
    VARIANCE is a fixed constant (`sigma_post`) instead of the exact Tweedie
    2nd moment  v_proj = sigma^2 · ∂D/∂x.  The (mean, precision) site is
    therefore carried as (LLR, fixed precision); the precision slot is exposed
    (`return_precision`, `posterior_pixel_std`, `projected_pixel_precision`)
    so Task 5 can drop in the per-pixel Tweedie variance without a rewrite.
  * Approx D (sigma not from cavity variance): `sigma` is still chosen
    externally (syndrome scheduler) rather than being the true cavity std;
    tied to the cavity precision in Task 5.

[onlyextrinsic variant — independent alpha/beta]
  Returns the source SITE update (turbo name: "pure source extrinsic"):
    src_site = src_post − input_llr
  With the EP cavity as input (decoder.py, ep_mode=True) this is exactly the
  EP site update  projection − cavity.  The decoder controls BP vs source
  weighting via alpha/beta (legacy) or the EP site cycle (ep_mode).
"""

import math
import torch
import torch.nn as nn

from score_denoiser import EDMPrecond


class SourcePriorDenoiser(nn.Module):

    def __init__(self,
                 img_h: int = 28,
                 img_w: int = 28,
                 bits_per_pixel: int = 8,
                 model_channels: int = 64,
                 channel_mult=(1, 2, 2),
                 num_blocks: int = 2,
                 attn_resolutions=(7,),
                 dropout: float = 0.0,
                 sigma_data: float = 0.5,
                 sigma_post: float = 3.0):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.bpp = bits_per_pixel
        self.n_pixels = img_h * img_w
        self.k_payload = self.n_pixels * bits_per_pixel
        # EP 2nd-moment (precision) approximation [Approx C].  ``sigma_post`` is
        # the FIXED std (in 0..255 pixel-value units) of the projected posterior
        # over each pixel's discrete level; it converts the denoised pixel MEAN
        # into bit-domain LLRs.  In exact EP this equals sqrt(v_proj) with the
        # Tweedie 2nd moment  v_proj = sigma^2 · ∂D/∂x.  Smaller ⇒ higher-precision
        # (sharper) bit-domain source site.  It is the fixed-variance path
        # required by Task 4; ``posterior_pixel_std`` overrides it per-call.
        # [Prompt C] Setting sigma_post to the denoiser's measured error scale
        # ε = 255·√(test MSE) (see denoiser.SoftDenoiser.set_posterior_std_from_mse,
        # "auto_mse") replaces the tuned 3.0 with a statistically honest global
        # 2nd moment: it auto-stratifies confidence by bit place-value (MSB stay
        # sharp, LSB→0).  Global ε cannot capture correlated error → not a full-EP
        # fix; it targets the damped-EP path.
        self.sigma_post = sigma_post
        # [Part D] Cavity per-pixel variance options (Approx A candidate).  The
        # mean-only bit→pixel collapse (`llr_to_soft_field`) makes uncertain
        # pixels mid-gray → an OOD denoiser input.  ``pixel_variance`` recovers a
        # per-pixel cavity variance v_j; these flags let it drive (a) the denoiser
        # σ (per-image scalar from √v_j) and/or (b) the pixel→bit read-out std.
        # Default off preserves the fixed-σ / fixed-sigma_post path.
        self.cavity_var_readout = False       # use √v_j as posterior_pixel_std
        self.cavity_var_sigma = None          # None | "median" | "mean"

        # ── [2a] Multistep diffusion posterior sampler (cavity-conditioned) ──
        # The single-shot projection  mu_proj = D(μ_cav; σ)  returns the Tweedie
        # posterior MEAN, which at large σ AVERAGES the modes → a gray blur (the
        # σ-experiment PSNR 15.9→6.2 failure).  The multistep sampler instead runs
        # an EDM diffusion trajectory (σ_max ↓ σ_min) that SELECTS one mode, with
        # a per-step cavity-guidance term pulling the *denoised* estimate toward the
        # cavity (ICDM-style conditional diffusion decoding).  These are plain
        # attributes (set after construction, like cavity_var_*); default
        # sampler="single_shot" leaves the exact single-shot path byte-for-byte.
        # Enabling multistep affects WHATEVER path calls the denoiser (intended use:
        # the legacy source step) — the decoder code itself is untouched.
        self.sampler = "single_shot"          # "single_shot" | "multistep"
        self.ms_steps = 10                    # K diffusion steps
        self.ms_sigma_max = 1.5               # σ_0  (explore modes; EDM range ≤3.32)
        self.ms_sigma_min = 0.05              # σ_K  (commit to a mode; ≥0.027)
        self.ms_guidance = 0.5                # ζ cavity-guidance scale
        self.ms_guidance_const = False        # True → constant ζ; False → anneal weak→strong
        self.ms_use_confidence = True         # weight guidance by per-pixel cavity precision
        self.ms_stochastic = False            # EDM Langevin churn (noise reinjection)
        self.ms_churn = 0.1                   # per-step churn γ when ms_stochastic
        self.last_ms_trace = None             # per-step diagnostics of the latest sample

        weights = torch.tensor(
            [2 ** (bits_per_pixel - 1 - i) for i in range(bits_per_pixel)],
            dtype=torch.float32)
        self.register_buffer('bit_weights', weights)

        masks = torch.zeros(bits_per_pixel, 2 ** bits_per_pixel)
        for i in range(bits_per_pixel):
            w = 2 ** (bits_per_pixel - 1 - i)
            for v in range(2 ** bits_per_pixel):
                masks[i, v] = float((v // w) % 2)
        self.register_buffer('bit_masks', masks)
        self.register_buffer('pixel_values',
                             torch.arange(2 ** bits_per_pixel, dtype=torch.float32))

        img_res = max(img_h, img_w)
        self.net = EDMPrecond(
            img_resolution=img_res,
            img_channels=1,
            sigma_data=sigma_data,
            model_type='SongUNet',
            model_channels=model_channels,
            channel_mult=list(channel_mult),
            channel_mult_emb=4,
            num_blocks=num_blocks,
            attn_resolutions=list(attn_resolutions),
            dropout=dropout,
            embedding_type='positional',
            channel_mult_noise=1,
            encoder_type='standard',
            decoder_type='standard',
            resample_filter=[1, 1],
        )

    # ------------------------------------------------------------------
    # LLR ↔ image conversions  (EP cavity ↔ pixel-domain moments)
    # ------------------------------------------------------------------

    def llr_to_soft_field(self, llr):
        """Cavity (bit-LLR) → Gaussian-observation MEAN in the pixel domain.

        Interprets the per-bit cavity as independent Bernoullis, takes the
        expected pixel value  E[pixel] = Σ_i 2^{bpp-1-i}·P(bit_i=1), and
        normalises to [0,1].  This is the mean μ_cav of the Gaussian
        observation that the denoiser (source projection) consumes.

        Variance propagation (bit → pixel).  [Approx A — mean-field cavity]:
        ONLY the first moment of the cavity is propagated here.  The cavity
        variance is NOT computed from the per-bit LLR magnitudes; it is supplied
        separately as the denoiser noise level ``sigma`` (the cavity std, see
        :meth:`forward`).  Exact EP would propagate the full per-pixel cavity
        variance (EP_THEORY.md §4.1 step 3; tied up in Task 5, Approx D).
        """
        B = llr.shape[0]
        p = torch.sigmoid(llr)                       # per-bit P(bit=1) (cavity)
        p = p.reshape(B, self.n_pixels, self.bpp)
        soft_pixel = (p * self.bit_weights).sum(dim=-1)   # E[pixel] ∈ [0,255]
        img = soft_pixel / 255.0                     # μ_cav ∈ [0,1]
        img = img.reshape(B, 1, self.img_h, self.img_w)
        return img

    def pixel_variance(self, llr):
        """Per-pixel variance of the mean-field cavity image (independent bits).

        μ_j = Σ_m w_m · p_jm   (the pixel mean `llr_to_soft_field` already uses)
        v_j = Σ_m w_m² · p_jm·(1−p_jm)                          [Part D]

        with w_m the bit place-value and p_jm = P(bit_m = 1) = sigmoid(llr).
        Large v_j ⇒ the pixel is uncertain (the mean collapses to gray).  Units:
        0..255² (variance); take sqrt for a per-pixel std in 0..255.  Shape [B, n_pix].
        """
        B = llr.shape[0]
        p = torch.sigmoid(llr).reshape(B, self.n_pixels, self.bpp)
        v = (p * (1.0 - p) * (self.bit_weights ** 2)).sum(dim=-1)
        return v

    def soft_field_to_posterior_logits(self, img, posterior_pixel_std=None):
        """Projected pixel MEAN → bit-domain projected posterior LLRs.

        Reads the projected posterior mean ``img`` (per pixel, [0,1]) and
        spreads it into a soft distribution over the 2^bpp discrete pixel levels
        using a Gaussian with std ``posterior_pixel_std`` — the projected
        posterior pixel std, i.e. the EP **2nd moment / precision**.
        Marginalising that distribution onto each bit yields the projected
        bit-LLRs.

        Variance propagation (pixel → bit).  A SMALLER ``posterior_pixel_std``
        (higher precision) makes P(pixel=v) sharper, hence larger-magnitude bit
        LLRs — a more confident, higher-precision Bernoulli site.  A LARGER std
        softens them.  So ``posterior_pixel_std`` is exactly the knob that sets
        the precision of the emitted bit-domain source site.

        Parameters
        ----------
        img : [B, 1, H, W] or [B, n_pixels]  projected posterior pixel mean.
        posterior_pixel_std : None → ``self.sigma_post`` (fixed 2nd moment,
            Approx C).  Scalar or per-pixel [B, n_pixels] tensor — the latter is
            the slot for the Task-5 Tweedie variance.
        """
        B = img.shape[0]
        pix = img.reshape(B, self.n_pixels) * 255.0        # projected mean in 0..255

        # Projected-posterior pixel std (EP 2nd moment).  Fixed by default
        # (Approx C); accept a scalar or a per-pixel [B, n_pixels] override.
        std = self.sigma_post if posterior_pixel_std is None else posterior_pixel_std
        std = torch.as_tensor(std, dtype=pix.dtype, device=pix.device)
        if std.dim() == 0:
            std_b = std.reshape(1, 1, 1)                   # broadcast over B, pixel, value
        else:
            std_b = std.reshape(B, self.n_pixels, 1)

        diff = pix.unsqueeze(-1) - self.pixel_values       # [B, n_pixels, 2^bpp]
        # Gaussian observation model  N(pixel_value; projected_mean, std^2):
        # the projected posterior over the discrete pixel level.
        log_pv = -0.5 * diff ** 2 / (std_b ** 2)
        p_v = torch.softmax(log_pv, dim=-1)                # P(pixel = v)

        p_bit1 = torch.einsum('bpv,iv->bip', p_v, self.bit_masks)   # P(bit_i = 1)
        p_bit1 = p_bit1.clamp(1e-7, 1 - 1e-7)
        llr = torch.log(p_bit1 / (1 - p_bit1))             # projected posterior LLR
        llr = llr.permute(0, 2, 1).reshape(B, -1)
        return llr

    def projected_pixel_precision(self, img, posterior_pixel_std=None):
        """PRECISION SLOT (EP 2nd moment) → per-pixel precision [B, n_pixels].

        Returns the per-pixel precision 1/var of the projected posterior.

        Fixed-variance path (Task 4, option (a)): precision = 1/std^2 with the
        fixed ``posterior_pixel_std`` (or ``self.sigma_post``) — constant across
        pixels.  [Approx C.]  Exact EP would return the Tweedie 2nd moment
        v_proj = sigma^2·∂D/∂x (per-pixel, input-dependent); the [B, n_pixels]
        shape is chosen so Task 5 can substitute it directly.
        """
        B = img.shape[0]
        std = self.sigma_post if posterior_pixel_std is None else posterior_pixel_std
        std = torch.as_tensor(std, dtype=img.dtype, device=img.device)
        prec = 1.0 / (std ** 2)
        if prec.dim() == 0:
            prec = prec.reshape(1, 1).expand(B, self.n_pixels)
        elif prec.dim() == 1:
            prec = prec.reshape(1, -1).expand(B, self.n_pixels)
        return prec

    # ------------------------------------------------------------------
    # Site update  (EP: projection − cavity)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_source_extrinsic(posterior_logits, input_llr):
        """EP source SITE update = projection − cavity (LLR subtraction, §4.1 step 4).

        Turbo name: "pure source extrinsic" src_ext = src_post − input_llr.
        When ``input_llr`` is the EP cavity (decoder ep_mode=True) this is
        exactly the new source site  projected_posterior_llr − cavity_llr.
        """
        return posterior_logits - input_llr

    # ------------------------------------------------------------------
    # [2a] Multistep diffusion posterior sampler (cavity-conditioned)
    # ------------------------------------------------------------------

    def _edm_sigmas(self, K, device):
        """Descending σ schedule σ_0 > … > σ_K, geometric between ms_sigma_max and
        ms_sigma_min (EDM-standard log-spacing).  Returns [K+1]."""
        smax = float(self.ms_sigma_max); smin = float(self.ms_sigma_min)
        j = torch.arange(K + 1, dtype=torch.float32, device=device) / max(K, 1)
        return smax * (smin / smax) ** j                       # [K+1]: smax … smin

    def _guidance_at(self, sj, s0, sK):
        """Cavity-guidance scale ζ_j.  const → ms_guidance; else annealed WEAK at
        large σ (explore) → STRONG at small σ (commit) — the ICDM prescription."""
        if self.ms_guidance_const:
            return float(self.ms_guidance)
        frac = float((s0 - sj) / (s0 - sK + 1e-8))             # 0 at σ_0, 1 at σ_K
        return float(self.ms_guidance) * frac

    def _multistep_sample(self, mu_cavity, cavity_llr):
        """EDM trajectory that samples a clean image consistent with the cavity.

        (spec [1])  x_{σ0} = μ_cav + σ_0·ε ; then for j=0…K-1:
            x̂0 = D(x_{σj}; σj)                                    (prior denoise)
            x̂0 ← x̂0 + ζ_j·w·(μ_cav − x̂0)                         (cavity guidance
                  on the DENOISED estimate — selects a mode, not the mode-mean)
            x_{σj+1} = x̂0 + σ_{j+1}·(x_{σj} − x̂0)/σj              (EDM Euler step)
        Returns the final denoised estimate x̂0 (∈[0,1]) as the projected mean.

        w is the per-pixel cavity precision (normalised to (0,1] per image): the
        A=I likelihood ∇½‖x−μ‖²/λ² weights each pixel by 1/var, so confident cavity
        pixels are pulled hard and uncertain ones defer to the prior.  ms_use_confidence
        off ⇒ w≡1.  ms_stochastic adds EDM churn (Langevin corrector).
        """
        dev = mu_cavity.device
        B = mu_cavity.shape[0]
        K = int(self.ms_steps)
        sig = self._edm_sigmas(K, dev)                         # [K+1]

        if self.ms_use_confidence:
            v = self.pixel_variance(cavity_llr).clamp(min=1e-6)          # [B,n_pix]
            prec = 1.0 / v
            w = prec / prec.amax(dim=1, keepdim=True)                    # (0,1]
            w = w.reshape(B, 1, self.img_h, self.img_w)
        else:
            w = torch.ones_like(mu_cavity)

        x = mu_cavity + sig[0] * torch.randn_like(mu_cavity)   # cavity-anchored init
        x0 = mu_cavity
        trace = []
        for j in range(K):
            sj = sig[j]; sj1 = sig[j + 1]
            if self.ms_stochastic:                             # EDM Alg-2 churn
                s_hat = sj * (1.0 + float(self.ms_churn))
                x = x + torch.sqrt((s_hat ** 2 - sj ** 2).clamp(min=0.0)) \
                    * torch.randn_like(x)
                sj = s_hat
            sig_b = torch.full((B,), float(sj), device=dev, dtype=x.dtype)
            x0 = self.net(x, sig_b).clamp(0.0, 1.0)            # x̂0 = D(x;σj)
            zeta = self._guidance_at(sj, sig[0], sig[-1])
            x0 = (x0 + zeta * w * (mu_cavity - x0)).clamp(0.0, 1.0)
            x = x0 + sj1 * (x - x0) / sj                       # deterministic step
            trace.append({
                "step": j, "sigma": float(sj), "zeta": float(zeta),
                "x0_mean": float(x0.mean()),
                "dist_cavity": float((x0 - mu_cavity).pow(2).mean().sqrt()),
            })
        self.last_ms_trace = trace
        return x0                                              # final projected mean

    # ------------------------------------------------------------------
    # Forward  (one EP source-factor projection)
    # ------------------------------------------------------------------

    def forward(self, llr, sigma, *,
                posterior_pixel_std=None, return_precision=False):
        """Amortized EP source-factor projection (docs/EP_THEORY.md §4.1).

        Parameters
        ----------
        llr : [B, K]
            The CAVITY bit-LLR over payload bits (channel+code belief with the
            source site removed; EP_THEORY.md §4.1 step 1).  Interpreted as a
            Gaussian observation of the clean image.
        sigma : scalar or [B]
            Gaussian-observation noise std = the cavity std used by the
            denoiser.  [Approx D]: should be the true cavity std; still supplied
            externally (Task 5 ties it to the cavity precision).
        posterior_pixel_std : optional
            Override of the fixed projected-posterior pixel std (the PRECISION
            SLOT; default ``self.sigma_post``).  Scalar or [B, n_pixels].
        return_precision : bool
            If True, also return the pixel-domain site precision [B, n_pixels]
            (fixed path → constant; Task 5 → Tweedie Jacobian).  Default False
            keeps the single-tensor return expected by ``denoiser.py``.

        Returns
        -------
        src_site : [B, K]
            Source site update = projected_posterior_llr − cavity_llr.
        pixel_precision : [B, n_pixels]   (only if ``return_precision``)
            EP 2nd-moment precision of the projected posterior (the site slot).
        """
        cavity_llr = llr

        # (2) tilted: interpret the cavity as a Gaussian observation of the clean
        #     image — mean = cavity pixel mean, variance = sigma^2 (Approx A/D).
        mu_cavity = self.llr_to_soft_field(cavity_llr)          # μ_cav ∈ [0,1]

        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma], dtype=torch.float32, device=llr.device)
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        sigma = sigma.expand(llr.shape[0])

        # [Part D] cavity per-pixel std √v_j (0..255) from the independent-bit
        # variance, used to (a) set a per-image σ and/or (b) the read-out std.
        cav_std_pix = None
        if self.cavity_var_readout or self.cavity_var_sigma is not None:
            cav_std_pix = torch.sqrt(
                self.pixel_variance(cavity_llr).clamp(min=1e-8))     # [B, n_pix], 0..255
        if self.cavity_var_sigma is not None:
            q = cav_std_pix / 255.0                                   # normalized per-pixel std
            s_img = (q.median(dim=1).values if self.cavity_var_sigma == "median"
                     else q.mean(dim=1))                              # per-image scalar
            sigma = s_img.clamp(min=1e-3).to(sigma.dtype)            # override global σ

        # (3a) projection, 1st moment.
        #   single_shot: Tweedie posterior MEAN via one EDM denoiser call
        #                D(μ_cav; σ) = E[clean | Gaussian obs] (mode-averaging).
        #   multistep  : [2a] EDM diffusion trajectory that SAMPLES one mode,
        #                cavity-conditioned (selects a peak, not the peak-mean).
        if self.sampler == "multistep":
            mu_proj = self._multistep_sample(mu_cavity, cavity_llr)
        else:
            mu_proj = self.net(mu_cavity, sigma)
        mu_proj = mu_proj.clamp(0.0, 1.0)

        # (3b) projection, 2nd moment.  [Part D] optionally use the per-pixel
        #      cavity std √v_j as the read-out std (else fixed sigma_post / arg).
        if self.cavity_var_readout and posterior_pixel_std is None:
            posterior_pixel_std = cav_std_pix
        proj_llr = self.soft_field_to_posterior_logits(
            mu_proj, posterior_pixel_std=posterior_pixel_std)

        # (4) site update = projection − cavity (EP division = LLR subtraction).
        src_site = self.compute_source_extrinsic(proj_llr, cavity_llr)

        if return_precision:
            # PRECISION SLOT: (mean_llr, precision) site.  Fixed 1/std^2 today;
            # Task 5 replaces with the per-pixel Tweedie variance.
            pixel_precision = self.projected_pixel_precision(
                mu_proj, posterior_pixel_std=posterior_pixel_std)
            return src_site, pixel_precision
        return src_site
