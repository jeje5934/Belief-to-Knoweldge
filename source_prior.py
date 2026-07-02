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
                 sigma_post: float = 3.0,
                 tweedie_precision: bool = False,
                 tweedie_probes: int = 4,
                 tweedie_eps: float = 0.02,
                 tweedie_std_floor: float = 0.5,
                 tweedie_std_cap: float = 128.0):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.bpp = bits_per_pixel
        self.n_pixels = img_h * img_w
        self.k_payload = self.n_pixels * bits_per_pixel
        # [Approx C — real Tweedie 2nd moment].  When enabled, the per-pixel
        # projected-posterior std used in the pixel→bit read-out is computed from
        # the denoiser Jacobian diagonal (Tweedie: Var[s|x̃]=σ²·∂D/∂x̃), estimated
        # by Hutchinson finite differences (NO backprop).  Uncertain pixels get a
        # large std ⇒ a weak bit-LLR site — the honest fix for the overconfident
        # fixed ``sigma_post``.  Default off preserves the fixed-variance path.
        self.tweedie_precision = bool(tweedie_precision)
        self.tweedie_probes = int(tweedie_probes)
        self.tweedie_eps = float(tweedie_eps)
        self.tweedie_std_floor = float(tweedie_std_floor)   # 0..255 units
        self.tweedie_std_cap = float(tweedie_std_cap)
        # EP 2nd-moment (precision) approximation [Approx C].  ``sigma_post`` is
        # the FIXED std (in 0..255 pixel-value units) of the projected posterior
        # over each pixel's discrete level; it converts the denoised pixel MEAN
        # into bit-domain LLRs.  In exact EP this equals sqrt(v_proj) with the
        # Tweedie 2nd moment  v_proj = sigma^2 · ∂D/∂x.  Smaller ⇒ higher-precision
        # (sharper) bit-domain source site.  It is the fixed-variance path
        # required by Task 4; ``posterior_pixel_std`` overrides it per-call.
        self.sigma_post = sigma_post

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
    # Tweedie 2nd moment  (real per-pixel precision, Approx C)
    # ------------------------------------------------------------------

    def _tweedie_pixel_std(self, mu_cavity, mu_proj, sigma):
        """Per-pixel projected-posterior std via the Tweedie 2nd moment.

        Tweedie:  Var[s | x̃] = σ² · ∂D/∂x̃   (diagonal, per pixel).
        We estimate the Jacobian diagonal diag(∂D/∂x̃) with **Hutchinson
        finite-difference** probes — no autograd/backprop:

            diag ≈ mean_m  v_m ⊙ (D(x̃ + ε v_m) − D(x̃)) / ε ,   v_m ~ Rademacher.

        A pixel where D tracks its input (∂D/∂x̃ large) is *uncertain* ⇒ large
        variance ⇒ weak bit-LLR site; a pixel D denoises confidently
        (∂D/∂x̃ small) ⇒ small variance ⇒ strong site.

        Returns per-pixel std in 0..255 units, shape [B, n_pixels].
        """
        B = mu_cavity.shape[0]
        sig4 = sigma.reshape(B, 1, 1, 1)
        diag = torch.zeros_like(mu_cavity)
        for _ in range(self.tweedie_probes):
            v = (torch.randint(0, 2, mu_cavity.shape, device=mu_cavity.device,
                               dtype=mu_cavity.dtype) * 2.0 - 1.0)   # ±1 Rademacher
            Dv = self.net(mu_cavity + self.tweedie_eps * v, sigma).clamp(0.0, 1.0)
            diag = diag + v * (Dv - mu_proj) / self.tweedie_eps
        diag = (diag / self.tweedie_probes).clamp(0.0, 1.0)   # ∂D/∂x̃ ∈ [0,1]
        v_proj = (sig4 ** 2) * diag                            # normalized-unit variance
        std_pix = (255.0 * torch.sqrt(v_proj.clamp(min=1e-12))).reshape(B, self.n_pixels)
        return std_pix.clamp(self.tweedie_std_floor, self.tweedie_std_cap)

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

        # (3a) projection, 1st moment: Tweedie posterior mean via the EDM
        #      denoiser.  D(μ_cav; σ) = E[clean image | Gaussian obs] = the
        #      projected posterior MEAN.  EXACT given the score network.
        mu_proj = self.net(mu_cavity, sigma)
        mu_proj = mu_proj.clamp(0.0, 1.0)

        # (3b) projection, 2nd moment.  If Tweedie precision is enabled (and no
        #      explicit override), compute the per-pixel std from the denoiser
        #      Jacobian diagonal [Approx C — real 2nd moment]; otherwise use the
        #      fixed/overridden ``posterior_pixel_std``.
        if self.tweedie_precision and posterior_pixel_std is None:
            posterior_pixel_std = self._tweedie_pixel_std(mu_cavity, mu_proj, sigma)

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
