"""
SoftDenoiser — TF layer wrapping the PyTorch SourcePriorDenoiser.

[onlyextrinsic variant — independent alpha/beta]
  Always returns pure source extrinsic (src_post - BP_post).
  gamma removed; BP vs source weighting is done in the decoder via alpha/beta.
"""

import numpy as np
import tensorflow as tf
import torch

from source_prior import SourcePriorDenoiser


class SoftDenoiser(tf.keras.layers.Layer):
    """
    Input:  LLR tensor [B, K] — BP posterior for payload bits
    Output: source extrinsic LLR [B, K] = src_post - input_llr
    """

    def __init__(self,
                 img_h: int = 28,
                 img_w: int = 28,
                 bits_per_pixel: int = 8,
                 model_channels: int = 64,
                 channel_mult=(1, 2, 2),
                 num_blocks: int = 2,
                 attn_resolutions=(7,),
                 sigma_data: float = 0.5,
                 sigma_post: float = 3.0,
                 device: str = 'cpu',
                 **kwargs):
        super().__init__(**kwargs)
        self._sigma = 1.0
        self._device = device
        self._prior = SourcePriorDenoiser(
            img_h=img_h,
            img_w=img_w,
            bits_per_pixel=bits_per_pixel,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            sigma_data=sigma_data,
            sigma_post=sigma_post,
        ).to(device)
        self._prior.eval()

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = float(value)

    @property
    def prior_model(self):
        return self._prior

    @property
    def sigma_post(self):
        """[Prompt C] Pixel→bit read-out std (EP 2nd moment) in 0..255 units.
        Default 3.0 (tuned, very sharp).  Set to the denoiser's measured error
        scale ε for a statistically honest per-bit posterior LLR."""
        return self._prior.sigma_post

    @sigma_post.setter
    def sigma_post(self, value):
        self._prior.sigma_post = float(value)

    def set_posterior_std_from_mse(self, mse_unit_interval):
        """[Prompt C, "auto_mse"] Set sigma_post to the denoiser error scale
        ε = 255·√(test MSE), converting a [0,1]-scale MSE to the 0..255 pixel
        units of ``sigma_post``.  This replaces the tuned 3.0 with the empirical
        posterior error; bits with place-value w_m ≲ ε then collapse to LLR→0
        (honest) while w_m ≫ ε (MSBs) stay sharp.  Returns ε.  Preserves the
        3.0 path unless called."""
        eps = 255.0 * float(mse_unit_interval) ** 0.5
        self._prior.sigma_post = eps
        return eps

    # ── [2a] multistep sampler pass-throughs (see source_prior.py) ──
    @property
    def sampler(self):
        """'single_shot' (Tweedie mean) | 'multistep' (EDM cavity-conditioned
        sampler)."""
        return self._prior.sampler

    @sampler.setter
    def sampler(self, value):
        assert value in ("single_shot", "multistep")
        self._prior.sampler = value

    def configure_multistep(self, **kw):
        """Set multistep knobs on the prior: ms_steps, ms_sigma_max, ms_sigma_min,
        ms_guidance, ms_guidance_const, ms_use_confidence, ms_stochastic, ms_churn."""
        allowed = {"ms_steps", "ms_sigma_max", "ms_sigma_min", "ms_guidance",
                   "ms_guidance_const", "ms_use_confidence", "ms_stochastic",
                   "ms_churn"}
        for k, v in kw.items():
            if k not in allowed:
                raise KeyError(f"unknown multistep knob: {k}")
            setattr(self._prior, k, v)

    @property
    def last_ms_trace(self):
        """Per-step diagnostics of the latest multistep sample (or None)."""
        return self._prior.last_ms_trace

    @property
    def cavity_var_readout(self):
        """[Part D] use per-pixel cavity std √v_j as the pixel→bit read-out std."""
        return self._prior.cavity_var_readout

    @cavity_var_readout.setter
    def cavity_var_readout(self, value):
        self._prior.cavity_var_readout = bool(value)

    @property
    def cavity_var_sigma(self):
        """[Part D] set the denoiser σ per image from √v_j: None | 'median' | 'mean'."""
        return self._prior.cavity_var_sigma

    @cavity_var_sigma.setter
    def cavity_var_sigma(self, value):
        self._prior.cavity_var_sigma = value

    def load_weights_pt(self, path):
        state = torch.load(path, map_location=self._device, weights_only=True)
        self._prior.load_state_dict(state)
        self._prior.eval()

    def save_weights_pt(self, path):
        torch.save(self._prior.state_dict(), path)

    def call(self, llr_tf, sigma=None):
        """
        Parameters
        ----------
        llr_tf : [B, K] BP posterior logits for payload bits.
        sigma : optional scalar or per-batch schedule. If None, uses ``self.sigma``
            (scalar). If a rank-1 tensor/array of length B, passed to the prior as
            per-example denoiser noise level.
        """
        llr_np = llr_tf.numpy()
        llr_pt = torch.from_numpy(llr_np).float().to(self._device)

        if sigma is None:
            sig_arg = self._sigma
        else:
            if isinstance(sigma, tf.Tensor):
                sig_arg = torch.from_numpy(sigma.numpy().astype("float32")).reshape(-1)
            else:
                sig_arg = torch.as_tensor(sigma, dtype=torch.float32, device=self._device).reshape(-1)
            sig_arg = sig_arg.to(self._device)
            if sig_arg.numel() == 1:
                sig_arg = float(sig_arg.item())

        with torch.no_grad():
            ext_pt = self._prior(llr_pt, sigma=sig_arg)

        ext_np = ext_pt.cpu().numpy()
        return tf.constant(ext_np, dtype=llr_tf.dtype)
