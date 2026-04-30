"""Hierarchical-target diffusion: subclass of ``CodecDiffusion`` that runs the
diffusion process in a multi-scale coefficient space (DCT or wavelet).

The base class diffuses per-note: noise is added to and predicted in the
``(B, T, F_p)`` per-note tensor. This subclass instead transforms both the
target and the score conditioning into coefficient space first
(``(B, K, F_p)`` with K ≪ T for low-pass smoothing), runs the diffusion
process there, and inverse-transforms the predicted x_0 back to note space
for the reconstruction loss and downstream rendering.

Why this is useful: the per-note tempo/dynamics signal in the v1 dataset has
high-frequency content that's mostly alignment / IOI-quantization noise, not
musical signal. Diffusing in note space forces the model to learn to
reproduce this noise. Diffusing in coefficient space with ``keep_low_k``
much smaller than ``seg_len`` means the model literally cannot output above
the bandwidth — the smoothness prior is enforced by the target representation
itself, not by an architectural choice.

Score conditioning. ``s_codec`` is also DCT-transformed channel-wise to
match the coef-space length of x_t. This is mathematically clean for
continuous channels (onset_div, duration_div, pitch) and a documented
small hack for the categorical voice-id channel. The model sees coefficients
on both sides; per-coef-token score conditioning is preserved.

Usage. The Lightning module is a drop-in replacement for ``CodecDiffusion``;
swap ``_target_`` in the task config and add ``target_transform: dct`` and
``target_keep_low_k: 16``. Everything else (the denoiser, sampling type,
loss-weight schema) is unchanged.

Limitations / future work.

* Score conditioning via channel-wise DCT is the simplest thing that compiles.
  Better: a learned downsampling layer (T→K) for s_codec, or cross-attention
  from K coef-tokens to T score-tokens. Drop in by overriding
  ``_transform_s_codec`` here.
* Categorical s_codec channels (voice id) are DCT'd as floats — not
  semantically meaningful but the coefficients still carry distributional
  information. Once XML feature extraction is wired in, separate the
  categorical features and embed them once globally rather than per-coef.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from task.diffusion import CodecDiffusion, q_sample, extract_x0
from utils import tensor_pair_swap
from model.hierarchical_target import build_target_transform


class HierarchicalCodecDiffusion(CodecDiffusion):
    """``CodecDiffusion`` variant that runs in coefficient space.

    Args (in addition to ``CodecDiffusion``'s):
        target_transform: ``'identity' | 'dct' | 'wavelet'``. ``'identity'``
            recovers exactly the parent's behaviour and is the safe default.
        target_keep_low_k: number of low-frequency coefficients to retain
            (DCT only). ``None`` = lossless. Set to e.g. 16 to enforce a
            smoothness prior with bandwidth ≈ K/seg_len.
        transform_score_codec: if True, DCT-transform s_codec channel-wise so
            its length matches the p_codec coef length. If False, the model
            must internally handle the length mismatch — useful only for
            architectures with explicit cross-attention.
    """

    def __init__(
        self,
        *args,
        target_transform: str = "identity",
        target_keep_low_k: int | None = None,
        transform_score_codec: bool = True,
        **kwargs,
    ):
        # Don't consume seg_len / p_codec_rows / s_codec_rows here — they're
        # also required by TransformerDenoiser / ClassifierFreeDenoiser. Read
        # them from hparams after super().__init__ has stored them.
        super().__init__(*args, **kwargs)
        self._target_transform_kind = target_transform
        self._transform_score_codec = transform_score_codec

        # The denoiser parent class stores these on hparams via Lightning's
        # save_hyperparameters() machinery (or directly).
        seg_len      = self.hparams.get("seg_len", kwargs.get("seg_len", 200))
        p_codec_rows = self.hparams.get("p_codec_rows", kwargs.get("p_codec_rows", 5))
        s_codec_rows = self.hparams.get("s_codec_rows", kwargs.get("s_codec_rows", 4))

        kwargs_p = {"keep_low_k": target_keep_low_k} if target_transform == "dct" else {}
        self.p_target_transform = build_target_transform(
            target_transform, seg_len=seg_len, n_features=p_codec_rows, **kwargs_p
        )
        if transform_score_codec and target_transform != "identity":
            self.s_target_transform = build_target_transform(
                target_transform, seg_len=seg_len, n_features=s_codec_rows, **kwargs_p
            )
        else:
            self.s_target_transform = None

        # cache the coef-axis length (== seg_len for identity, K for dct/wavelet)
        self.coef_len = getattr(self.p_target_transform, "coef_len", seg_len) or seg_len

    # ----------------------------------------------------------------- helpers

    def _transform_p_codec(self, p_codec_btf: torch.Tensor) -> torch.Tensor:
        """(B, T, F_p) → (B, K, F_p)."""
        return self.p_target_transform.transform(p_codec_btf)

    def _inverse_p_codec(self, p_coef_bkf: torch.Tensor) -> torch.Tensor:
        """(B, K, F_p) → (B, T, F_p)."""
        return self.p_target_transform.inverse(p_coef_bkf)

    def _transform_s_codec(self, s_codec_btF: torch.Tensor) -> torch.Tensor:
        """(B, T, F_s) → (B, K, F_s), or pass-through if disabled.

        DataLoader convention is channel-last (B, T, F); transforms operate on
        the time axis directly so no transpose dance is needed.
        """
        if self.s_target_transform is None:
            return s_codec_btF
        return self.s_target_transform.transform(s_codec_btF.float())

    # --------------------------------------------------------------- training

    def step(self, batch, batch_idx):
        p_codec, s_codec, c_codec = batch["p_codec"], batch["s_codec"], batch["c_codec"]

        if self.hparams.drop_c_con:
            c_codec = torch.zeros_like(c_codec)

        device = p_codec.device

        # ---- transform target & score-codec to coefficient space
        # p_codec shape: (B, T, F_p) before unsqueeze
        p_coef = self._transform_p_codec(p_codec)               # (B, K, F_p)
        p_coef = p_coef.unsqueeze(1)                            # (B, 1, K, F_p)

        s_codec_coef = self._transform_s_codec(s_codec)         # (B, K, F_s)
        # c_codec is unused (dropped in v2); transform anyway for shape parity
        if self.s_target_transform is not None:
            c_codec_coef = self.s_target_transform.transform(c_codec.float())
        else:
            c_codec_coef = c_codec

        batch_size = p_coef.shape[0]

        t = torch.randint(
            0, self.hparams.timesteps, (batch_size,), device=device
        ).long().cpu()

        noise = torch.randn_like(p_coef)
        if self.hparams.training["target"] == "transfer":
            noise = tensor_pair_swap(p_coef)
            c_codec_coef = tensor_pair_swap(c_codec_coef) - c_codec_coef

        x_t = q_sample(
            x_start=p_coef,
            t=t,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            noise=noise,
        )

        # ---- diffusion loss in coefficient space
        if self.hparams.training["mode"] == "epsilon":
            epsilon_pred, _ = self(x_t, s_codec_coef, c_codec_coef, t)
            diffusion_loss = self.p_losses(noise, epsilon_pred, loss_type=self.hparams.loss_type)
            pred_p_coef = extract_x0(
                x_t, epsilon_pred, t,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            )
        elif self.hparams.training["mode"] == "x_0":
            pred_p_coef, _ = self(x_t, s_codec_coef, c_codec_coef, t)
            diffusion_loss = self.p_losses(p_coef, pred_p_coef, loss_type=self.hparams.loss_type)
        elif self.hparams.training["mode"] == "ex_0":
            epsilon_pred, _ = self(x_t, s_codec_coef, c_codec_coef, t)
            pred_p_coef = extract_x0(
                x_t, epsilon_pred, t,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            )
            diffusion_loss = self.p_losses(p_coef, pred_p_coef, loss_type=self.hparams.loss_type)
        else:
            raise ValueError(f"unsupported training mode: {self.hparams.training['mode']!r}")

        # ---- reconstruction losses in note space (interpretable, comparable to v1)
        # squeeze coef axis-1 sentinel from pred_p_coef: (B, 1, K, F_p) -> (B, K, F_p) -> (B, T, F_p)
        pred_p_codec_note = self._inverse_p_codec(pred_p_coef.squeeze(1))

        losses: dict = {}
        weight_sum = sum(self.hparams.loss_weight)
        for idx, name in enumerate(["tempo", "velocity", "timing", "duration", "pedal"]):
            losses[f"recon_{name}_loss"] = (
                self.p_losses(p_codec[..., idx], pred_p_codec_note[..., idx], loss_type="l2")
                * self.hparams.loss_weight[idx] / weight_sum * self.hparams.recon_loss_weight
            )
        losses["diffusion_loss"] = diffusion_loss

        tensors = {
            "pred_pcodec": pred_p_codec_note.unsqueeze(1),    # (B, 1, T, F_p) — match v1 contract
            "label_pcodec": p_codec.unsqueeze(1),
        }
        return losses, tensors

    # --------------------------------------------------------------- sampling

    def p_sample(self, batch, start_noise=None, sample_steps=None, c_codec=None):
        """Reverse-diffuse in coefficient space, then inverse-transform to note
        space at the end. The returned ``noise_list`` is a *note-space* trace
        of the denoising trajectory, so existing animation / visualization code
        keeps working unchanged.
        """
        p_codec = batch["p_codec"]                              # (B, T, F_p)
        s_codec = batch["s_codec"]                              # (B, F_s, T)

        if c_codec is None:
            c_codec = batch["c_codec"]

        # transform conditioning to coef space (channel-last convention)
        s_codec_coef = self._transform_s_codec(s_codec)
        if self.s_target_transform is not None:
            c_codec_coef = self.s_target_transform.transform(c_codec.float())
        else:
            c_codec_coef = c_codec

        if hasattr(self, "inner_loop"):
            self.inner_loop.refresh()
            self.inner_loop.reset()

        # initial noise lives in coef space
        if start_noise is not None:
            noise_coef = start_noise
        else:
            B, T, F_p = p_codec.shape
            noise_coef = torch.randn(B, 1, self.coef_len, F_p, device=self.device)

        # Trace stored in note space for the existing animation path:
        noise_list = []
        x_note = self._inverse_p_codec(noise_coef.squeeze(1)).unsqueeze(1)
        noise_list.append((x_note, self.hparams.timesteps))

        if sample_steps is None:
            sample_steps = self.hparams.timesteps

        x_coef = noise_coef
        for t_index in reversed(range(0, sample_steps)):
            x_coef, _ = self.reverse_diffusion(x_coef, s_codec_coef, c_codec_coef, t_index)
            x_note = self._inverse_p_codec(x_coef.detach().squeeze(1)).unsqueeze(1)
            noise_list.append((x_note.cpu().numpy(), t_index))
            if hasattr(self, "inner_loop"):
                self.inner_loop.update()

        return noise_list
