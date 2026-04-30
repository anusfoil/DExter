"""DiT-style transformer denoiser for DExter v2.

Drop-in replacement for ``model.diffwave.ClassifierFreeDenoiser`` — same
forward signature ``(x_t, s_codec, c_codec, diffusion_step, sampling=False)``
so it plugs into the existing ``task.diffusion.CodecDiffusion`` superclass
without changes there. ``c_codec`` is accepted but ignored (v2 drops the
perceptual conditioning latent — see dexter-v2-design memory).

Hierarchy is expressed in two places, neither of which is VirtuosoNet's
HAN-stack:

1. **Score-derived structured positional encoding** — each note gets a
   sinusoidal position embedding on its onset_quarter (continuous beat
   coordinate from the score), plus learned embeddings for voice and
   metrical-position-in-measure. Hierarchy is encoded in the inputs.
2. **Self-attention over the full note sequence**. The attention pattern is
   not factored note→beat→measure; the model learns whatever neighborhood
   structure helps. With score positions, "same beat" = "very close in
   onset_quarter" naturally.

The wavelet/multi-scale target representation is implemented separately
(see model.hierarchical_target — coming next).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.diffwave import DiffusionEmbedding
from model.utils import Normalization
from task.diffusion import CodecDiffusion


# ---------------------------------------------------------------------------
# AdaLN-Zero modulation (from DiT, Peebles & Xie 2023)
# ---------------------------------------------------------------------------

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply per-token shift + scale: x * (1+scale) + shift. Broadcasts on tokens."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZeroBlock(nn.Module):
    """A pre-norm transformer block with AdaLN-Zero modulation.

    The diffusion-step embedding produces 6 modulation parameters (shift_msa,
    scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) that gate the residual
    contributions of self-attention and MLP. Initialized so the gates start at
    zero (the "Zero" in AdaLN-Zero) — at init the block is the identity, which
    helps training stability for diffusion.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        # 6 modulation vectors per layer, conditioned on the diffusion-step embedding.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        # Zero-init so blocks start as identity (DiT convention).
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)   c: (B, d_model)  conditioning vector (diffusion-step embedding)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class AdaLNFinalLayer(nn.Module):
    """Final AdaLN modulation + linear projection to output dim."""

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        self.linear = nn.Linear(d_model, out_dim, bias=True)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Score-derived positional encoding
# ---------------------------------------------------------------------------

def _sinusoidal_pos_emb(values: torch.Tensor, d_model: int, max_period: float = 10000.0):
    """1-D continuous sinusoidal embedding (transformer-style).

    Args:
        values: (B, N) float — the positions to embed (e.g. onset_quarter).
        d_model: even output channels.
    Returns:
        (B, N, d_model)
    """
    half = d_model // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=values.device) / half
    )
    args = values.unsqueeze(-1) * freqs        # (B, N, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ScorePositionalEncoder(nn.Module):
    """Builds per-token positional embeddings from score features.

    Currently consumes the v1-shape s_codec ``(onset_div, duration_div, pitch,
    voice)`` because that's what the existing pipeline produces. The
    ``onset_div`` channel becomes the continuous score-time position; ``voice``
    becomes a learned per-voice embedding. When prepare_data starts emitting
    the richer XML-derived s_codec (see features.xml_features), this class
    is the integration point — extend ``forward`` to consume the extra
    columns (measure_idx, beat_offset, etc.) without touching the rest of
    the model.
    """

    def __init__(
        self,
        d_model: int,
        s_codec_rows: int = 4,
        max_voice: int = 16,
        onset_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.s_codec_rows = s_codec_rows
        self.onset_scale = onset_scale
        # voice embedding (clip voice id to [0, max_voice-1])
        self.voice_embed = nn.Embedding(max_voice, d_model)
        # continuous-feature projection: takes [onset_div_norm, duration_div_norm, pitch_norm]
        # plus any extra continuous columns from richer s_codec (handled at runtime)
        self.cont_proj = nn.Linear(s_codec_rows - 1, d_model)
        self.max_voice = max_voice

    def forward(self, s_codec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_codec: (B, N, F) — channel-last (matches DataLoader convention).
                     Channel 0 is onset_div, channel 3 is voice id; remaining channels
                     are projected linearly into d_model.
        Returns:
            (B, N, d_model) positional embedding to add to token embeddings.
        """
        B, N, F_ = s_codec.shape
        onset_div = s_codec[..., 0]                                # (B, N)
        # rough normalization — onset_div can be huge; scale to keep sinusoid args bounded
        onset_norm = onset_div * self.onset_scale
        pos_sin = _sinusoidal_pos_emb(onset_norm, self.d_model)    # (B, N, d_model)

        voice = s_codec[..., 3].long().clamp(0, self.max_voice - 1)
        v_emb = self.voice_embed(voice)                            # (B, N, d_model)

        # continuous projection of remaining channels (duration, pitch, ...)
        # exclude onset (channel 0) and voice (channel 3), keep the rest
        cont = (
            torch.cat([s_codec[..., 1:3], s_codec[..., 4:]], dim=-1)
            if F_ > 4 else s_codec[..., 1:3]
        )
        # If cont has fewer columns than cont_proj.in_features, right-pad with zeros.
        if cont.shape[-1] < self.cont_proj.in_features:
            pad = torch.zeros(
                *cont.shape[:-1],
                self.cont_proj.in_features - cont.shape[-1],
                device=cont.device, dtype=cont.dtype,
            )
            cont = torch.cat([cont, pad], dim=-1)
        cont_emb = self.cont_proj(cont.float())

        return pos_sin + v_emb + cont_emb


# ---------------------------------------------------------------------------
# Top-level denoiser
# ---------------------------------------------------------------------------

class TransformerDenoiser(CodecDiffusion):
    """DiT-style transformer denoiser, plugged into ``CodecDiffusion``.

    Args mirror ``ClassifierFreeDenoiser`` for config compatibility, with a few
    new ones (``d_model``, ``n_heads``, ``n_layers``). The conv-stack-specific
    hyperparams (``residual_channels``, ``dilation_*``, ``kernel_size``) are
    accepted-and-ignored so the existing config schema works unchanged.
    """

    def __init__(
        self,
        # required by CodecDiffusion / config compatibility
        residual_channels: int,
        unconditional: bool,
        condition: str,
        p_codec_rows: int,
        s_codec_rows: int,
        c_codec_rows: int,    # accepted but unused — v2 drops c_codec
        norm_args,
        seg_len: int,
        # transformer-specific
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cond_dropout: float = 0.1,
        # accepted-and-ignored (conv-stack legacy)
        residual_layers: int = 12,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dilation_bound: int = 4,
        **kwargs,
    ):
        self.cond_dropout = cond_dropout
        super().__init__(**kwargs)

        self.d_model = d_model
        self.p_codec_rows = p_codec_rows
        self.s_codec_rows = s_codec_rows

        # token embedder for the noised p_codec
        self.x_proj = nn.Linear(p_codec_rows, d_model)
        # score-position / s_codec embedding
        self.score_pos = ScorePositionalEncoder(d_model, s_codec_rows=s_codec_rows)
        # diffusion-step embedding (reuse the v1 module — sinusoidal then 2 MLPs to 512)
        self.diffusion_embedding = DiffusionEmbedding(len(self.betas))
        # project diffusion embedding to d_model so AdaLN-Zero blocks consume it
        self.t_proj = nn.Linear(512, d_model)

        # condition normalization (kept for parity with v1; might drop later)
        self.condition_normalize = Normalization(norm_args[0], norm_args[1], norm_args[2])

        # transformer stack
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_layer = AdaLNFinalLayer(d_model, p_codec_rows)

        # CFG dropout strategy (drop s_codec to enable classifier-free guidance)
        if condition == "trainable_score":
            trainable_parameters = torch.full((s_codec_rows, seg_len), -1.0).float()
            self.register_parameter("trainable_parameters", nn.Parameter(trainable_parameters, requires_grad=True))
            self.uncon_dropout = self._trainable_dropout
        elif condition == "fixed":
            self.uncon_dropout = self._fixed_dropout
        else:
            raise ValueError(f"unrecognized condition '{condition}'")

    # ------- forward / dropout helpers (same semantics as v1) -------------

    def _fixed_dropout(self, x, p, masked_value=-1):
        mask = torch.distributions.Bernoulli(probs=p).sample((x.shape[0],)).long()
        mask_idx = mask.nonzero()
        x[mask_idx] = masked_value
        return x

    def _trainable_dropout(self, x, p):
        mask = torch.distributions.Bernoulli(probs=p).sample((x.shape[0],)).long()
        mask_idx = mask.nonzero()
        x[mask_idx] = self.trainable_parameters
        return x

    def forward(self, x_t, s_codec, c_codec, diffusion_step, sampling: bool = False):
        """
        Same I/O contract as ClassifierFreeDenoiser.forward.

        Args:
            x_t: (B, 1, N, p_codec_rows)  — noised codec
            s_codec: (B, s_codec_rows, N) — score features (as currently emitted by prepare_data)
            c_codec: (B, c_codec_rows, N) — accepted, ignored.
            diffusion_step: (B,) timestep
        Returns:
            (B, 1, N, p_codec_rows) — predicted noise (or x_0, depending on training.mode)
            second value is s_codec (returned for compatibility with v1 caller signature)
        """
        del c_codec  # v2 ignores c_codec; documented in module docstring
        B, _, N, _ = x_t.shape

        s_codec = s_codec.float()
        s_codec = self.condition_normalize(s_codec)

        # CFG dropout on s_codec during training
        if self.training:
            s_codec = self.uncon_dropout(s_codec, self.hparams.cond_dropout)

        # ---- token embedding
        x = x_t.squeeze(1)                                          # (B, N, p_codec_rows)
        tok = self.x_proj(x)                                        # (B, N, d_model)
        tok = tok + self.score_pos(s_codec)                         # (B, N, d_model)

        # ---- diffusion-step conditioning vector
        t_emb = self.diffusion_embedding(diffusion_step)            # (B, 512)
        c = self.t_proj(t_emb)                                      # (B, d_model)

        # ---- transformer stack with AdaLN-Zero modulation
        for blk in self.blocks:
            tok = blk(tok, c)

        # ---- final modulation + project to p_codec_rows
        out = self.final_layer(tok, c)                              # (B, N, p_codec_rows)
        return out.unsqueeze(1), s_codec                            # (B, 1, N, p_codec_rows), s_codec
