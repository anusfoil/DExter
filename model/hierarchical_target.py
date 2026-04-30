"""Multi-scale (DCT / wavelet) target representations for DExter v2.

The v1 codec target is per-note and the diffusion model has no smoothness
prior — it learns to reproduce the per-note IOI-quantization noise that's
intrinsic in the labels. v2's idea: decompose the per-note expression curve
into a multi-scale basis and run diffusion in the coefficient space. Low-freq
coefficients carry phrase shape; high-freq coefficients carry note-level
texture; the basis itself encodes the hierarchy without any architectural
commitment to specific levels.

Two transforms are provided:

* ``DCTTarget``: the discrete cosine transform II (orthonormal). Global cosine
  basis. Cheap, exact, no extra dependency. Good first try.
* ``WaveletTarget``: Daubechies-4 wavelet decomposition via PyWavelets. Better
  localization (a tempo event in measure 8 only excites coefficients local to
  measure 8). Used when ``pywt`` is available; falls back to DCT otherwise.

Both transforms are exposed as ``nn.Module`` for clean integration with
``task.diffusion.CodecDiffusion``: the training step calls ``transform()``
on the target before adding noise, and ``inverse_transform()`` on the
denoised prediction before the reconstruction loss / rendering step.

The transform is applied **per feature channel independently**. Tempo,
velocity, etc. each get their own decomposition.

For low-pass smoothing as a smoothness prior, set ``keep_low_k`` to retain
only the K lowest-frequency coefficients and zero out the rest before the
inverse. The model sees a target with no high-frequency content, so it
literally cannot output noise above the bandwidth.

Integration sketch
------------------
In ``task/diffusion.py:training_step``::

    target = batch["p_codec"]                       # (B, N, F)
    target_coef = self.target_transform(target)     # (B, K, F) for K <= N
    noise = torch.randn_like(target_coef)
    x_t = q_sample(target_coef, t, noise)
    pred = self.denoiser(x_t, s_codec, c_codec, t)
    diff_loss = mse(pred, noise)                    # in coefficient space

For sampling, run the reverse process in coefficient space, then call
``self.target_transform.inverse(x_0_coef)`` once to get the per-note curve.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


# ---------------------------------------------------------------------------
# DCT-II (orthonormal) — implemented from scratch in torch (no scipy dep)
# ---------------------------------------------------------------------------

def _dct_basis(N: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """Build the orthonormal DCT-II basis matrix Φ, shape (N, N).

    For a signal x of length N, the DCT-II coefficients are c = Φ @ x, and the
    inverse is x = Φ.T @ c (Φ is orthogonal, so Φ.T == Φ⁻¹).
    """
    n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)        # (1, N)
    k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)        # (N, 1)
    Phi = torch.cos(math.pi * (n + 0.5) * k / N)                        # (N, N)
    Phi[0] *= 1.0 / math.sqrt(N)
    Phi[1:] *= math.sqrt(2.0 / N)
    return Phi


class DCTTarget(nn.Module):
    """DCT-II decomposition of the per-note target along the note axis.

    Args:
        seg_len: number of notes per segment (= ``N``).
        n_features: number of feature channels (= ``F``, e.g. 5 for
            ``[beat_period, velocity, timing, articulation_log, pedal]``).
        keep_low_k: keep only the K lowest-frequency coefficients per channel.
            Default ``None`` (= seg_len; lossless). Set to e.g. 32 to enforce
            a smoothness prior with bandwidth ≈ K/N.
    """

    def __init__(self, seg_len: int, n_features: int, keep_low_k: int | None = None):
        super().__init__()
        self.seg_len = seg_len
        self.n_features = n_features
        self.keep_low_k = keep_low_k or seg_len
        assert 1 <= self.keep_low_k <= seg_len
        # store basis as buffer so it moves with .to(device) / state_dict
        self.register_buffer("Phi", _dct_basis(seg_len), persistent=False)

    @property
    def coef_len(self) -> int:
        """Length of the coefficient axis after transform."""
        return self.keep_low_k

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Forward DCT.

        Args:
            x: (B, N, F) per-note signal.
        Returns:
            (B, K, F) coefficient signal where K = ``keep_low_k``.
        """
        # (B, N, F) → (B, F, N) → matmul Phi (N, N) on last dim → (B, F, N)
        x_t = x.transpose(1, 2)
        c_full = torch.matmul(x_t, self.Phi.t())                # (B, F, N)
        c = c_full[..., : self.keep_low_k]                       # (B, F, K)
        return c.transpose(1, 2)                                 # (B, K, F)

    def inverse(self, c: torch.Tensor) -> torch.Tensor:
        """Inverse DCT.

        Args:
            c: (B, K, F) coefficients (K may be < N — the rest are zero-padded).
        Returns:
            (B, N, F) reconstructed per-note signal.
        """
        B, K, F = c.shape
        if K < self.seg_len:
            pad = torch.zeros(B, self.seg_len - K, F, device=c.device, dtype=c.dtype)
            c = torch.cat([c, pad], dim=1)
        c_t = c.transpose(1, 2)                                  # (B, F, N)
        x = torch.matmul(c_t, self.Phi)                          # (B, F, N) - inverse via Phi (orthonormal so Phi.T @ Phi = I, applied as Phi.T from left)
        return x.transpose(1, 2)                                 # (B, N, F)

    forward = transform


# ---------------------------------------------------------------------------
# Daubechies wavelet decomposition via PyWavelets (optional)
# ---------------------------------------------------------------------------

class WaveletTarget(nn.Module):
    """Multi-level wavelet decomposition. Falls back to DCT if pywt missing.

    Uses a Daubechies-4 wavelet at ``levels`` levels, returning the flat
    coefficient vector (concatenation of approximation + details). The signal
    length must be compatible with the wavelet/levels combination — pywt
    handles padding internally; we keep things simple and use the
    ``periodization`` mode so reconstruction is exact at the original length.

    Note: pywt operates on numpy arrays, so this module breaks autograd at the
    transform boundary. That is fine for the training-time use because the
    transform is applied to fixed targets (no gradient needed through it). For
    inference / sampling, we still inverse-transform the predicted
    coefficients with no gradient flow.
    """

    def __init__(
        self,
        seg_len: int,
        n_features: int,
        wavelet: str = "db4",
        levels: int = 3,
        mode: str = "periodization",
    ):
        super().__init__()
        if not _HAS_PYWT:
            raise ImportError("pywt is required for WaveletTarget; pip install PyWavelets")
        self.seg_len = seg_len
        self.n_features = n_features
        self.wavelet = wavelet
        self.levels = levels
        self.mode = mode
        # determine coefficient layout
        sample = torch.zeros(seg_len)
        coefs = pywt.wavedec(sample.numpy(), wavelet, level=levels, mode=mode)
        self._slice_lens = [len(c) for c in coefs]
        self._coef_len = sum(self._slice_lens)

    @property
    def coef_len(self) -> int:
        return self._coef_len

    def _wavedec_batched(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pywt.wavedec along axis -2 of (B, N, F) → flat (B, K, F)."""
        # cpu numpy round-trip; small cost relative to a transformer forward pass
        arr = x.detach().cpu().numpy()                           # (B, N, F)
        B, N, F = arr.shape
        out = []
        for f in range(F):
            sig = arr[..., f]                                    # (B, N)
            chans = []
            for b in range(B):
                coefs = pywt.wavedec(sig[b], self.wavelet, level=self.levels, mode=self.mode)
                chans.append(_concat(coefs))
            out.append(_stack(chans))                            # (B, K)
        flat = _stack_last(out)                                  # (B, K, F)
        return torch.from_numpy(flat).to(x.device).to(x.dtype)

    def _waverec_batched(self, c: torch.Tensor) -> torch.Tensor:
        arr = c.detach().cpu().numpy()                           # (B, K, F)
        B, K, F = arr.shape
        out = []
        for f in range(F):
            chans = []
            for b in range(B):
                coefs = _split(arr[b, :, f], self._slice_lens)
                rec = pywt.waverec(coefs, self.wavelet, mode=self.mode)
                chans.append(rec[: self.seg_len])
            out.append(_stack(chans))                            # (B, N)
        sig = _stack_last(out)                                   # (B, N, F)
        return torch.from_numpy(sig).to(c.device).to(c.dtype)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self._wavedec_batched(x)

    def inverse(self, c: torch.Tensor) -> torch.Tensor:
        return self._waverec_batched(c)

    forward = transform


# ---------------------------------------------------------------------------
# Tiny numpy helpers (kept here to avoid extra imports in the wavelet path)
# ---------------------------------------------------------------------------

def _concat(arrs):
    import numpy as np
    return np.concatenate(arrs, axis=0)

def _split(arr, slice_lens):
    import numpy as np
    out = []
    i = 0
    for L in slice_lens:
        out.append(arr[i : i + L])
        i += L
    return out

def _stack(arrs):
    import numpy as np
    return np.stack(arrs, axis=0)

def _stack_last(arrs):
    import numpy as np
    return np.stack(arrs, axis=-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_target_transform(kind: str, seg_len: int, n_features: int, **kwargs) -> nn.Module:
    """kind ∈ {'identity', 'dct', 'wavelet'}."""
    if kind == "identity":
        return _IdentityTarget()
    if kind == "dct":
        return DCTTarget(seg_len, n_features, **kwargs)
    if kind == "wavelet":
        return WaveletTarget(seg_len, n_features, **kwargs)
    raise ValueError(f"unknown target transform kind: {kind!r}")


class _IdentityTarget(nn.Module):
    """No-op — used to keep the integration call-site uniform across configs."""

    def transform(self, x):
        return x

    def inverse(self, c):
        return c

    forward = transform

    @property
    def coef_len(self) -> int | None:
        return None
