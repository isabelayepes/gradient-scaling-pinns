# models/fourier_fixed_trunk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import math
import torch
import torch.nn as nn


@dataclass(frozen=True)
class FourierFixedTrunkConfig:
    """
    Fixed-frequency Fourier trunk.

    Phi(t) = [sin(w1 t), ..., sin(wM t), cos(w1 t), ..., cos(wM t)]
    D = 2M (+1 if include_bias)

    Notes:
      - omegas are rad/s
      - t is assumed in seconds on [0, T]
      - omegas may be any 1D sequence/array (np.ndarray, list, tuple, torch tensor on CPU)
    """
    omegas: Sequence[float]
    include_bias: bool = False


def build_omegas_logspace(w_min: float, w_max: float, n: int) -> np.ndarray:
    """
    Log-spaced frequencies in [w_min, w_max] (rad/s).
    Returns np.ndarray of shape (M,) dtype float32.
    """
    if n <= 0:
        raise ValueError("M must be positive.")
    if w_min <= 0 or w_max <= 0:
        raise ValueError("w_min and w_max must be positive for logspace.")
    if w_max < w_min:
        raise ValueError("w_max must be >= w_min.")
    return np.logspace(np.log10(w_min), np.log10(w_max), num=n, dtype=np.float32)


def build_omegas_linear(w_min: float, w_max: float, n: int) -> np.ndarray:
    """
    Linearly spaced frequencies in [w_min, w_max] (rad/s).
    Returns np.ndarray of shape (M,) dtype float32.
    """
    if n <= 0:
        raise ValueError("M must be positive.")
    if w_max < w_min:
        raise ValueError("w_max must be >= w_min.")
    return np.linspace(w_min, w_max, num=n, dtype=np.float32)


def build_omegas_bands(
    low: Tuple[float, float, int],
    mid: Tuple[float, float, int],
    high: Tuple[float, float, int],
    spacing: str = "log",
) -> np.ndarray:
    """
    Convenience: build low/mid/high frequency bands.

    Each band is (w_min, w_max, n_band). Spacing can be 'log' or 'linear'.
    Returns concatenated omegas (low then mid then high) as np.ndarray float32.
    """
    if spacing not in {"log", "linear"}:
        raise ValueError("spacing must be 'log' or 'linear'")

    builder = build_omegas_logspace if spacing == "log" else build_omegas_linear

    chunks: list[np.ndarray] = []
    for (a, b, n) in (low, mid, high):
        if n <= 0:
            continue
        chunks.append(builder(a, b, n))

    if not chunks:
        raise ValueError("At least one band must have n_band > 0.")

    return np.concatenate(chunks, axis=0).astype(np.float32)


class FourierFixedTrunk(nn.Module):
    """
    Fixed Fourier-feature trunk.
    """

    def __init__(self, cfg: FourierFixedTrunkConfig):
        super().__init__()

        # Convert omegas to a 1D torch buffer (device-movable, saved in state_dict).
        omegas = torch.as_tensor(cfg.omegas, dtype=torch.float32)
        if omegas.ndim != 1:
            omegas = omegas.reshape(-1)
        if omegas.numel() == 0:
            raise ValueError("cfg.omegas must be non-empty.")

        self.include_bias = bool(cfg.include_bias)
        self.register_buffer("omegas", omegas, persistent=True)

    @property
    def num_freqs(self) -> int:
        return int(self.omegas.numel())

    @property
    def out_dim(self) -> int:
        d = 2 * self.num_freqs
        if self.include_bias:
            d += 1
        return d

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Accept (N,) or (N,1) or (...,1)
        if t.ndim == 1:
            t_in = t[..., None]
        elif t.ndim >= 2 and t.shape[-1] == 1:
            t_in = t
        else:
            raise ValueError(f"Expected t shape (N,) or (...,1); got {tuple(t.shape)}")

        t_in = t_in.to(dtype=self.omegas.dtype)
        phase = t_in * self.omegas  # (..., M)

        sin_part = torch.sin(phase)
        cos_part = torch.cos(phase)
        feats = torch.cat([sin_part, cos_part], dim=-1)  # (..., 2M)

        if self.include_bias:
            ones = torch.ones_like(t_in[..., :1])
            feats = torch.cat([ones, feats], dim=-1)  # (..., 1+2M)

        return feats


def fourier_features_and_grads(
    trunk: FourierFixedTrunk,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Analytic Phi(t), dPhi/dt, d2Phi/dt2 for fixed-frequency Fourier trunk.
    Useful for debugging, not required by the main PINN pipeline.
    """
    if t.ndim == 1:
        t_in = t[..., None]
    elif t.ndim >= 2 and t.shape[-1] == 1:
        t_in = t
    else:
        raise ValueError(f"Expected t shape (N,) or (...,1); got {tuple(t.shape)}")

    t_in = t_in.to(dtype=trunk.omegas.dtype)
    w = trunk.omegas  # (M,)
    phase = t_in * w  # (..., M)

    sin_part = torch.sin(phase)
    cos_part = torch.cos(phase)

    d_sin = cos_part * w
    d_cos = -sin_part * w

    w2 = w * w
    d2_sin = -sin_part * w2
    d2_cos = -cos_part * w2

    phi = torch.cat([sin_part, cos_part], dim=-1)
    dphi = torch.cat([d_sin, d_cos], dim=-1)
    d2phi = torch.cat([d2_sin, d2_cos], dim=-1)

    if trunk.include_bias:
        ones = torch.ones_like(t_in[..., :1])
        zeros = torch.zeros_like(t_in[..., :1])
        phi = torch.cat([ones, phi], dim=-1)
        dphi = torch.cat([zeros, dphi], dim=-1)
        d2phi = torch.cat([zeros, d2phi], dim=-1)

    return phi, dphi, d2phi
