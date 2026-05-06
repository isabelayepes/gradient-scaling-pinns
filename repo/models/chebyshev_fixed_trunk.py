# models/chebyshev_fixed_trunk.py
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ChebyshevFixedTrunkConfig:
    """
    Fixed Chebyshev trunk on t in [0, T].

    Time scaling:
        x = 2 * (t / T) - 1  in [-1, 1]

    Features:
        Phi(t) = [T0(x), T1(x), ..., T_degree(x)]

    Notes:
      - Chebyshev polynomials are well-conditioned on [-1,1]
      - Feature normalization is critical for stable optimization
    """
    degree: int              # polynomial degree M
    T: float                 # final time horizon
    include_bias: bool = False
    normalize_features: bool = True


class ChebyshevFixedTrunk(nn.Module):
    def __init__(self, cfg: ChebyshevFixedTrunkConfig):
        super().__init__()
        if cfg.degree < 0:
            raise ValueError("degree must be >= 0.")
        if cfg.T <= 0:
            raise ValueError("T must be > 0.")

        self.degree = int(cfg.degree)
        self.T = float(cfg.T)
        self.include_bias = bool(cfg.include_bias)
        self.normalize_features = bool(cfg.normalize_features)

    @property
    def out_dim(self) -> int:
        d = self.degree + 1
        if self.include_bias:
            d += 1
        return d

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute Chebyshev features Phi(t).

        Input:
          t shape (N,), (N,1), or (...,1)

        Output:
          Phi shape (..., out_dim)
        """
        # Ensure (..., 1)
        if t.ndim == 1:
            t_in = t[..., None]
        elif t.ndim >= 2 and t.shape[-1] == 1:
            t_in = t
        else:
            raise ValueError(f"Expected t shape (N,) or (...,1); got {tuple(t.shape)}")

        # --- Time scaling: map t ∈ [0, T] → x ∈ [-1, 1]
        x = 2.0 * (t_in / self.T) - 1.0  # (..., 1)
        x = x.squeeze(-1)               # (...,)

        # --- Chebyshev recurrence (stable)
        # T0(x) = 1
        # T1(x) = x
        # T_{k+1}(x) = 2 x T_k(x) - T_{k-1}(x)

        feats = []

        T0 = torch.ones_like(x)
        feats.append(T0)

        if self.degree >= 1:
            T1 = x
            feats.append(T1)

        for k in range(1, self.degree):
            Tk = feats[k]
            Tk_1 = feats[k - 1]
            Tk1 = 2.0 * x * Tk - Tk_1
            feats.append(Tk1)

        # (..., degree+1)
        phi = torch.stack(feats, dim=-1)

        # --- Optional feature normalization (VERY important)
        if self.normalize_features:
            phi = phi / math.sqrt(phi.shape[-1])

        # --- Optional bias term
        if self.include_bias:
            ones = torch.ones_like(phi[..., :1])
            phi = torch.cat([ones, phi], dim=-1)

        return phi
