# models/adaptive_spectral_pinn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mixins import ICEmbeddingMixin, TimeDerivativesMixin


@dataclass(frozen=True)
class AdaptiveSpectralPINNConfig:
    """
    Model 4: Adaptive-frequency Fourier trunk + linear head (single-task upper bound).

    Trunk:
      Phi(t) = [sin(w1 t),...,sin(wK t), cos(w1 t),...,cos(wK t)] (+ optional bias)
      where w_k are learnable (optionally constrained positive).

    Head:
      [rho_tilde, theta_tilde] = Phi(t) @ W + b
    Then IC/positivity embedding as in the thesis.
    """
    omegas_init: torch.Tensor                 # (K,)
    include_bias: bool = False

    learnable_omegas: bool = True
    positive_omegas: bool = True
    softplus_beta: float = 1.0

    out_dim: int = 2
    head_bias: bool = False
    r_min: float = 1e-4


class AdaptiveFourierTrunk(nn.Module):
    """
    Adaptive-frequency Fourier trunk with shared sin/cos pair per omega.
    Exposes:
      - buffer `omegas_init` (initial frequencies)
      - property `omegas` (current frequencies, learned or fixed)
    """

    def __init__(
        self,
        omegas_init: torch.Tensor,
        include_bias: bool = False,
        learnable_omegas: bool = True,
        positive_omegas: bool = True,
        softplus_beta: float = 1.0,
    ):
        super().__init__()
        # Accept np.ndarray / list / tuple / torch.Tensor
        omegas0 = torch.as_tensor(omegas_init, dtype=torch.float32).clone().detach()
        if omegas0.ndim != 1:
            omegas0 = omegas0.reshape(-1)
        if omegas0.numel() == 0:
            raise ValueError("omegas_init must have at least 1 element.")

        self.include_bias = bool(include_bias)
        self.learnable_omegas = bool(learnable_omegas)
        self.positive_omegas = bool(positive_omegas)
        self.softplus_beta = float(softplus_beta)

        # Keep explicit fields for the different branches
        self._omegas_raw: Optional[nn.Parameter] = None      # for positive-learnable (softplus parametrization)
        self._omegas_param: Optional[nn.Parameter] = None    # for unconstrained learnable case

        # Always keep the initial vector as a buffer named exactly "omegas_init",
        # because other utilities (logger) will look for that name.
        self.register_buffer("omegas_init", omegas0.clone(), persistent=True)
        # Also keep a fixed copy (used if not learnable)
        self.register_buffer("_omegas_fixed", omegas0.clone(), persistent=True)

        if self.learnable_omegas:
            if self.positive_omegas:
                # invert softplus transform to get a raw parameter that will produce omegas0
                beta = self.softplus_beta
                w = omegas0.clamp_min(1e-8)
                # raw ≈ log(expm1(beta * w)) / beta so that softplus(raw, beta) ≈ w
                raw = (torch.log(torch.expm1(beta * w)) / beta).to(dtype=omegas0.dtype)
                self._omegas_raw = nn.Parameter(raw)
            else:
                # unconstrained learnable parameter
                self._omegas_param = nn.Parameter(omegas0.clone())
        # else: keep _omegas_fixed as registered buffer (already done above)

    @property
    def num_freqs(self) -> int:
        w = self._get_omegas()
        return int(w.numel())

    @property
    def out_dim(self) -> int:
        d = 2 * self.num_freqs
        if self.include_bias:
            d += 1
        return d

    @property
    def omegas(self) -> torch.Tensor:
        """Always returns the current frequencies on CPU/GPU as a Tensor."""
        return self._get_omegas()

    def _get_omegas(self) -> torch.Tensor:
        # Priority: positive-learnable (_omegas_raw) -> unconstrained-learnable (_omegas_param) -> fixed buffer
        if self._omegas_raw is not None:
            # softplus with beta ensures positivity and smoothness
            return F.softplus(self._omegas_raw, beta=self.softplus_beta)
        if self._omegas_param is not None:
            return self._omegas_param
        # fallback: registered fixed buffer
        return self._omegas_fixed

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t_in = t[..., None]
        elif t.ndim >= 2 and t.shape[-1] == 1:
            t_in = t
        else:
            raise ValueError(f"Expected t shape (N,) or (...,1); got {tuple(t.shape)}")

        # move omegas to the same device/dtype as t
        w = self._get_omegas().to(dtype=t_in.dtype, device=t_in.device)  # (K,)
        phase = t_in * w  # (..., K)

        sin_part = torch.sin(phase)
        cos_part = torch.cos(phase)
        feats = torch.cat([sin_part, cos_part], dim=-1)

        if self.include_bias:
            ones = torch.ones_like(t_in[..., :1])
            feats = torch.cat([ones, feats], dim=-1)

        return feats


class AdaptiveSpectralPINN(ICEmbeddingMixin, TimeDerivativesMixin, nn.Module):
    def __init__(self, cfg: AdaptiveSpectralPINNConfig):
        super().__init__()
        self.cfg = cfg

        if int(cfg.out_dim) != 2:
            raise ValueError("AdaptiveSpectralPINN currently assumes out_dim=2 for [r, theta] via [rho, theta].")

        self.trunk = AdaptiveFourierTrunk(
            omegas_init=cfg.omegas_init,
            include_bias=cfg.include_bias,
            learnable_omegas=cfg.learnable_omegas,
            positive_omegas=cfg.positive_omegas,
            softplus_beta=cfg.softplus_beta,
        )

        self.head = nn.Linear(self.trunk.out_dim, cfg.out_dim, bias=cfg.head_bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-2)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        # IC buffers
        self.register_buffer("_u0_dot", torch.zeros(1, 1))
        self.register_buffer("_rho0_dot", torch.zeros(1, 1))
        self._has_u0_dot = False
        self._has_rho0_dot = False
        self.register_buffer("_u0", torch.zeros(1, cfg.out_dim))
        self.register_buffer("_rho0", torch.zeros(1, 1))
        self._has_u0 = False
        self._has_rho0 = False
        self.r_min = float(cfg.r_min)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_in = self._normalize_time_input(t)

        phi = self.trunk(t_in)
        u_tilde = self.head(phi)  # (N,2) = [rho_tilde, theta_tilde]

        rho_tilde = u_tilde[:, 0:1]
        theta_tilde = u_tilde[:, 1:2]

        return self._embed_ic_and_positivity(t_in, rho_tilde, theta_tilde)
