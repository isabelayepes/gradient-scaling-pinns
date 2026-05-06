# models/fixed_spectral_pinn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from models.mixins import ICEmbeddingMixin, TimeDerivativesMixin
from models.fourier_fixed_trunk import FourierFixedTrunk, FourierFixedTrunkConfig
from models.chebyshev_fixed_trunk import ChebyshevFixedTrunk, ChebyshevFixedTrunkConfig


BasisName = Literal["fourier", "chebyshev"]


@dataclass(frozen=True)
class FixedSpectralPINNConfig:
    """
    Model 3: Fixed spectral trunk + linear head (single-task).

    Choose basis in {"fourier", "chebyshev"}.
    The model predicts latent [rho_tilde, theta_tilde], then applies:
      rho_hat   = rho0   + t * rho_tilde(t)
      theta_hat = theta0 + t * theta_tilde(t)
      r_hat     = r_min + softplus(rho_hat)
    """
    basis: BasisName

    # Provide the relevant trunk config based on basis.
    fourier: Optional[FourierFixedTrunkConfig] = None
    chebyshev: Optional[ChebyshevFixedTrunkConfig] = None

    out_dim: int = 2
    head_bias: bool = False
    r_min: float = 1e-4

    head_init: str = "small_normal"
    fourier_init_std: float = 1e-2
    cheby_init_std: float = 1e-3

class FixedSpectralPINN(ICEmbeddingMixin, TimeDerivativesMixin, nn.Module):
    def __init__(self, cfg: FixedSpectralPINNConfig):
        super().__init__()
        self.cfg = cfg
        if int(cfg.out_dim) != 2:
            raise ValueError("FixedSpectralPINN currently assumes out_dim=2 for [r, theta] via [rho, theta].")

        # --- Build trunk ---
        if cfg.basis == "fourier":
            if cfg.fourier is None:
                raise ValueError("cfg.fourier must be provided when basis='fourier'.")
            self.trunk: nn.Module = FourierFixedTrunk(cfg.fourier)
        elif cfg.basis == "chebyshev":
            if cfg.chebyshev is None:
                raise ValueError("cfg.chebyshev must be provided when basis='chebyshev'.")
            self.trunk = ChebyshevFixedTrunk(cfg.chebyshev)
        else:
            raise ValueError(f"Unknown basis {cfg.basis!r}")

        trunk_out_dim = getattr(self.trunk, "out_dim")
        if not isinstance(trunk_out_dim, int):
            trunk_out_dim = int(trunk_out_dim)  # type: ignore[arg-type]

        # --- Linear head maps Phi(t)->[rho_tilde, theta_tilde] ---
        self.head = nn.Linear(trunk_out_dim, cfg.out_dim, bias=cfg.head_bias)

        head_init = str(getattr(cfg, "head_init", "small_normal")).lower().strip()

        if head_init == "zeros":
            nn.init.zeros_(self.head.weight)

        elif head_init == "small_normal":
            if cfg.basis == "chebyshev":
                with torch.no_grad():
                    D = self.head.weight.shape[1]
                    base_std = float(cfg.cheby_init_std)
                    p = 2.0
                    ks = torch.arange(D, device=self.head.weight.device, dtype=self.head.weight.dtype)
                    scales = 1.0 / torch.pow(ks + 1.0, p)
                    self.head.weight.normal_(mean=0.0, std=base_std)
                    self.head.weight.mul_(scales.unsqueeze(0))
            else:
                nn.init.normal_(self.head.weight, mean=0.0, std=float(cfg.fourier_init_std))
        else:
            raise ValueError(f"Unknown head_init '{cfg.head_init}' (use 'small_normal' or 'zeros').")

        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        # --- IC buffers ---
        self.register_buffer("_u0_dot", torch.zeros(1, 1))
        self.register_buffer("_rho0_dot", torch.zeros(1, 1))
        self._has_u0_dot = False
        self._has_rho0_dot = False
        self.register_buffer("_u0", torch.zeros(1, cfg.out_dim))   # stores theta0 in slot 1
        self.register_buffer("_rho0", torch.zeros(1, 1))           # stores rho0
        self._has_u0 = False
        self._has_rho0 = False
        self.r_min = float(cfg.r_min)

    @property
    def out_dim(self) -> int:
        return int(self.cfg.out_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_in = self._normalize_time_input(t)

        phi = self.trunk(t_in)
        u_tilde = self.head(phi)  # (N,2) as [rho_tilde, theta_tilde]
        rho_tilde = u_tilde[:, 0:1]
        theta_tilde = u_tilde[:, 1:2]

        return self._embed_ic_and_positivity(t_in, rho_tilde, theta_tilde)
