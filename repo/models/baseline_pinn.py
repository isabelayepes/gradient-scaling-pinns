# models/baseline_pinn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mixins import ICEmbeddingMixin, TimeDerivativesMixin


@dataclass(frozen=True)
class BaselinePINNConfig:
    hidden_layers: Tuple[int, ...] = (128, 128, 128)
    activation: str = "tanh"
    out_dim: int = 2
    r_min: float = 1e-4
    ic_embed: bool = True   # enforce u(0)=u0 exactly (recommended for fair comparison)


def _make_activation(name: str) -> nn.Module:
    name = name.lower().strip()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


class BaselinePINN(ICEmbeddingMixin, TimeDerivativesMixin, nn.Module):
    """
    Standard MLP baseline:
      u_tilde(t) = MLP(t) -> [rho_tilde, theta_tilde]
      r(t) = r_min + softplus(rho)
      theta(t) = theta

    With optional IC embedding:
      rho_hat   = rho0   + t * rho_tilde(t)
      theta_hat = theta0 + t * theta_tilde(t)
      r_hat     = r_min + softplus(rho_hat)
    """

    def __init__(self, cfg: BaselinePINNConfig):
        super().__init__()
        self.cfg = cfg
        self.r_min = float(cfg.r_min)
        self.ic_embed = bool(cfg.ic_embed)

        act = _make_activation(cfg.activation)

        layers = []
        in_dim = 1
        for h in cfg.hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act)
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.out_dim))
        self.net = nn.Sequential(*layers)

        # IC buffers
        self.register_buffer("_u0_dot", torch.zeros(1, 1))
        self.register_buffer("_rho0_dot", torch.zeros(1, 1))
        self._has_u0_dot = False
        self._has_rho0_dot = False
        self.register_buffer("_u0", torch.zeros(1, cfg.out_dim))   # store theta0 in slot 1
        self.register_buffer("_rho0", torch.zeros(1, 1))           # store rho0
        self._has_u0 = False
        self._has_rho0 = False

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_in = self._normalize_time_input(t)  # from TimeDerivativesMixin

        u_tilde = self.net(t_in)
        rho_tilde = u_tilde[:, 0:1]
        theta_tilde = u_tilde[:, 1:2]

        if not self.ic_embed:
            r = self.r_min + F.softplus(rho_tilde)
            return torch.cat([r, theta_tilde], dim=1)

        return self._embed_ic_and_positivity(t_in, rho_tilde, theta_tilde)
