# models/cascaded_pinn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models.mixins import ICEmbeddingMixin, TimeDerivativesMixin


@dataclass(frozen=True)
class MLPConfig:
    in_dim: int = 1
    hidden_layers: Tuple[int, ...] = (128, 128, 128)
    activation: str = "tanh"
    out_dim: int = 2  # predicts [rho_tilde, theta_tilde]
    bias: bool = True


def _make_activation(name: str) -> nn.Module:
    name = name.lower().strip()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation '{name}' (use tanh/silu/relu).")


class SimpleMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        layers: List[nn.Module] = []
        act = _make_activation(cfg.activation)

        d = cfg.in_dim
        for h in cfg.hidden_layers:
            layers.append(nn.Linear(d, h, bias=cfg.bias))
            layers.append(act)
            d = h

        layers.append(nn.Linear(d, cfg.out_dim, bias=cfg.bias))
        self.net = nn.Sequential(*layers)

        # small init helps second-derivative PINNs
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


@dataclass(frozen=True)
class CascadedPINNConfig:
    """
    Cascaded residual PINN (single-task baseline).

    Latent output is additive:
        u_tilde(t) = sum_{b=1..B} u_tilde_b(t)

    IC embedding + positivity is applied AFTER summation, using ICEmbeddingMixin.

    By design, training code will:
      - train blocks sequentially,
      - freeze previous blocks by default,
      - optionally do residual-weighted sampling or reweighting as an ablation.
    """
    blocks: int = 3
    mlp: MLPConfig = MLPConfig()
    r_min: float = 1e-4
    out_dim: int = 2  # [r, theta]


class CascadedPINN(ICEmbeddingMixin, TimeDerivativesMixin, nn.Module):
    def __init__(self, cfg: CascadedPINNConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.blocks < 1:
            raise ValueError("cfg.blocks must be >= 1")
        if cfg.out_dim != 2:
            raise ValueError("CascadedPINN assumes out_dim=2 for [r, theta] via [rho, theta].")

        self.blocks = nn.ModuleList([SimpleMLP(cfg.mlp) for _ in range(cfg.blocks)])
        self.r_min = float(cfg.r_min)

        # IC buffers (same convention as your spectral models)
        self.register_buffer("_u0_dot", torch.zeros(1, 1))
        self.register_buffer("_rho0_dot", torch.zeros(1, 1))
        self._has_u0_dot = False
        self._has_rho0_dot = False
        self.register_buffer("_u0", torch.zeros(1, cfg.out_dim))  # stores theta0 in slot 1
        self.register_buffer("_rho0", torch.zeros(1, 1))          # stores rho0
        self._has_u0 = False
        self._has_rho0 = False

        # training controls which blocks are active (sum over 0..k)
        # _active_stage=None => sum all blocks
        self._active_stage: Optional[int] = None

    # -------------------------
    # Block control (for training)
    # -------------------------
    def set_active_stage(self, stage_idx: Optional[int]) -> None:
        """
        stage_idx:
          - None: all blocks active (sum over all)
          - int k: only blocks <= k are included in forward
        """
        if stage_idx is not None:
            if stage_idx < 0 or stage_idx >= len(self.blocks):
                raise ValueError(f"stage_idx out of range: {stage_idx}")
        self._active_stage = stage_idx

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def freeze_only_block(self, stage_idx: int) -> None:
        """
        Freeze all blocks except stage_idx.
        (This is the simple behavior used by the cascaded trainer.)
        """
        for i, blk in enumerate(self.blocks):
            req = (i == stage_idx)
            for p in blk.parameters():
                p.requires_grad_(req)

    # -------------------------
    # Forward
    # -------------------------
    def _tilde_sum(self, t_in: torch.Tensor) -> torch.Tensor:
        # t_in: (N,1)
        if self._active_stage is None:
            upto = len(self.blocks) - 1
        else:
            upto = self._active_stage

        out = 0.0
        for i in range(upto + 1):
            out = out + self.blocks[i](t_in)
        return out  # (N,2) [rho_tilde_sum, theta_tilde_sum]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_in = self._normalize_time_input(t)

        u_tilde = self._tilde_sum(t_in)
        rho_tilde = u_tilde[:, 0:1]
        theta_tilde = u_tilde[:, 1:2]

        return self._embed_ic_and_positivity(t_in, rho_tilde, theta_tilde)
