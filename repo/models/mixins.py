# models/mixins.py
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICEmbeddingMixin:
    """
    Shared IC embedding + positivity logic for [r, theta] models with
    representation r(t) = r_min + softplus(rho(t)).

    Convention:
      - The model predicts latent outputs [rho_tilde(t), theta_tilde(t)].
      - Physical radius is enforced positive by:
            r(t) = r_min + softplus(rho_hat(t))
      - Position ICs are enforced by construction in latent variables:
            rho_hat(t)   = rho0   + g(t) * rho_tilde(t)
            theta_hat(t) = theta0 + g(t) * theta_tilde(t)
        where g(t) = 1 - exp(-t).

      - Optional HARD velocity ICs (ablation) use:
            rho_hat(t)   = rho0   + g(t) * rho_dot0   + g(t)^2 * rho_tilde(t)
            theta_hat(t) = theta0 + g(t) * theta_dot0 + g(t)^2 * theta_tilde(t)

    Assumes the subclass defines buffers/flags:
      - self.r_min: float
      - self._u0:       (1,2) buffer (theta0 stored in [:,1])
      - self._rho0:     (1,1) buffer
      - self._u0_dot:   (1,1) buffer for theta_dot0
      - self._rho0_dot: (1,1) buffer for rho_dot0
      - self._has_u0, self._has_rho0, self._has_u0_dot, self._has_rho0_dot: bool
    """

    _u0: torch.Tensor
    _rho0: torch.Tensor
    _u0_dot: torch.Tensor
    _rho0_dot: torch.Tensor
    _has_u0: bool
    _has_rho0: bool
    _has_u0_dot: bool
    _has_rho0_dot: bool
    r_min: float

    @property
    def hard_vel_ic(self) -> bool:
        """
        True iff velocity ICs are enforced by construction (i.e., dot buffers set).
        Training code can use this to skip the velocity IC loss term.
        """
        return bool(
            self._has_u0
            and self._has_rho0
            and self._has_u0_dot
            and self._has_rho0_dot
        )
    
    def _ic_gate(self, t_in: torch.Tensor) -> torch.Tensor:
        """
        Gate function for IC embedding.

        Modes:
        - "exp":      g(t)=1-exp(-alpha*t)
        - "linear":   g(t)=t
        - "identity": g(t)=1
        """
        mode = str(getattr(self, "ic_gate", "exp")).lower().strip()
        if mode == "exp":
            alpha = float(getattr(self, "ic_gate_alpha", 1.0))
            return 1.0 - torch.exp(-alpha * t_in)

        if mode == "linear":
            return t_in

        if mode == "identity":
            return torch.ones_like(t_in)

        raise ValueError(f"Unknown ic_gate='{mode}'. Use 'exp', 'linear', or 'identity'.")

    def set_task_ic(self, ic: dict) -> None:
        """
        Set IC buffers for the current task.

        Expected keys (always):
          - r0, theta0

        Optional keys (only used if ic["hard_velocity"] == True):
          - rdot0, thetadot0

        Control flag:
          - hard_velocity: bool (default False)
        """
        r0 = float(ic["r0"])
        th0 = float(ic["theta0"])

        # ---- velocity IC handling (hard vs soft) ----
        hard_velocity = bool(ic.get("hard_velocity", False))

        # ---- position IC (always enforced) ----
        target = max(r0 - float(self.r_min), 1e-8)
        device = self._rho0.device
        dtype = self._rho0.dtype
        target_t = torch.tensor(target, device=device, dtype=dtype)

        # rho0 such that softplus(rho0) = target  (inv-softplus)
        rho0 = torch.log(torch.expm1(target_t).clamp_min(1e-12))

        with torch.no_grad():
            # theta0 in slot 1 of u0
            self._u0[0, 1] = torch.tensor(th0, device=self._u0.device, dtype=self._u0.dtype)
            self._rho0[0, 0] = rho0

        self._has_u0 = True
        self._has_rho0 = True

        # ---- velocity IC (optional) ----
        if hard_velocity:
            if "rdot0" not in ic or "thetadot0" not in ic:
                raise KeyError("Hard velocity IC requires 'rdot0' and 'thetadot0'.")

            rdot0 = float(ic["rdot0"])
            thdot0 = float(ic["thetadot0"])

            # Map physical rdot0 to latent rho_dot0:
            # rdot(0) = sigmoid(rho0) * rho_dot(0)
            sigma0 = torch.sigmoid(rho0).clamp_min(1e-8)
            rho_dot0 = torch.tensor(rdot0, device=device, dtype=dtype) / sigma0

            with torch.no_grad():
                self._u0_dot[0, 0] = torch.tensor(thdot0, device=self._u0_dot.device, dtype=self._u0_dot.dtype)
                self._rho0_dot[0, 0] = rho_dot0

            self._has_u0_dot = True
            self._has_rho0_dot = True
        else:
            # Soft velocity IC: do NOT embed velocity
            self._has_u0_dot = False
            self._has_rho0_dot = False

    def _embed_ic_and_positivity(
        self,
        t_in: torch.Tensor,           # (N,1)
        rho_tilde: torch.Tensor,      # (N,1)
        theta_tilde: torch.Tensor,    # (N,1)
    ) -> torch.Tensor:
        """
        Shared mapping from [rho_tilde, theta_tilde] to [r, theta],
        with optional IC embedding (position-only or position+velocity).
        """
        # No IC set yet: just positivity + passthrough
        if not (self._has_u0 and self._has_rho0):
            r = float(self.r_min) + F.softplus(rho_tilde)
            theta = theta_tilde
            return torch.cat([r, theta], dim=1)

        hard_vel = bool(self.hard_vel_ic)
        g = self._ic_gate(t_in)  # (N,1)

        if hard_vel:
            g2 = g * g
            rho_hat = self._rho0 + g * self._rho0_dot + g2 * rho_tilde
            theta_hat = self._u0[:, 1:2] + g * self._u0_dot + g2 * theta_tilde
        else:
            rho_hat = self._rho0 + g * rho_tilde
            theta_hat = self._u0[:, 1:2] + g * theta_tilde

        r_hat = float(self.r_min) + F.softplus(rho_hat)
        return torch.cat([r_hat, theta_hat], dim=1)


class TimeDerivativesMixin:
    """
    Shared autograd boilerplate for forward_with_derivatives(t).

    Assumes the subclass has a forward(t_in) -> u(t_in) with t_in shape (N,1).
    """

    def _normalize_time_input(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            return t[:, None]
        if t.ndim == 2 and t.shape[1] == 1:
            return t
        raise ValueError(f"Expected t shape (N,) or (N,1), got {tuple(t.shape)}")

    def forward_with_derivatives(
        self,
        t: torch.Tensor,
        create_graph: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import torch.autograd  # local import to avoid circulars

        t_in = self._normalize_time_input(t)
        t_in = t_in.clone().detach().requires_grad_(True)

        # u(t)
        u = self.forward(t_in)

        internal_create_graph = True  # needed for second derivatives

        # du/dt
        du_dt = []
        for j in range(u.shape[1]):
            duj = torch.autograd.grad(
                outputs=u[:, j],
                inputs=t_in,
                grad_outputs=torch.ones_like(u[:, j]),
                create_graph=internal_create_graph,
                retain_graph=True,
                only_inputs=True,
            )[0]
            du_dt.append(duj)
        du_dt_full = torch.cat(du_dt, dim=1)

        # d2u/dt2
        d2u = []
        for j in range(u.shape[1]):
            d2uj = torch.autograd.grad(
                outputs=du_dt_full[:, j],
                inputs=t_in,
                grad_outputs=torch.ones_like(du_dt_full[:, j]),
                create_graph=internal_create_graph,
                retain_graph=True,
                only_inputs=True,
            )[0]
            d2u.append(d2uj)
        d2u_full = torch.cat(d2u, dim=1)

        if not create_graph:
            u = u.detach()
            du_dt_full = du_dt_full.detach()
            d2u_full = d2u_full.detach()

        return u, du_dt_full, d2u_full
