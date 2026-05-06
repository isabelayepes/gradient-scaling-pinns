# training/losses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LossWeights:
    """
    Simple fixed weights for PINN losses.

    phys: weight on balanced physics loss (used for optimization)
    ic_vel: weight on IC velocity loss
    """
    phys: float = 1.0
    ic_vel: float = 1.0


def mse(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x ** 2)


@dataclass
class PhysicsResidualEMA:
    """
    Tracks an exponential moving average (EMA) of per-component RMS(residual).

    We use EMA RMS as a *stable* normalization scale to avoid the "per-batch RMS = 1"
    effect that makes the balanced loss constant and can stall optimization.
    """
    beta: float = 0.99
    eps: float = 1e-12
    ema_rms: Optional[torch.Tensor] = None  # shape (2,)
    steps: int = 0

    def update(self, batch_rms: torch.Tensor) -> torch.Tensor:
        """
        Update EMA from current batch RMS (shape (2,)).
        Returns the updated EMA RMS (shape (2,)).
        """
        if batch_rms.ndim != 1:
            batch_rms = batch_rms.reshape(-1)
        if batch_rms.numel() != 2:
            raise ValueError(f"Expected batch_rms to have 2 elements, got {batch_rms.numel()}")

        batch_rms = batch_rms.detach()

        if self.ema_rms is None:
            self.ema_rms = batch_rms.clone()
        else:
            self.ema_rms = self.beta * self.ema_rms + (1.0 - self.beta) * batch_rms

        self.steps += 1
        return self.ema_rms


def physics_residual_loss(
    t_col: torch.Tensor,
    u: torch.Tensor,
    du_dt: torch.Tensor,
    d2u_dt2: torch.Tensor,
    k: torch.Tensor,
    params,
    residual_fn,
    balance_components: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Physics MSE loss from residual function (unweighted).

    If balance_components=True, normalize each residual component by its batch RMS:
        L = mean((res_r / rms_r)^2) + mean((res_theta / rms_theta)^2)
    """
    res = residual_fn(
        t=t_col,
        u=u,
        du_dt=du_dt,
        d2u_dt2=d2u_dt2,
        k=k,
        params=params,
    )  # (N,2)

    if not balance_components:
        return mse(res)

    rms = torch.sqrt(torch.mean(res**2, dim=0)).detach() + eps  # (2,)
    res_scaled = res / rms  # (N,2)
    return torch.mean(res_scaled[:, 0] ** 2) + torch.mean(res_scaled[:, 1] ** 2)


def physics_residual_loss_weighted(
    t_col: torch.Tensor,
    u: torch.Tensor,
    du_dt: torch.Tensor,
    d2u_dt2: torch.Tensor,
    k: torch.Tensor,
    params,
    residual_fn,
    weights_per_sample: Optional[torch.Tensor] = None,
    balance_components: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Weighted physics loss where each collocation sample i has an importance weight w_i >= 0.

    weights_per_sample shape: (N,) or None.

    Weighted EMA/balancing logic:
      - Compute weighted per-component RMS:
          rms_comp = sqrt( sum_i w_i * res_i^2 / sum_i w_i )
      - Scale residuals by rms_comp (detached)
      - Compute weighted mean of scaled-squared residuals:
          L = sum_i w_i * (res_scaled_i^2) / sum_i w_i  (summing both components)
    """
    res = residual_fn(
        t=t_col,
        u=u,
        du_dt=du_dt,
        d2u_dt2=d2u_dt2,
        k=k,
        params=params,
    )  # (N,2)

    N = res.shape[0]
    if weights_per_sample is None:
        # fallback to unweighted behavior (same as physics_residual_loss)
        if not balance_components:
            return mse(res)
        rms = torch.sqrt(torch.mean(res**2, dim=0)).detach() + eps  # (2,)
        res_scaled = res / rms
        return torch.mean(res_scaled[:, 0] ** 2) + torch.mean(res_scaled[:, 1] ** 2)

    # weights: ensure shape (N,) and nonnegative
    w = weights_per_sample.reshape(-1).to(dtype=res.dtype, device=res.device)
    w = torch.clamp(w, min=0.0)
    sum_w = torch.sum(w) + eps

    # weighted per-component mean square
    # msec = sum_i w_i * (res_i^2) / sum_w  -> shape (2,)
    msec = torch.sum((res ** 2) * w.unsqueeze(1), dim=0) / sum_w
    rms = torch.sqrt(msec + eps).detach()  # (2,) detached scale

    # scale and compute weighted mean of squared scaled residuals
    res_scaled = res / rms  # (N,2)
    weighted_sq = (res_scaled ** 2) * w.unsqueeze(1)  # (N,2)
    loss = torch.sum(weighted_sq) / sum_w
    return loss


def physics_residual_loss_ema(
    t_col: torch.Tensor,
    u: torch.Tensor,
    du_dt: torch.Tensor,
    d2u_dt2: torch.Tensor,
    k: torch.Tensor,
    params,
    residual_fn,
    ema: PhysicsResidualEMA,
    weights_per_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    EMA-balanced physics loss, optionally per-sample weighted.

    Returns:
      loss, stats dict with batch_rms and ema_rms (both components).
    """
    # Evaluate residuals and compute batch RMS (weighted if weights provided)
    res = residual_fn(
        t=t_col,
        u=u,
        du_dt=du_dt,
        d2u_dt2=d2u_dt2,
        k=k,
        params=params,
    )  # (N,2)

    if weights_per_sample is None:
        batch_rms = torch.sqrt(torch.mean(res**2, dim=0) + ema.eps)  # (2,)
    else:
        w = weights_per_sample.reshape(-1).to(dtype=res.dtype, device=res.device)
        w = torch.clamp(w, min=0.0)
        sum_w = torch.sum(w) + ema.eps
        # weighted mean square per component
        msec = torch.sum((res ** 2) * w.unsqueeze(1), dim=0) / sum_w
        batch_rms = torch.sqrt(msec + ema.eps)  # (2,)

    ema_rms = ema.update(batch_rms)  # (2,)

    # Use ema_rms as scale (detached)
    scale = ema_rms.detach() + ema.eps
    res_scaled = res / scale  # (N,2)

    if weights_per_sample is None:
        loss = torch.mean(res_scaled[:, 0] ** 2) + torch.mean(res_scaled[:, 1] ** 2)
    else:
        w = weights_per_sample.reshape(-1).to(dtype=res.dtype, device=res.device)
        w = torch.clamp(w, min=0.0)
        sum_w = torch.sum(w) + ema.eps
        weighted_sq = (res_scaled ** 2) * w.unsqueeze(1)
        loss = torch.sum(weighted_sq) / sum_w

    stats = {
        "batch_rms_r": float(batch_rms[0].detach().cpu().item()),
        "batch_rms_theta": float(batch_rms[1].detach().cpu().item()),
        "ema_rms_r": float(ema_rms[0].detach().cpu().item()),
        "ema_rms_theta": float(ema_rms[1].detach().cpu().item()),
    }
    return loss, stats


def initial_condition_loss(
    u0: torch.Tensor,
    du_dt0: torch.Tensor,
    ic: Dict[str, float],
    ic_residual_fn,
) -> torch.Tensor:
    """
    Compute MSE loss enforcing initial conditions using ic_residual_fn.

    Supports two signatures:
      (A) ic_residual_fn(u0, du_dt0, ic) -> (N0, 4)  # full ICs
      (B) ic_residual_fn(du_dt0, ic)     -> (N0, 2)  # velocity-only ICs
    """
    try:
        res_ic = ic_residual_fn(u0=u0, du_dt0=du_dt0, ic=ic)
    except TypeError as e:
        if "unexpected keyword" not in str(e) and "got an unexpected" not in str(e):
            raise
        res_ic = ic_residual_fn(du_dt0=du_dt0, ic=ic)

    return mse(res_ic)


def build_pinn_loss(
    model: nn.Module,
    t_col: torch.Tensor,
    t0: torch.Tensor,
    k: torch.Tensor,
    ic: Dict[str, float],
    params,
    residual_fn,
    ic_residual_fn,
    weights: LossWeights = LossWeights(),
    ema: Optional[PhysicsResidualEMA] = None,
    point_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Total PINN loss for a single task:
      total = w_phys * L_phys + w_ic_vel * L_ic_vel

    Supports optional per-sample point_weights (shape (N,); detached before being used).
    """
    # Evaluate forward derivatives
    u, du_dt, d2u_dt2 = model.forward_with_derivatives(t_col)

    if ema is None:
        if point_weights is None:
            L_phys_balanced = physics_residual_loss(
                t_col=t_col,
                u=u,
                du_dt=du_dt,
                d2u_dt2=d2u_dt2,
                k=k,
                params=params,
                residual_fn=residual_fn,
                balance_components=True,
            )
            ema_stats: Dict[str, float] = {}
        else:
            L_phys_balanced = physics_residual_loss_weighted(
                t_col=t_col,
                u=u,
                du_dt=du_dt,
                d2u_dt2=d2u_dt2,
                k=k,
                params=params,
                residual_fn=residual_fn,
                weights_per_sample=point_weights.detach(),
                balance_components=True,
            )
            ema_stats = {}
    else:
        L_phys_balanced, ema_stats = physics_residual_loss_ema(
            t_col=t_col,
            u=u,
            du_dt=du_dt,
            d2u_dt2=d2u_dt2,
            k=k,
            params=params,
            residual_fn=residual_fn,
            ema=ema,
            weights_per_sample=(None if point_weights is None else point_weights.detach()),
        )

    # Logging-only raw residual stats (unweighted MSE for interpretability)
    with torch.no_grad():
        res_dbg = residual_fn(
            t=t_col,
            u=u,
            du_dt=du_dt,
            d2u_dt2=d2u_dt2,
            k=k,
            params=params,
        )  # (N,2)

        L_phys_raw = mse(res_dbg)
        rms_r = torch.sqrt(torch.mean(res_dbg[:, 0] ** 2)).item()
        rms_th = torch.sqrt(torch.mean(res_dbg[:, 1] ** 2)).item()

    # IC velocity loss: skip if model enforces velocity IC by construction
    hard_vel = bool(getattr(model, "hard_vel_ic", False))

    if hard_vel:
        L_ic = torch.zeros((), device=u.device, dtype=u.dtype)
    else:
        u0, du_dt0, _ = model.forward_with_derivatives(t0)
        L_ic = initial_condition_loss(
            u0=u0,
            du_dt0=du_dt0,
            ic=ic,
            ic_residual_fn=ic_residual_fn,
        )

    total = float(weights.phys) * L_phys_balanced + float(weights.ic_vel) * L_ic

    logs = {
        "loss_total": float(total.detach().cpu().item()),
        "hard_vel_ic": float(hard_vel),
        "loss_phys_balanced": float(L_phys_balanced.detach().cpu().item()),
        "loss_phys_raw": float(L_phys_raw.detach().cpu().item()),
        "loss_ic_vel": float(L_ic.detach().cpu().item()),
        "rms_res_r": float(rms_r),
        "rms_res_theta": float(rms_th),
        **ema_stats,
    }
    return total, logs
