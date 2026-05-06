# problem/spring_pendulum.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


# ----------------------------
# Parameters / configuration
# ----------------------------
@dataclass(frozen=True)
class SpringPendulumParams:
    """
    Physical parameters for the planar spring–pendulum in polar coordinates.

    Governing equations (as used in the thesis draft):

      r_ddot     = r * theta_dot^2 - (k/m) * (r - L0) + g * cos(theta) - (c_r/m) * r_dot
      theta_ddot = - (2 r_dot theta_dot)/r - (g/r) * sin(theta) - c_theta * theta_dot

    Notes:
    - r should remain > 0 to avoid division by zero in theta equation.
    - c_theta is used as a direct damping coefficient on theta_dot (as in your draft).
      If you later want physical torque damping, you can revisit this term.
    """
    g: float = 9.81
    m: float = 1.0
    L0: float = 1.0
    c_r: float = 0.0
    c_theta: float = 0.0


# ----------------------------
# Utilities
# ----------------------------
def _get_tensor_param(problem_cfg: dict, key: str, default: float) -> float:
    if "problem" not in problem_cfg:
        raise KeyError("Expected cfg to have top-level key 'problem'.")
    return float(problem_cfg["problem"].get(key, default))


def load_params_from_cfg(cfg: dict) -> SpringPendulumParams:
    """
    Load SpringPendulumParams from a config dict matching configs/stage1.yaml.
    """
    g = _get_tensor_param(cfg, "g", 9.81)
    m = _get_tensor_param(cfg, "m", 1.0)
    L0 = _get_tensor_param(cfg, "L0", 1.0)
    c_r = _get_tensor_param(cfg, "c_r", 0.0)
    c_theta = _get_tensor_param(cfg, "c_theta", 0.0)
    return SpringPendulumParams(g=g, m=m, L0=L0, c_r=c_r, c_theta=c_theta)


# ----------------------------
# Physics: RHS and residuals
# ----------------------------
def rhs_first_order(
    t: torch.Tensor,
    y: torch.Tensor,
    k: torch.Tensor,
    params: SpringPendulumParams,
    r_eps: float = 1e-6,
) -> torch.Tensor:
    """
    First-order RHS for numerical integration / reference generation.

    State y = [r, theta, rdot, thetadot] with shape (..., 4)
    Time t is unused (autonomous) but included for signature consistency.

    k: spring constant tensor broadcastable to y[..., 0]
    Returns dy/dt with shape (..., 4)
    """
    r = y[..., 0]
    theta = y[..., 1]
    rdot = y[..., 2]
    thetadot = y[..., 3]

    # Prevent division by zero for r in theta equation.
    r_safe = torch.clamp(r, min=r_eps)  # should rarely activate if model enforces r>0

    rddot = (
        r_safe * thetadot**2
        - (k / params.m) * (r_safe - params.L0)
        + params.g * torch.cos(theta)
        - (params.c_r / params.m) * rdot
    )

    thetaddot = (
        - (2.0 * rdot * thetadot) / r_safe
        - (params.g / r_safe) * torch.sin(theta)
        - params.c_theta * thetadot
    )

    dy = torch.stack([rdot, thetadot, rddot, thetaddot], dim=-1)
    return dy


def unpack_uv_to_state(
    u: torch.Tensor,
    du_dt: torch.Tensor,
    d2u_dt2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper to map network outputs to physical variables.

    Expected:
      u         shape (N, 2): [r, theta]
      du_dt     shape (N, 2): [r_dot, theta_dot]
      d2u_dt2   shape (N, 2): [r_ddot, theta_ddot]
    """
    r = u[:, 0]
    theta = u[:, 1]
    rdot = du_dt[:, 0]
    thetadot = du_dt[:, 1]
    rddot = d2u_dt2[:, 0]
    thetaddot = d2u_dt2[:, 1]
    return r, theta, rdot, thetadot, rddot, thetaddot


def residual_second_order(
    t: torch.Tensor,
    u: torch.Tensor,
    du_dt: torch.Tensor,
    d2u_dt2: torch.Tensor,
    k: torch.Tensor,
    params: SpringPendulumParams,
    r_eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the physics residuals for the second-order spring–pendulum equations
    in polar coordinates.

    Inputs:
      t         shape (N, 1) or (N,) : time collocation points (not used explicitly)
      u         shape (N, 2)         : [r(t), theta(t)]
      du_dt     shape (N, 2)         : [r_dot(t), theta_dot(t)]
      d2u_dt2   shape (N, 2)         : [r_ddot(t), theta_ddot(t)]
      k         shape (N,) or (N,1) or scalar tensor : spring constant (task parameter)
      params    SpringPendulumParams
    Returns:
      res       shape (N, 2)         : [res_r, res_theta]
    """
    # Ensure shapes are compatible
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError(f"Expected u shape (N,2), got {tuple(u.shape)}")
    if du_dt.shape != u.shape or d2u_dt2.shape != u.shape:
        raise ValueError(f"du_dt and d2u_dt2 must match u shape (N,2); got {tuple(du_dt.shape)}, {tuple(d2u_dt2.shape)}")

    # Broadcast k to (N,)
    k_flat = k.reshape(-1)
    if k_flat.numel() == 1:
        k_flat = k_flat.expand(u.shape[0])
    elif k_flat.numel() != u.shape[0]:
        raise ValueError(f"k must be scalar or have N elements; got {k_flat.numel()} vs N={u.shape[0]}")

    r, theta, rdot, thetadot, rddot, thetaddot = unpack_uv_to_state(u, du_dt, d2u_dt2)
    r_safe = torch.clamp(r, min=r_eps)  # should rarely activate if model enforces r>0

    # Model-predicted accelerations (from the ODE)
    rddot_model = (
        r_safe * thetadot**2
        - (k_flat / params.m) * (r_safe - params.L0)
        + params.g * torch.cos(theta)
        - (params.c_r / params.m) * rdot
    )

    thetaddot_model = (
        - (2.0 * rdot * thetadot) / r_safe
        - (params.g / r_safe) * torch.sin(theta)
        - params.c_theta * thetadot
    )

    # Residuals: predicted second derivative minus physics model second derivative
    res_r = rddot - rddot_model
    res_theta = thetaddot - thetaddot_model

    return torch.stack([res_r, res_theta], dim=1)


# ----------------------------
# Initial condition helpers
# ----------------------------
def ic_residual_vel_only(
    du_dt0: torch.Tensor,
    ic: Dict[str, float],
) -> torch.Tensor:
    """
    Velocity-only IC residual at t=0.

    Inputs:
      du_dt0: shape (N0,2): predicted [r_dot(0), theta_dot(0)]
      ic: dict with keys rdot0, thetadot0

    Returns:
      res_vel shape (N0, 2): [r_dot-rdot0, theta_dot-thetadot0]
    """
    for key in ["rdot0", "thetadot0"]:
        if key not in ic:
            raise KeyError(f"ic missing key '{key}'")

    if du_dt0.ndim != 2 or du_dt0.shape[1] != 2:
        raise ValueError(f"Expected du_dt0 shape (N0,2), got {tuple(du_dt0.shape)}")

    rd0 = torch.as_tensor(float(ic["rdot0"]), device=du_dt0.device, dtype=du_dt0.dtype)
    thd0 = torch.as_tensor(float(ic["thetadot0"]), device=du_dt0.device, dtype=du_dt0.dtype)

    res = torch.stack([du_dt0[:, 0] - rd0, du_dt0[:, 1] - thd0], dim=1)
    return res

def ic_residual(
    u0: torch.Tensor,
    du_dt0: torch.Tensor,
    ic: Dict[str, float],
) -> torch.Tensor:
    """
    Residual enforcing initial conditions at t=0.

    Inputs:
      u0     shape (1,2) or (N0,2): predicted [r(0), theta(0)]
      du_dt0 shape (1,2) or (N0,2): predicted [r_dot(0), theta_dot(0)]
      ic: dict with keys: r0, theta0, rdot0, thetadot0

    Returns:
      res_ic shape (N0, 4): [r-r0, theta-theta0, rdot-rdot0, thetadot-thetadot0]
    """
    for key in ["r0", "theta0", "rdot0", "thetadot0"]:
        if key not in ic:
            raise KeyError(f"ic missing key '{key}'")

    if u0.ndim != 2 or u0.shape[1] != 2:
        raise ValueError(f"Expected u0 shape (N0,2), got {tuple(u0.shape)}")
    if du_dt0.shape != u0.shape:
        raise ValueError(f"Expected du_dt0 shape to match u0; got {tuple(du_dt0.shape)} vs {tuple(u0.shape)}")

    r0 = torch.as_tensor(float(ic["r0"]), device=u0.device, dtype=u0.dtype)
    th0 = torch.as_tensor(float(ic["theta0"]), device=u0.device, dtype=u0.dtype)
    rd0 = torch.as_tensor(float(ic["rdot0"]), device=u0.device, dtype=u0.dtype)
    thd0 = torch.as_tensor(float(ic["thetadot0"]), device=u0.device, dtype=u0.dtype)

    res = torch.stack(
        [u0[:, 0] - r0, u0[:, 1] - th0, du_dt0[:, 0] - rd0, du_dt0[:, 1] - thd0],
        dim=1,
    )
    return res


# ----------------------------
# Convenience: task -> dict
# ----------------------------
def task_to_ic_dict(task) -> Dict[str, float]:
    """
    Accepts a Task-like object (e.g., from problem.task_sampling.Task) and returns
    IC dict for ic_residual.
    """
    return {
        "r0": float(task.r0),
        "theta0": float(task.theta0),
        "rdot0": float(task.rdot0),
        "thetadot0": float(task.thetadot0),
    }
