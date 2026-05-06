# ground_truth/solve_reference.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from problem.spring_pendulum import SpringPendulumParams


# ----------------------------
# RHS for solve_ivp (numpy)
# ----------------------------
def spring_pendulum_rhs(
    t: float,
    y: np.ndarray,
    k: float,
    params: SpringPendulumParams,
    r_eps: float = 1e-6,
) -> np.ndarray:
    """
    First-order RHS for the planar spring–pendulum system.

    State y = [r, theta, rdot, thetadot]

    Returns dy/dt as a numpy array of shape (4,).
    """
    r, theta, rdot, thetadot = y

    r_safe = max(r, r_eps)

    rddot = (
        r_safe * thetadot**2
        - (k / params.m) * (r_safe - params.L0)
        + params.g * np.cos(theta)
        - (params.c_r / params.m) * rdot
    )

    thetaddot = (
        - (2.0 * rdot * thetadot) / r_safe
        - (params.g / r_safe) * np.sin(theta)
        - params.c_theta * thetadot
    )

    return np.array([rdot, thetadot, rddot, thetaddot], dtype=np.float64)


# ----------------------------
# Single-task reference solve
# ----------------------------
def solve_reference_task(
    ic: Dict[str, float],
    k: float,
    params: SpringPendulumParams,
    t_span: Tuple[float, float],
    t_eval: np.ndarray,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Solve a single spring–pendulum task with high accuracy.

    Inputs:
      ic: dict with keys r0, theta0, rdot0, thetadot0
      k: spring constant
      params: SpringPendulumParams
      t_span: (t0, t1)
      t_eval: 1D numpy array of evaluation times

    Returns dict with:
      t      : (N,)
      r      : (N,)
      theta : (N,)
      rdot   : (N,)
      thetadot : (N,)
    """
    y0 = np.array(
        [ic["r0"], ic["theta0"], ic["rdot0"], ic["thetadot0"]],
        dtype=np.float32,
    )

    sol = solve_ivp(
        fun=lambda t, y: spring_pendulum_rhs(t, y, k, params),
        t_span=t_span,
        y0=y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    r = sol.y[0]
    theta = sol.y[1]
    rdot = sol.y[2]
    thetadot = sol.y[3]

    return {
        "t": sol.t,
        "r": r,
        "theta": theta,
        "rdot": rdot,
        "thetadot": thetadot,
    }


# ----------------------------
# Convenience: stack for PINN comparison
# ----------------------------
def reference_to_state_matrix(ref: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert reference dict to (N, 2) matrix [r(t), theta(t)].
    """
    return np.stack([ref["r"], ref["theta"]], axis=1)


def reference_to_full_state(ref: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert reference dict to (N, 4) matrix [r, theta, rdot, thetadot].
    """
    return np.stack(
        [ref["r"], ref["theta"], ref["rdot"], ref["thetadot"]],
        axis=1,
    )
