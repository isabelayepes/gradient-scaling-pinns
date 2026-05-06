# experiments/single_reference_trajectory.py
# run with: python3 -m experiments.single_reference_trajectory

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from problem.spring_pendulum import SpringPendulumParams
from ground_truth.solve_reference import solve_reference_task


# -------------------------
# Output configuration
# -------------------------
OUT_DIR = Path("outputs/reference")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    params = SpringPendulumParams()
    ic = dict(r0=1.0, theta0=0.2, rdot0=0.0, thetadot0=0.0)
    t_eval = np.linspace(0.0, 10.0, 2000)

    ref = solve_reference_task(
        ic=ic,
        k=20.0,
        params=params,
        t_span=(0.0, 10.0),
        t_eval=t_eval,
    )

    # -------------------------
    # Time series
    # -------------------------
    plt.figure()
    plt.plot(ref["t"], ref["r"], label="r(t)")
    plt.plot(ref["t"], ref["theta"], label="theta(t)")
    plt.legend()
    plt.xlabel("t")
    plt.title("Spring–Pendulum Time Series")
    plt.savefig(OUT_DIR / "reference_time_series.png", dpi=150)
    plt.close()

    # -------------------------
    # Cartesian trajectory
    # -------------------------
    x = ref["r"] * np.sin(ref["theta"])
    y = ref["r"] * np.cos(ref["theta"])

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Spring–Pendulum Trajectory")
    plt.axis("equal")
    plt.savefig(OUT_DIR / "reference_trajectory.png", dpi=150)
    plt.close()

    print(f"Saved reference plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
