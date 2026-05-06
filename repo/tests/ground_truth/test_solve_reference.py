# tests/ground_truth/test_solve_reference.py
# run with: python3 -m tests.ground_truth.test_solve_reference

from problem.spring_pendulum import SpringPendulumParams
from ground_truth.solve_reference import solve_reference_task
import numpy as np


def test_solve_reference_basic():
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

    # Shape checks
    assert ref["r"].shape == t_eval.shape
    assert ref["theta"].shape == t_eval.shape

    # Finite values
    assert np.all(np.isfinite(ref["r"]))
    assert np.all(np.isfinite(ref["theta"]))

    # Initial conditions respected
    assert abs(ref["r"][0] - ic["r0"]) < 1e-6
    assert abs(ref["theta"][0] - ic["theta0"]) < 1e-6


if __name__ == "__main__":
    test_solve_reference_basic()
    print("✓ solve_reference_task basic test passed")
