"""Tier 1.2/2: Rossler system. Same params as
tests/test_lyapunov_core.py::test_rossler_lambda1_order_of_magnitude.
"""
import jax.numpy as jnp

from lyapax.core import ode_problem, lyapunov_spectrum
from lyapax import systems

from _common import time_and_run, emit


def run(integrator):
    a, b, c = 0.2, 0.2, 5.7
    rhs = systems.rossler(a, b, c)
    dt = 1e-2
    problem = ode_problem(rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=dt, integrator=integrator)
    return lyapunov_spectrum(problem, n_steps=200_000, renorm_every=10, t_transient=200.0)


if __name__ == "__main__":
    for integrator, tool in [("rk4", "lyapax"), ("rk6", "lyapax-rk6")]:
        first_s, warm_s, result = time_and_run(run, integrator)
        emit(tool, "rossler_tier1.2", result.exponents, first_s, warm_s)
