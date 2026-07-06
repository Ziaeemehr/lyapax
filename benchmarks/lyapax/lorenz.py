"""Tier 1.1/2: Lorenz system. Same params as
tests/test_lyapunov_core.py::test_lorenz_lambda1_matches_published_value.
"""
import jax.numpy as jnp

from lyapax.core import ode_problem, lyapunov_spectrum
from lyapax import systems

from _common import time_and_run, emit


def run(integrator):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rhs = systems.lorenz(sigma, rho, beta)
    dt = 1e-2
    problem = ode_problem(rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=dt, integrator=integrator)
    return lyapunov_spectrum(problem, n_steps=50_000, renorm_every=10, t_transient=100.0)


if __name__ == "__main__":
    for integrator, tool in [("rk4", "lyapax"), ("rk6", "lyapax-rk6")]:
        first_s, warm_s, result = time_and_run(run, integrator)
        emit(tool, "lorenz_tier1.1", result.exponents, first_s, warm_s)
