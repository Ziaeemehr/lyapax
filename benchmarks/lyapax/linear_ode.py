"""Tier 0.1: linear ODE, 3 distinct real eigenvalues. Same system/params as
tests/test_lyapunov_core.py::test_linear_system_distinct_real_eigenvalues.
"""
from _common import time_and_run, emit  # noqa: I001 -- must set JAX_PLATFORMS before jax import

import jax.numpy as jnp

from lyapax.core import ode_problem, lyapunov_spectrum
from lyapax import systems


def run(integrator):
    A = jnp.diag(jnp.array([-1.0, -2.0, -5.0]))
    rhs = systems.linear_system(A)
    dt = 1e-3
    problem = ode_problem(rhs, state0=jnp.array([0.3, -0.2, 0.5]), dt=dt, integrator=integrator)
    return lyapunov_spectrum(problem, n_steps=20_000, renorm_every=10, t_transient=5.0)


if __name__ == "__main__":
    for integrator, tool in [("rk4", "lyapax"), ("rk6", "lyapax-rk6")]:
        first_s, warm_s, result = time_and_run(run, integrator)
        emit(tool, "linear_ode_tier0.1", result.exponents, first_s, warm_s)
