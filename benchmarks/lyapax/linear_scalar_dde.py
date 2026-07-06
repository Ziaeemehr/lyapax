"""Tier 4.2: linear scalar DDE, x'(t) = -a*x(t-tau). Same params as
tests/test_dde.py::test_linear_scalar_dde_matches_lambert_w_root.
"""
import jax.numpy as jnp

from lyapax.dde import dde_problem, lyapunov_spectrum_dde
from lyapax import systems

from _common import time_and_run, emit


def run(integrator):
    a, tau, dt = 0.5, 0.3, 1e-2
    rhs = systems.linear_scalar_dde(a=a)
    problem = dde_problem(
        rhs, state0=jnp.array([1.0]), tau=tau, dt=dt, integrator=integrator,
    )
    return lyapunov_spectrum_dde(
        problem, n_steps=20_000, k=1, renorm_every=5, t_transient=10.0,
    )


if __name__ == "__main__":
    for integrator, tool in [("heun", "lyapax"), ("rk6", "lyapax-rk6")]:
        first_s, warm_s, result = time_and_run(run, integrator)
        emit(tool, "linear_scalar_dde_tier4.2", result.exponents, first_s, warm_s)
