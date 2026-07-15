"""Tier 4.1: Mackey-Glass. Same params as
tests/test_dde.py::test_mackey_glass_qualitative_chaos.
"""
from _common import time_and_run, emit  # noqa: I001 -- must set JAX_PLATFORMS before jax import

import jax.numpy as jnp
import numpy as np

from lyapax.core import kaplan_yorke_dimension
from lyapax.dde import dde_problem, lyapunov_spectrum_dde
from lyapax import systems


def run(integrator):
    beta, gamma, n, tau = 0.2, 0.1, 10.0, 17.0
    dt = 1.0
    rhs = systems.mackey_glass(beta=beta, gamma=gamma, n=n)
    problem = dde_problem(
        rhs, state0=jnp.array([1.2]), tau=tau, dt=dt, integrator=integrator,
    )
    return lyapunov_spectrum_dde(
        problem, n_steps=30_000, k=8, renorm_every=10, t_transient=3_000.0,
    )


if __name__ == "__main__":
    for integrator, tool in [("heun", "lyapax"), ("rk6", "lyapax-rk6")]:
        first_s, warm_s, result = time_and_run(run, integrator)
        exponents = np.array(result.exponents)
        ky = kaplan_yorke_dimension(exponents)
        emit(tool, "mackey_glass_tier4.1", exponents, first_s, warm_s, kaplan_yorke_dim=ky)
