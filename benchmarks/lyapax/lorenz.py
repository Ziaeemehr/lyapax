"""Tier 1.1/2: Lorenz system. Same params as
tests/test_lyapunov_core.py::test_lorenz_lambda1_matches_published_value.
"""
import jax.numpy as jnp

from lyapax.core import lyapunov_spectrum
from lyapax.integrators import rk4_step
from lyapax import systems

from _common import time_and_run, emit


def run():
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rhs = systems.lorenz(sigma, rho, beta)
    dt = 1e-2
    step = rk4_step(rhs, dt)
    return lyapunov_spectrum(
        step, state0=jnp.array([1.0, 1.0, 1.0]),
        dt=dt, n_steps=50_000, renorm_every=10, t_transient=100.0,
    )


if __name__ == "__main__":
    first_s, warm_s, result = time_and_run(run)
    emit("lyapax", "lorenz_tier1.1", result.exponents, first_s, warm_s)
