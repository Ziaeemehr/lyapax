"""Tier 1.2/2: Rossler system. Same params as
tests/test_lyapunov_core.py::test_rossler_lambda1_order_of_magnitude.
"""
import jax.numpy as jnp

from lyapax.core import lyapunov_spectrum
from lyapax.integrators import rk4_step
from lyapax import systems

from _common import time_and_run, emit


def run():
    a, b, c = 0.2, 0.2, 5.7
    rhs = systems.rossler(a, b, c)
    dt = 1e-2
    step = rk4_step(rhs, dt)
    return lyapunov_spectrum(
        step, state0=jnp.array([1.0, 1.0, 1.0]),
        dt=dt, n_steps=200_000, renorm_every=10, t_transient=200.0,
    )


if __name__ == "__main__":
    first_s, warm_s, result = time_and_run(run)
    emit("lyapax", "rossler_tier1.2", result.exponents, first_s, warm_s)
