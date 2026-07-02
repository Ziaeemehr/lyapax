"""Tier 0.1: linear ODE, 3 distinct real eigenvalues. Same system/params as
tests/test_lyapunov_core.py::test_linear_system_distinct_real_eigenvalues.
"""
import jax.numpy as jnp

from lyapax.core import lyapunov_spectrum
from lyapax.integrators import rk4_step
from lyapax import systems

from _common import time_and_run, emit


def run():
    A = jnp.diag(jnp.array([-1.0, -2.0, -5.0]))
    rhs = systems.linear_system(A)
    dt = 1e-3
    step = rk4_step(rhs, dt)
    return lyapunov_spectrum(
        step, state0=jnp.array([0.3, -0.2, 0.5]),
        dt=dt, n_steps=20_000, renorm_every=10, t_transient=5.0,
    )


if __name__ == "__main__":
    first_s, warm_s, result = time_and_run(run)
    emit("lyapax", "linear_ode_tier0.1", result.exponents, first_s, warm_s)
