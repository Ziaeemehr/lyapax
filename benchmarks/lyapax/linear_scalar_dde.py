"""Tier 4.2: linear scalar DDE, x'(t) = -a*x(t-tau). Same params as
tests/test_dde.py::test_linear_scalar_dde_matches_lambert_w_root.
"""
import jax.numpy as jnp

from lyapax.dde import (
    lyapunov_spectrum_dde, resolve_tau_steps,
    make_scalar_delayed_step_fn, scalar_delayed_history0,
)
from lyapax import systems

from _common import time_and_run, emit


def run():
    a, tau, dt = 0.5, 0.3, 1e-2
    tau_steps = resolve_tau_steps(tau, dt)
    rhs = systems.linear_scalar_dde(a=a)
    step_fn = make_scalar_delayed_step_fn(rhs, m=1, tau_steps=tau_steps, dt=dt)
    state0, buf0 = scalar_delayed_history0(jnp.array([1.0]), tau_steps)
    return lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
        n_steps=20_000, k=1, renorm_every=5, t_transient=10.0,
    )


if __name__ == "__main__":
    first_s, warm_s, result = time_and_run(run)
    emit("lyapax", "linear_scalar_dde_tier4.2", result.exponents, first_s, warm_s)
