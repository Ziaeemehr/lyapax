"""Tier 4.1: Mackey-Glass. Same params as
tests/test_dde.py::test_mackey_glass_qualitative_chaos.
"""
import jax.numpy as jnp
import numpy as np

from lyapax.dde import (
    lyapunov_spectrum_dde, resolve_tau_steps,
    make_scalar_delayed_step_fn, scalar_delayed_history0,
)
from lyapax import systems

from _common import time_and_run, emit


def _kaplan_yorke_dimension(exponents: np.ndarray) -> float:
    cumsum = np.cumsum(exponents)
    j = 0
    for i in range(1, len(exponents) + 1):
        if cumsum[i - 1] >= 0:
            j = i
        else:
            break
    if j == 0 or j >= len(exponents):
        return float(j)
    return j + cumsum[j - 1] / abs(exponents[j])


def run():
    beta, gamma, n, tau = 0.2, 0.1, 10.0, 17.0
    dt = 1.0
    tau_steps = resolve_tau_steps(tau, dt)
    rhs = systems.mackey_glass(beta=beta, gamma=gamma, n=n)
    step_fn = make_scalar_delayed_step_fn(rhs, m=1, tau_steps=tau_steps, dt=dt)
    state0, buf0 = scalar_delayed_history0(jnp.array([1.2]), tau_steps)
    return lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
        n_steps=30_000, k=8, renorm_every=10, t_transient=3_000.0,
    )


if __name__ == "__main__":
    first_s, warm_s, result = time_and_run(run)
    exponents = np.array(result.exponents)
    ky = _kaplan_yorke_dimension(exponents)
    emit("lyapax", "mackey_glass_tier4.1", exponents, first_s, warm_s, kaplan_yorke_dim=ky)
