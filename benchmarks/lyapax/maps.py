"""Tier 0.2/0.3: logistic map, tent map, Henon map. Same params as
tests/test_lyapunov_core.py's map tests.
"""
import jax.numpy as jnp

from lyapax.core import ODEProblem, lyapunov_spectrum
from lyapax import systems

from _common import time_and_run, emit


def run_logistic():
    step = systems.logistic_map(r=4.0)
    problem = ODEProblem(step_fn=step, state0=jnp.array([0.4]), dt=1.0)
    return lyapunov_spectrum(problem, n_steps=500_000, renorm_every=1, t_transient=1_000.0)


def run_tent():
    step = systems.tent_map()
    problem = ODEProblem(step_fn=step, state0=jnp.array([0.4]), dt=1.0)
    return lyapunov_spectrum(problem, n_steps=500_000, renorm_every=1, t_transient=1_000.0)


def run_henon():
    step = systems.henon_map(a=1.4, b=0.3)
    problem = ODEProblem(step_fn=step, state0=jnp.array([0.1, 0.1]), dt=1.0)
    return lyapunov_spectrum(problem, n_steps=200_000, renorm_every=1, t_transient=1_000.0)


if __name__ == "__main__":
    for name, fn in [("logistic_map_tier0.2", run_logistic),
                      ("tent_map_tier0.2", run_tent),
                      ("henon_map_tier0.3", run_henon)]:
        first_s, warm_s, result = time_and_run(fn)
        emit("lyapax", name, result.exponents, first_s, warm_s)
