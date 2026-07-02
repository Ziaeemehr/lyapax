"""Tier 3.1: 4-node linear network, 4-cycle graph. Same params as
tests/test_network.py::test_linear_network_matches_eigenvalues_of_full_jacobian.
"""
import jax.numpy as jnp

from lyapax.core import lyapunov_spectrum
from lyapax.coupling import linear_coupling
from lyapax.network import make_network_step_fn
from lyapax.simulator import ModelSpec, StateVar, Parameter, build_jax_dfun

from _common import time_and_run, emit


def _linear_node_model(gamma: float) -> ModelSpec:
    return ModelSpec(
        name="linear_node",
        state_variables=(StateVar("x", default_init=0.0),),
        parameters=(Parameter("gamma", gamma),),
        cvar=("x",),
        dfun_str={"x": "gamma * x + c"},
    )


def run():
    weights = jnp.array([
        [0., 1., 0., 1.],
        [1., 0., 1., 0.],
        [0., 1., 0., 1.],
        [1., 0., 1., 0.],
    ])
    gamma, G = -2.0, 0.5
    model = _linear_node_model(gamma)
    dfun = build_jax_dfun(model)
    params = {"gamma": gamma, "G": G}
    dt = 1e-3
    step = make_network_step_fn(
        dfun, weights, model.cvar_indices, params, dt,
        coupling_fn=linear_coupling(a=1.0, b=0.0),
    )
    return lyapunov_spectrum(
        step, state0=jnp.array([0.3, -0.1, 0.2, -0.4]),
        dt=dt, n_steps=20_000, renorm_every=10, t_transient=5.0,
    )


if __name__ == "__main__":
    first_s, warm_s, result = time_and_run(run)
    emit("lyapax", "linear_network_tier3.1", result.exponents, first_s, warm_s)
