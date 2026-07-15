"""Sanity tests: environment is correct and the vendored plumbing works.

Not Lyapunov-spectrum correctness tests yet (those are in
docs/background/validation.md's tiers) -- just "does the foundation hold."
"""
from importlib import metadata

import jax
import jax.numpy as jnp
import numpy as np

import lyapax
from lyapax.__version__ import __version__
from lyapax.simulator import (
    Connectivity,
    ModelSpec,
    Parameter,
    StateVar,
    build_jax_dfun,
    make_step_fn,
)


def test_package_version_is_single_sourced():
    assert lyapax.__version__ == __version__
    assert metadata.version("lyapax") == __version__


def test_x64_enabled():
    assert jnp.array(1.0).dtype == jnp.float64


def test_running_on_cpu():
    # Confirms conftest's JAX_PLATFORMS=cpu took effect.
    assert jax.default_backend() == "cpu"


def _linear_model(gamma_default: float = -1.0) -> ModelSpec:
    return ModelSpec(
        name="linear",
        state_variables=(StateVar("x", default_init=1.0),),
        parameters=(Parameter("gamma", gamma_default),),
        cvar=("x",),
        dfun_str={"x": "gamma * x + c"},
    )


def test_uncoupled_linear_decay_matches_analytic_solution():
    """Single node, G=0 -> pure exponential decay. Heun should track
    exp(gamma*t) closely for small dt. Validates the vendored dfun codegen
    + integrator before the tangent-propagation layer (M1) touches it."""
    model = _linear_model(gamma_default=-1.0)
    dfun = build_jax_dfun(model)

    n_nodes = 1
    weights = jnp.zeros((n_nodes, n_nodes))
    delay_steps = jnp.zeros((n_nodes, n_nodes), dtype=jnp.int32)
    dt = 1e-3
    n_steps = 2000  # t_final = 2.0

    step_fn = make_step_fn(
        dfun=dfun, weights=weights, delay_steps=delay_steps,
        has_delays=False, horizon=1, n_nodes=n_nodes,
        cvar_indices=model.cvar_indices, dt=dt,
        G_default=0.0, coup_a=1.0, coup_b=0.0, integrator="heun",
    )

    state0 = jnp.array([[1.0]])  # (n_sv, n_nodes)
    buf0 = jnp.zeros((1, len(model.cvar_indices), n_nodes))
    params = {"gamma": -1.0}
    carry0 = (state0, buf0, jnp.int32(0), params)

    (final_state, *_rest), _all_states = jax.lax.scan(
        step_fn, carry0, None, length=n_steps)

    t_final = n_steps * dt
    expected = np.exp(-1.0 * t_final)
    assert np.allclose(float(final_state[0, 0]), expected, rtol=1e-3)


def test_delayed_network_smoke():
    """Two-node network with a nonzero inter-node delay: confirms the
    ring-buffer path runs and produces finite output of the right shape.
    Spectrum correctness for delayed networks is M4/M5's job, not M0's."""
    model = _linear_model(gamma_default=-1.0)
    dfun = build_jax_dfun(model)

    n_nodes = 2
    weights = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    conn = Connectivity(
        weights=np.array([[0.0, 1.0], [1.0, 0.0]]),
        tract_lengths=np.array([[0.0, 1.0], [1.0, 0.0]]),
        speed=1.0,
    )
    assert conn.has_delays

    dt = 1e-2
    delay_steps = jnp.array(conn.delay_steps(dt))
    horizon = conn.horizon(dt)

    step_fn = make_step_fn(
        dfun=dfun, weights=weights, delay_steps=delay_steps,
        has_delays=True, horizon=horizon, n_nodes=n_nodes,
        cvar_indices=model.cvar_indices, dt=dt,
        G_default=0.5, coup_a=1.0, coup_b=0.0, integrator="heun",
    )

    state0 = jnp.ones((1, n_nodes))
    buf0 = jnp.zeros((horizon, len(model.cvar_indices), n_nodes))
    params = {"gamma": -1.0}
    carry0 = (state0, buf0, jnp.int32(0), params)

    _carry_final, all_states = jax.lax.scan(step_fn, carry0, None, length=500)

    assert all_states.shape == (500, 1, n_nodes)
    assert np.all(np.isfinite(np.array(all_states)))
