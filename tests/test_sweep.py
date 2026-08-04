"""M6 tests: jax.vmap parameter/initial-condition sweeps
(lyapax.sweep.sweep_lyapunov_spectrum).

The only thing worth testing here is that batching via jax.vmap produces
the *same* numbers as calling lyapax.core.lyapunov_spectrum once per grid
point in a Python loop (the pre-M6 pattern used by
examples/plot_05_kuramoto_sync.py) -- sweep_lyapunov_spectrum adds no new
tangent-propagation or QR math, it only batches the existing engine, so
correctness here means "matches the loop exactly," not a new analytic
reference.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lyapax.core import lyapunov_spectrum
from lyapax.coupling import kuramoto_coupling, linear_coupling
from lyapax.network import make_network_step_fn, make_parametrized_network_step_fn
from lyapax.simulator import ModelSpec, Parameter, StateVar, build_jax_dfun
from lyapax.sweep import _warn_if_trajectory_too_large, sweep_lyapunov_spectrum, sweep_trajectory


def _kuramoto_model(omega: float) -> ModelSpec:
    return ModelSpec(
        name="kuramoto", state_variables=(StateVar("theta", default_init=0.0),),
        parameters=(Parameter("omega", omega),), cvar=("theta",),
        dfun_str={"theta": "omega + c"},
    )


def _linear_node_model(gamma: float) -> ModelSpec:
    return ModelSpec(
        name="linear_node", state_variables=(StateVar("x", default_init=0.0),),
        parameters=(Parameter("gamma", gamma),), cvar=("x",),
        dfun_str={"x": "gamma * x + c"},
    )


def test_param_sweep_matches_python_loop_over_lyapunov_spectrum():
    n_nodes = 4
    omega = jnp.linspace(-1.0, 1.0, n_nodes)
    weights = jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)
    model = _kuramoto_model(omega=0.0)
    dfun = build_jax_dfun(model)
    dt = 1e-2
    state0 = jnp.linspace(0.0, 2 * jnp.pi, n_nodes, endpoint=False)

    G_values = jnp.array([0.0, 1.0, 3.0])
    n_sweep = G_values.shape[0]
    params_batch = {
        "omega": jnp.broadcast_to(omega, (n_sweep, n_nodes)),
        "G": G_values,
    }

    step_p = make_parametrized_network_step_fn(
        dfun, weights, model.cvar_indices, dt, kuramoto_coupling(alpha=0.0))
    swept = sweep_lyapunov_spectrum(
        step_p, state0, params_batch, dt, n_steps=2_000, k=2, renorm_every=10, t_transient=5.0)

    assert swept.exponents.shape == (n_sweep, 2)

    reference = []
    for G in G_values:
        params = {"omega": omega, "G": float(G)}
        step = make_network_step_fn(
            dfun, weights, model.cvar_indices, params, dt,
            coupling_fn=kuramoto_coupling(alpha=0.0))
        result = lyapunov_spectrum(
            step, state0=state0, dt=dt, n_steps=2_000, k=2, renorm_every=10, t_transient=5.0)
        reference.append(np.array(result.exponents))
    reference = np.array(reference)

    np.testing.assert_allclose(np.array(swept.exponents), reference, atol=1e-8)


def test_param_sweep_zero_coupling_still_exactly_zero():
    """G=0 is exact for every element of the batch, not just a single
    Python-loop call -- confirms vmap doesn't perturb the exact-identity
    Jacobian case from test_network.py::test_kuramoto_zero_coupling_gives_exactly_zero_spectrum."""
    n_nodes = 3
    weights = jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)
    model = _kuramoto_model(omega=1.0)
    dfun = build_jax_dfun(model)
    dt = 1e-2
    state0 = jnp.array([0.0, 1.0, 2.0])

    n_sweep = 3
    params_batch = {
        "omega": jnp.ones((n_sweep, n_nodes)),
        "G": jnp.zeros(n_sweep),
    }
    step_p = make_parametrized_network_step_fn(
        dfun, weights, model.cvar_indices, dt, kuramoto_coupling(alpha=0.0))
    swept = sweep_lyapunov_spectrum(
        step_p, state0, params_batch, dt, n_steps=2_000, renorm_every=10)

    np.testing.assert_allclose(np.array(swept.exponents), np.zeros((n_sweep, n_nodes)), atol=1e-8)


def test_initial_condition_sweep_matches_python_loop():
    """state0_batch: sweeping initial conditions instead of (or alongside)
    params -- same batching mechanism, exercised on the linear-network
    system from test_network.py so the individual answers are already a
    known exact-eigenvalue case."""
    weights = np.array([[0., 1.], [1., 0.]])
    gamma, G = -1.5, 0.7
    A = gamma * np.eye(2) + G * weights
    expected = np.sort(np.linalg.eigvalsh(A))[::-1]

    model = _linear_node_model(gamma)
    dfun = build_jax_dfun(model)
    dt = 1e-3
    step_p = make_parametrized_network_step_fn(
        dfun, jnp.array(weights), model.cvar_indices, dt, linear_coupling(a=1.0, b=0.0))

    state0_batch = jnp.array([[0.2, -0.3], [0.5, 0.1], [-0.4, 0.4]])
    n_sweep = state0_batch.shape[0]
    params_batch = {"gamma": jnp.full((n_sweep,), gamma), "G": jnp.full((n_sweep,), G)}

    swept = sweep_lyapunov_spectrum(
        step_p, state0=state0_batch[0], params_batch=params_batch, dt=dt, n_steps=20_000,
        renorm_every=10, t_transient=5.0, state0_batch=state0_batch)

    assert swept.exponents.shape == (n_sweep, 2)
    for row in np.array(swept.exponents):
        np.testing.assert_allclose(row, expected, atol=3e-3)


# ---------------------------------------------------------------------------
# save_states / decimate / batch_size (memory-bounded sweeps + sweep_trajectory)
# ---------------------------------------------------------------------------

def _linear_sweep_fixture(n_sweep=4):
    """Small, non-chaotic linear network -- deterministic, no transient
    needed for exact-value comparisons, fast enough for a handful of
    ``save_states``/chunking variants."""
    weights = jnp.array([[0., 1.], [1., 0.]])
    gamma = -1.5
    model = _linear_node_model(gamma)
    dfun = build_jax_dfun(model)
    dt = 1e-3
    step_p = make_parametrized_network_step_fn(
        dfun, weights, model.cvar_indices, dt, linear_coupling(a=1.0, b=0.0))
    state0 = jnp.array([0.2, -0.3])
    G_values = jnp.linspace(0.2, 1.0, n_sweep)
    params_batch = {"gamma": jnp.full((n_sweep,), gamma), "G": G_values}
    sweep_kwargs = dict(dt=dt, n_steps=500, renorm_every=10, k=2, t_transient=2.0)
    return step_p, state0, params_batch, sweep_kwargs


def test_sweep_save_states_matches_single_call_per_row():
    step_p, state0, params_batch, sweep_kwargs = _linear_sweep_fixture(n_sweep=4)
    decimate = 5
    swept = sweep_lyapunov_spectrum(
        step_p, state0, params_batch, save_states=True, decimate=decimate,
        **sweep_kwargs)

    n_saves = sweep_kwargs["n_steps"] // decimate
    assert swept.trajectory.shape == (4, n_saves, 2)
    assert swept.save_times.shape == (4, n_saves)

    for i in range(4):
        params_i = {k: v[i] for k, v in params_batch.items()}
        single = lyapunov_spectrum(
            lambda state, p=params_i: step_p(state, p), state0,
            save_states=True, decimate=decimate, **sweep_kwargs)
        np.testing.assert_array_equal(
            np.array(swept.trajectory[i]), np.array(single.trajectory))
        np.testing.assert_array_equal(
            np.array(swept.save_times[i]), np.array(single.save_times))


def test_sweep_batch_size_chunking_matches_unchunked():
    step_p, state0, params_batch, sweep_kwargs = _linear_sweep_fixture(n_sweep=5)
    decimate = 5

    unchunked = sweep_lyapunov_spectrum(
        step_p, state0, params_batch, save_states=True, decimate=decimate,
        **sweep_kwargs)
    chunked = sweep_lyapunov_spectrum(
        step_p, state0, params_batch, save_states=True, decimate=decimate,
        batch_size=2, **sweep_kwargs)  # 5 % 2 != 0 -- exercises a ragged chunk

    np.testing.assert_array_equal(
        np.array(chunked.exponents), np.array(unchunked.exponents))
    np.testing.assert_array_equal(
        np.array(chunked.history), np.array(unchunked.history))
    np.testing.assert_array_equal(
        np.array(chunked.times), np.array(unchunked.times))
    np.testing.assert_array_equal(
        np.array(chunked.trajectory), np.array(unchunked.trajectory))
    np.testing.assert_array_equal(
        np.array(chunked.save_times), np.array(unchunked.save_times))


def test_sweep_trajectory_to_host_returns_numpy_other_fields_stay_device():
    step_p, state0, params_batch, sweep_kwargs = _linear_sweep_fixture(n_sweep=3)
    on_device = sweep_lyapunov_spectrum(
        step_p, state0, params_batch, save_states=True, decimate=5,
        batch_size=2, **sweep_kwargs)
    on_host = sweep_lyapunov_spectrum(
        step_p, state0, params_batch, save_states=True, decimate=5,
        batch_size=2, trajectory_to_host=True, **sweep_kwargs)

    assert isinstance(on_host.trajectory, np.ndarray)
    assert not isinstance(on_host.trajectory, jax.Array)
    assert isinstance(on_host.exponents, jax.Array)
    np.testing.assert_array_equal(on_host.trajectory, np.array(on_device.trajectory))


def test_sweep_trajectory_matches_sweep_lyapunov_spectrum_trajectory():
    """The standalone replay path (no tangent propagation) must reproduce
    exactly the states the LE run saved -- same post-transient window,
    same decimation stride."""
    step_p, state0, params_batch, sweep_kwargs = _linear_sweep_fixture(n_sweep=4)
    decimate = 5

    swept = sweep_lyapunov_spectrum(
        step_p, state0, params_batch, save_states=True, decimate=decimate,
        **sweep_kwargs)
    replayed_t, replayed_traj = sweep_trajectory(
        step_p, state0, params_batch, dt=sweep_kwargs["dt"],
        n_steps=sweep_kwargs["n_steps"], t_transient=sweep_kwargs["t_transient"],
        renorm_every=sweep_kwargs["renorm_every"], decimate=decimate,
    )

    np.testing.assert_allclose(
        np.array(replayed_traj), np.array(swept.trajectory), atol=1e-12)
    np.testing.assert_allclose(
        np.array(replayed_t), np.array(swept.save_times), atol=1e-12)


def test_sweep_trajectory_batch_size_chunking_matches_unchunked():
    step_p, state0, params_batch, sweep_kwargs = _linear_sweep_fixture(n_sweep=5)
    decimate = 5

    unchunked_t, unchunked_traj = sweep_trajectory(
        step_p, state0, params_batch, dt=sweep_kwargs["dt"],
        n_steps=sweep_kwargs["n_steps"], t_transient=sweep_kwargs["t_transient"],
        renorm_every=sweep_kwargs["renorm_every"], decimate=decimate,
    )
    chunked_t, chunked_traj = sweep_trajectory(
        step_p, state0, params_batch, dt=sweep_kwargs["dt"],
        n_steps=sweep_kwargs["n_steps"], t_transient=sweep_kwargs["t_transient"],
        renorm_every=sweep_kwargs["renorm_every"], decimate=decimate,
        batch_size=2,
    )

    np.testing.assert_array_equal(np.array(chunked_traj), np.array(unchunked_traj))
    np.testing.assert_array_equal(np.array(chunked_t), np.array(unchunked_t))


# ---------------------------------------------------------------------------
# Size guard: warn (not raise -- the sweep still runs) when an unchunked
# save_states=True call would materialize a very large trajectory at once.
# ---------------------------------------------------------------------------

def test_warns_when_unchunked_trajectory_exceeds_threshold():
    with pytest.warns(UserWarning, match="batch_size"):
        _warn_if_trajectory_too_large(
            n_sweep=1000, n_saves=1000, d=100, itemsize=8, batch_size=None,
            threshold_bytes=100,
        )


def test_no_warning_below_threshold():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_trajectory_too_large(
            n_sweep=10, n_saves=10, d=10, itemsize=8, batch_size=None,
            threshold_bytes=1 << 30,
        )


def test_no_warning_when_batch_size_given():
    """batch_size already bounds peak memory to one chunk -- the guard is
    about the unchunked case specifically."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_trajectory_too_large(
            n_sweep=1000, n_saves=1000, d=100, itemsize=8, batch_size=8,
            threshold_bytes=100,
        )


def test_sweep_lyapunov_spectrum_warns_for_large_unchunked_save_states():
    step_p, state0, params_batch, sweep_kwargs = _linear_sweep_fixture(n_sweep=4)
    with pytest.warns(UserWarning, match="batch_size"):
        sweep_lyapunov_spectrum(
            step_p, state0, params_batch, save_states=True, decimate=5,
            _trajectory_warn_threshold_bytes=1,  # force the warning path
            **sweep_kwargs)


def test_sweep_lyapunov_spectrum_no_warning_at_default_threshold():
    step_p, state0, params_batch, sweep_kwargs = _linear_sweep_fixture(n_sweep=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sweep_lyapunov_spectrum(
            step_p, state0, params_batch, save_states=True, decimate=5,
            **sweep_kwargs)
