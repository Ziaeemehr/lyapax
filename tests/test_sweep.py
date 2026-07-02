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
import jax.numpy as jnp
import numpy as np

from lyapax.core import lyapunov_spectrum
from lyapax.coupling import kuramoto_coupling, linear_coupling
from lyapax.network import make_network_step_fn, make_parametrized_network_step_fn
from lyapax.simulator import ModelSpec, StateVar, Parameter, build_jax_dfun
from lyapax.sweep import sweep_lyapunov_spectrum


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
