"""Tests for coupled zero-delay networks (docs/background/validation.md Tier 3),
plus smoke tests for sigmoidal/Kuramoto coupling and a demonstration that a
user-written coupling function (no import from lyapax.coupling at all)
works exactly like a built-in one -- this is the concrete answer to "how
does a user provide a custom coupling."
"""
import time

import jax.numpy as jnp
import numpy as np

from lyapax.core import lyapunov_spectrum
from lyapax.coupling import kuramoto_coupling, linear_coupling, sigmoidal_coupling
from lyapax.network import make_network_step_fn
from lyapax.simulator import ModelSpec, Parameter, StateVar, build_jax_dfun


def _linear_node_model(gamma: float) -> ModelSpec:
    return ModelSpec(
        name="linear_node",
        state_variables=(StateVar("x", default_init=0.0),),
        parameters=(Parameter("gamma", gamma),),
        cvar=("x",),
        dfun_str={"x": "gamma * x + c"},
    )


def _kuramoto_model(omega: float) -> ModelSpec:
    return ModelSpec(
        name="kuramoto",
        state_variables=(StateVar("theta", default_init=0.0),),
        parameters=(Parameter("omega", omega),),
        cvar=("theta",),
        dfun_str={"theta": "omega + c"},
    )


# ---------------------------------------------------------------------------
# Tier 3.1 -- linear coupled network, exact eigenvalues of the full Jacobian
# ---------------------------------------------------------------------------

def test_linear_network_matches_eigenvalues_of_full_jacobian():
    # 4-cycle graph adjacency (symmetric -> A is symmetric -> real eigenvalues).
    # Circulant eigenvalues of the 4-cycle are {2, 0, 0, -2}.
    weights = np.array([
        [0., 1., 0., 1.],
        [1., 0., 1., 0.],
        [0., 1., 0., 1.],
        [1., 0., 1., 0.],
    ])
    gamma, G = -2.0, 0.5
    A = gamma * np.eye(4) + G * weights
    expected = np.sort(np.linalg.eigvalsh(A))[::-1]  # descending

    model = _linear_node_model(gamma)
    dfun = build_jax_dfun(model)
    params = {"gamma": gamma, "G": G}
    dt = 1e-3
    step = make_network_step_fn(
        dfun, jnp.array(weights), model.cvar_indices, params, dt,
        coupling_fn=linear_coupling(a=1.0, b=0.0),
    )

    result = lyapunov_spectrum(
        step, state0=jnp.array([0.3, -0.1, 0.2, -0.4]),
        dt=dt, n_steps=20_000, renorm_every=10, t_transient=5.0,
    )

    np.testing.assert_allclose(np.array(result.exponents), expected, atol=3e-3)


def test_user_defined_custom_coupling_function_matches_builtin():
    """No import from lyapax.coupling -- a bare function with the expected
    signature is a first-class coupling. Reproduces the exact-eigenvalue
    check above with a hand-written coupling_fn instead of linear_coupling()."""
    weights = np.array([[0., 1.], [1., 0.]])
    gamma, G = -1.5, 0.7
    A = gamma * np.eye(2) + G * weights
    expected = np.sort(np.linalg.eigvalsh(A))[::-1]

    def my_linear_coupling(cvar_state, weights, params):
        return params["G"] * jnp.einsum("ts,cs->ct", weights, cvar_state)

    model = _linear_node_model(gamma)
    dfun = build_jax_dfun(model)
    params = {"gamma": gamma, "G": G}
    dt = 1e-3
    step = make_network_step_fn(
        dfun, jnp.array(weights), model.cvar_indices, params, dt,
        coupling_fn=my_linear_coupling,
    )

    result = lyapunov_spectrum(
        step, state0=jnp.array([0.2, -0.3]),
        dt=dt, n_steps=20_000, renorm_every=10, t_transient=5.0,
    )

    np.testing.assert_allclose(np.array(result.exponents), expected, atol=3e-3)


# ---------------------------------------------------------------------------
# Sigmoidal coupling -- smoke test (no equally clean closed-form reference)
# ---------------------------------------------------------------------------

def test_sigmoidal_coupling_smoke():
    weights = jnp.array([[0., 1.], [1., 0.]])
    gamma = -1.0
    model = _linear_node_model(gamma)
    dfun = build_jax_dfun(model)
    params = {"gamma": gamma, "G": 1.0}
    dt = 1e-3
    step = make_network_step_fn(
        dfun, weights, model.cvar_indices, params, dt,
        coupling_fn=sigmoidal_coupling(a=1.0, b=0.0, midpoint=0.0, sigma=1.0),
    )

    result = lyapunov_spectrum(
        step, state0=jnp.array([0.5, -0.3]),
        dt=dt, n_steps=5_000, renorm_every=10,
    )

    assert result.exponents.shape == (2,)
    assert np.all(np.isfinite(np.array(result.exponents)))
    assert np.all(np.isfinite(np.array(result.history)))


# ---------------------------------------------------------------------------
# Kuramoto coupling -- exact zero-coupling check + smoke test when coupled
# ---------------------------------------------------------------------------

def test_kuramoto_zero_coupling_gives_exactly_zero_spectrum():
    """G=0 -> dtheta/dt = omega (constant): the step map is state + dt*omega,
    whose Jacobian is exactly the identity, so every Lyapunov exponent must
    be exactly 0 -- an exact check with no literature/tolerance needed."""
    n_nodes = 3
    weights = jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)
    model = _kuramoto_model(omega=1.0)
    dfun = build_jax_dfun(model)
    params = {"omega": 1.0, "G": 0.0}
    dt = 1e-2
    step = make_network_step_fn(
        dfun, weights, model.cvar_indices, params, dt,
        coupling_fn=kuramoto_coupling(alpha=0.0),
    )

    result = lyapunov_spectrum(
        step, state0=jnp.array([0.0, 1.0, 2.0]),
        dt=dt, n_steps=2_000, renorm_every=10,
    )

    np.testing.assert_allclose(np.array(result.exponents), np.zeros(3), atol=1e-8)


def test_kuramoto_coupled_smoke():
    n_nodes = 3
    weights = jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)
    model = _kuramoto_model(omega=1.0)
    dfun = build_jax_dfun(model)
    params = {"omega": 1.0, "G": 2.0}
    dt = 1e-2
    step = make_network_step_fn(
        dfun, weights, model.cvar_indices, params, dt,
        coupling_fn=kuramoto_coupling(alpha=0.0),
    )

    result = lyapunov_spectrum(
        step, state0=jnp.array([0.0, 1.0, 2.0]),
        dt=dt, n_steps=2_000, renorm_every=10,
    )

    assert result.exponents.shape == (3,)
    assert np.all(np.isfinite(np.array(result.exponents)))


# ---------------------------------------------------------------------------
# M6 -- scale benchmark: the actual point of switching to jvp/vmap tangent
# propagation -- a network with d well beyond what dense jacfwd computes
# cheaply per step when only a few (k) exponents are tracked. Mirrors
# tests/test_dde.py::test_delayed_network_benchmark_scale, the analogous
# scale check M4 wrote when the DDE engine got this same mechanism first.
# ---------------------------------------------------------------------------

def test_large_network_benchmark_scale():
    n_nodes = 200
    weights = jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)
    model = _kuramoto_model(omega=1.0)
    dfun = build_jax_dfun(model)
    params = {"omega": jnp.linspace(-1.0, 1.0, n_nodes), "G": 1.0}
    dt = 1e-2
    step = make_network_step_fn(
        dfun, weights, model.cvar_indices, params, dt,
        coupling_fn=kuramoto_coupling(alpha=0.0),
    )
    state0 = jnp.linspace(0.0, 2 * jnp.pi, n_nodes, endpoint=False)
    assert state0.shape[0] > 100  # representative of "beyond dense-jacfwd-cheap" scale

    t0 = time.perf_counter()
    result = lyapunov_spectrum(
        step, state0=state0, dt=dt, n_steps=2_000, k=5, renorm_every=10, t_transient=5.0,
    )
    elapsed = time.perf_counter() - t0

    assert result.exponents.shape == (5,)
    assert np.all(np.isfinite(np.array(result.exponents)))
    assert elapsed < 30.0  # generous CI ceiling
