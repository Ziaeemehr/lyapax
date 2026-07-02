"""M5 validation tests: Tier 4.3-style checks from notes/validation_systems.md
for genuine per-edge (heterogeneous) delayed networks.

Unlike M4, this needed no new engine code: lyapax.dde.lyapunov_spectrum_dde
already differentiates through whatever carry step_fn produces regardless
of delay structure, so a per-edge delay_steps matrix (via the untouched
legacy lyapax.simulator.make_step_fn(coupling_fn=None, delay_steps=...)
path) just works -- see notes/milestones.md (M5) for how this was verified
before writing any new code. These tests are the validation that
verification implied, not proof of new machinery.
"""
import jax.numpy as jnp
import numpy as np
from scipy.special import lambertw

from lyapax.core import lyapunov_spectrum
from lyapax.coupling import linear_coupling
from lyapax.dde import constant_history_buf0, lyapunov_spectrum_dde
from lyapax.network import make_network_step_fn
from lyapax.simulator import (
    Connectivity,
    ModelSpec,
    Parameter,
    StateVar,
    build_jax_dfun,
    make_step_fn,
)


def _linear_node_model(gamma: float) -> ModelSpec:
    return ModelSpec(
        name="linear_node",
        state_variables=(StateVar("x", default_init=0.0),),
        parameters=(Parameter("gamma", gamma),),
        cvar=("x",),
        dfun_str={"x": "gamma * x + c"},
    )


def test_per_edge_delay_near_zero_recovers_m3_eigenvalues():
    """delay -> 0 limit recovers M3's exact-eigenvalue result (Tier 3.1),
    via a genuine per-edge delay_steps matrix at its smallest resolvable
    value (tau_steps=1 everywhere) rather than a literal delay_steps=0
    (which reduces to the has_delays=False path, not a delay-limit check)."""
    weights = np.array([
        [0., 1., 0., 1.],
        [1., 0., 1., 0.],
        [0., 1., 0., 1.],
        [1., 0., 1., 0.],
    ])
    gamma, G = -2.0, 0.5
    A = gamma * np.eye(4) + G * weights
    expected = np.sort(np.linalg.eigvalsh(A))[::-1]

    model = _linear_node_model(gamma)
    dfun = build_jax_dfun(model)
    params = {"gamma": gamma, "G": G}
    dt = 1e-3
    state0 = jnp.array([[0.3, -0.1, 0.2, -0.4]])

    # M3, zero-delay.
    step_ode = make_network_step_fn(
        dfun, jnp.array(weights), model.cvar_indices, params, dt,
        coupling_fn=linear_coupling(a=1.0, b=0.0),
    )
    result_ode = lyapunov_spectrum(
        step_ode, state0=state0[0], dt=dt, n_steps=20_000, renorm_every=10, t_transient=5.0)

    # M5, minimal per-edge delay (every edge delayed by 1 step, uniform
    # value but still routed through the general per-edge delay_steps
    # matrix machinery, not the uniform-tau_steps/coupling_fn path).
    tau_steps = 1
    horizon = tau_steps + 1
    delay_steps = jnp.full((4, 4), tau_steps, dtype=jnp.int32)
    step_dde = make_step_fn(
        dfun=dfun, weights=jnp.array(weights), has_delays=True, horizon=horizon,
        n_nodes=4, cvar_indices=model.cvar_indices, dt=dt, delay_steps=delay_steps,
        G_default=G, coup_a=1.0, coup_b=0.0, use_heun=True,
    )
    buf0 = constant_history_buf0(state0, horizon)
    result_dde = lyapunov_spectrum_dde(
        step_dde, state0=state0, buf0=buf0, params=params, dt=dt,
        n_steps=20_000, k=4, renorm_every=10, t_transient=5.0,
    )

    np.testing.assert_allclose(np.array(result_ode.exponents), expected, atol=3e-3)
    np.testing.assert_allclose(np.array(result_dde.exponents), expected, atol=3e-3)
    np.testing.assert_allclose(
        np.array(result_dde.exponents), np.array(result_ode.exponents), atol=1e-3)


def test_two_node_symmetric_delayed_network_matches_lambert_w():
    """Tier 4.3: a symmetric 2-node delayed linear network
    (x1'=gamma*x1+G*x2(t-tau), x2'=gamma*x2+G*x1(t-tau)) has a tractable
    characteristic equation, generalizing Tier 4.2's scalar case. Trying
    x1=x2=e^{lambda t} (the symmetric mode) gives
    lambda = gamma + G*e^{-lambda*tau}; substituting mu=lambda-gamma and
    solving via Lambert W gives lambda_sym = gamma + W(G*tau*e^{-gamma*tau})/tau.
    The antisymmetric mode (x1=-x2) gives the same form with -G instead of
    G: lambda_antisym = gamma + W(-G*tau*e^{-gamma*tau})/tau."""
    gamma, G, tau, dt = -1.0, 0.5, 0.3, 1e-3
    lambda_sym = gamma + float((lambertw(G * tau * np.exp(-gamma * tau), k=0) / tau).real)
    lambda_antisym = gamma + float((lambertw(-G * tau * np.exp(-gamma * tau), k=0) / tau).real)
    expected = np.sort([lambda_sym, lambda_antisym])[::-1]

    model = _linear_node_model(gamma)
    dfun = build_jax_dfun(model)
    weights = jnp.array([[0., 1.], [1., 0.]])
    conn = Connectivity(
        weights=np.array([[0., 1.], [1., 0.]]),
        tract_lengths=np.array([[0., tau], [tau, 0.]]), speed=1.0)
    delay_steps = jnp.array(conn.delay_steps(dt))
    horizon = conn.horizon(dt)

    step_fn = make_step_fn(
        dfun=dfun, weights=weights, has_delays=True, horizon=horizon, n_nodes=2,
        cvar_indices=model.cvar_indices, dt=dt, delay_steps=delay_steps,
        G_default=G, coup_a=1.0, coup_b=0.0, use_heun=True,
    )
    state0 = jnp.array([[0.3, -0.2]])
    buf0 = constant_history_buf0(state0, horizon)
    params = {"gamma": gamma, "G": G}

    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params=params, dt=dt,
        n_steps=20_000, k=2, renorm_every=10, t_transient=5.0,
    )

    np.testing.assert_allclose(np.array(result.exponents), expected, atol=5e-3)
