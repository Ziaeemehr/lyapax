"""M4 validation tests: Tier 4 from notes/validation_systems.md, plus
engine-mechanism checks for the DDE Lyapunov engine (src/lyapax/dde.py),
which reuses the vendored ring-buffer simulator (src/lyapax/vendored/step.py)
rather than a second history mechanism -- see notes/milestones.md (M4) for
the design.

Tier 4.2 (linear scalar DDE) is checked first, against the transcendental
characteristic equation's dominant root (Lambert W) -- this isolates a DDE
tangent-propagation bug from Mackey-Glass's nonlinear chaos. Benchmarks are
expressed as 1-node self-loop ModelSpec/coupling networks (a node "coupled
to its own delayed history"), mirroring how test_network.py builds its
ModelSpec helpers locally. None of these compare against anything in
lyapunov-master/.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import lambertw

from lyapax.dde import lyapunov_spectrum_dde, resolve_tau_steps, constant_history_buf0
from lyapax.coupling import linear_coupling, kuramoto_coupling
from lyapax.vendored import ModelSpec, StateVar, Parameter, build_jax_dfun, make_step_fn


def _linear_scalar_dde_model() -> ModelSpec:
    """x'(t) = -a*x(t-tau) via a 1-node self-loop: c = -a*x(t-tau)."""
    return ModelSpec(
        name="linear_scalar_dde",
        state_variables=(StateVar("x", default_init=1.0),),
        parameters=(),
        cvar=("x",),
        dfun_str={"x": "c"},
    )


def _mackey_glass_model(beta: float, gamma: float, n: float) -> ModelSpec:
    return ModelSpec(
        name="mackey_glass",
        state_variables=(StateVar("x", default_init=1.2),),
        parameters=(Parameter("beta", beta), Parameter("gamma", gamma), Parameter("n", n)),
        cvar=("x",),
        dfun_str={"x": "beta*c/(1+c**n) - gamma*x"},
    )


def _make_linear_scalar_dde_step(a: float, tau_steps: int, dt: float):
    model = _linear_scalar_dde_model()
    dfun = build_jax_dfun(model)
    weights = jnp.array([[1.0]])
    step_fn = make_step_fn(
        dfun=dfun, weights=weights, has_delays=True, horizon=tau_steps + 1,
        n_nodes=1, cvar_indices=model.cvar_indices, dt=dt,
        coupling_fn=linear_coupling(a=-a, b=0.0, G_default=1.0),
        tau_steps=tau_steps, use_heun=True,
    )
    return step_fn


# ---------------------------------------------------------------------------
# Tier 4.2 -- linear scalar DDE, dominant exponent vs Lambert W root
# ---------------------------------------------------------------------------

def test_linear_scalar_dde_matches_lambert_w_root():
    # Small a*tau (< 1/e): non-oscillatory decay, real dominant root.
    a, tau, dt = 0.5, 0.3, 1e-2
    tau_steps = resolve_tau_steps(tau, dt)
    step_fn = _make_linear_scalar_dde_step(a, tau_steps, dt)

    state0 = jnp.array([[1.0]])
    buf0 = constant_history_buf0(state0, tau_steps + 1)

    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
        n_steps=20_000, k=1, renorm_every=5, t_transient=10.0,
    )

    expected = float((lambertw(-a * tau, k=0) / tau).real)
    assert abs(float(result.exponents[0]) - expected) < 0.01


def test_linear_scalar_dde_dt_convergence():
    """Same physical tau at two different dt: the LE estimate should be
    dt-stable (cross-cutting test hygiene in notes/validation_systems.md),
    catching integer-step delay rounding bugs (risk #4 in
    notes/milestones.md) rather than genuine dt-dependence."""
    a, tau = 0.5, 0.3
    expected = float((lambertw(-a * tau, k=0) / tau).real)

    estimates = []
    for dt in (2e-2, 1e-2):
        tau_steps = resolve_tau_steps(tau, dt)
        step_fn = _make_linear_scalar_dde_step(a, tau_steps, dt)
        state0 = jnp.array([[1.0]])
        buf0 = constant_history_buf0(state0, tau_steps + 1)
        result = lyapunov_spectrum_dde(
            step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
            n_steps=20_000, k=1, renorm_every=5, t_transient=10.0,
        )
        estimates.append(float(result.exponents[0]))

    assert abs(estimates[0] - estimates[1]) < 0.01
    for est in estimates:
        assert abs(est - expected) < 0.015


def test_linear_scalar_dde_small_delay_recovers_ode_decay():
    """tau_steps as small as the scheme allows should recover the plain
    ẋ=-a*x decay rate -a -- a cheap regression check tying the DDE engine
    back to the non-delayed case, independent of the Lambert W reference."""
    a, dt, tau_steps = 1.0, 1e-3, 2
    step_fn = _make_linear_scalar_dde_step(a, tau_steps, dt)
    state0 = jnp.array([[1.0]])
    buf0 = constant_history_buf0(state0, tau_steps + 1)

    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
        n_steps=20_000, k=1, renorm_every=10, t_transient=2.0,
    )

    assert abs(float(result.exponents[0]) - (-a)) < 0.02


def test_transient_floor_prevents_bias_from_short_user_transient():
    """lyapunov_spectrum_dde internally floors t_transient to at least one
    full ring cycle (horizon*dt) -- see the module docstring's discussion
    of why a shorter transient silently under-converges the tangent basis
    for a DDE. Passing t_transient=0.0 explicitly should still land within
    the same tolerance as a well-transiented run, not degrade silently."""
    a, tau, dt = 0.5, 0.3, 1e-2
    tau_steps = resolve_tau_steps(tau, dt)
    step_fn = _make_linear_scalar_dde_step(a, tau_steps, dt)
    state0 = jnp.array([[1.0]])
    buf0 = constant_history_buf0(state0, tau_steps + 1)
    expected = float((lambertw(-a * tau, k=0) / tau).real)

    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
        n_steps=20_000, k=1, renorm_every=5, t_transient=0.0,
    )

    assert abs(float(result.exponents[0]) - expected) < 0.01


# ---------------------------------------------------------------------------
# Engine-mechanism check: jvp/vmap tangent propagation vs dense jacfwd
# ---------------------------------------------------------------------------

def test_tangent_propagation_matches_dense_jacfwd():
    """Directly validates the jvp/vmap machinery in lyapunov_spectrum_dde's
    _advance, independent of any downstream QR/statistical convergence:
    one raw step's tangent action (dY = J @ Y) computed via the vmapped-jvp
    approach must match a dense jax.jacfwd of the same step over the
    flattened (state, buf) pair, to machine precision. Not a public API --
    an ad hoc reference computed inline, so we don't reintroduce a second
    public DDE mechanism just to test the first one."""
    a, dt, tau_steps = 0.7, 0.05, 4
    horizon = tau_steps + 1
    step_fn = _make_linear_scalar_dde_step(a, tau_steps, dt)

    state0 = jnp.array([[0.8]])
    buf0 = constant_history_buf0(state0, horizon)
    params = {}
    d_state, d_buf = state0.size, buf0.size
    d_total = d_state + d_buf

    def flat_step(x_flat):
        state = x_flat[:d_state].reshape(state0.shape)
        buf = x_flat[d_state:].reshape(buf0.shape)
        (new_state, new_buf, _t, _p), _ = step_fn((state, buf, jnp.int32(0), params), None)
        return jnp.concatenate([new_state.reshape(-1), new_buf.reshape(-1)])

    x0_flat = jnp.concatenate([state0.reshape(-1), buf0.reshape(-1)])
    dense_jac = jax.jacfwd(flat_step)(x0_flat)

    key = jax.random.PRNGKey(0)
    Y_flat0, _ = jnp.linalg.qr(jax.random.normal(key, (d_total, d_total), dtype=jnp.float64))
    Y_state0 = Y_flat0[:d_state].reshape(state0.shape + (d_total,))
    Y_buf0 = Y_flat0[d_state:].reshape(buf0.shape + (d_total,))

    def f(state, buf):
        (new_state, new_buf, _t2, _p2), _ = step_fn((state, buf, jnp.int32(0), params), None)
        return new_state, new_buf

    def _single_column(ys, yb):
        return jax.jvp(f, (state0, buf0), (ys, yb))

    (_new_state_rep, _new_buf_rep), (dY_state, dY_buf) = jax.vmap(
        _single_column, in_axes=(-1, -1), out_axes=((0, 0), (-1, -1))
    )(Y_state0, Y_buf0)

    computed_action = jnp.concatenate(
        [dY_state.reshape(d_state, d_total), dY_buf.reshape(d_buf, d_total)], axis=0)
    expected_action = dense_jac @ Y_flat0

    np.testing.assert_allclose(np.array(computed_action), np.array(expected_action), atol=1e-10)


# ---------------------------------------------------------------------------
# Tier 4.1 -- Mackey-Glass, qualitative chaos check
# ---------------------------------------------------------------------------

def _kaplan_yorke_dimension(exponents: np.ndarray) -> float:
    """exponents must be sorted descending. Standard KY formula:
    j + sum(exponents[:j]) / |exponents[j]|, j = largest index with a
    non-negative partial sum."""
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


def test_mackey_glass_qualitative_chaos():
    beta, gamma, n, tau = 0.2, 0.1, 10.0, 17.0
    dt = 1.0
    tau_steps = resolve_tau_steps(tau, dt)
    horizon = tau_steps + 1
    model = _mackey_glass_model(beta, gamma, n)
    dfun = build_jax_dfun(model)
    weights = jnp.array([[1.0]])
    params = {"beta": beta, "gamma": gamma, "n": n}
    step_fn = make_step_fn(
        dfun=dfun, weights=weights, has_delays=True, horizon=horizon,
        n_nodes=1, cvar_indices=model.cvar_indices, dt=dt,
        coupling_fn=linear_coupling(a=1.0, b=0.0, G_default=1.0),
        tau_steps=tau_steps, use_heun=True,
    )
    state0 = jnp.array([[1.2]])
    buf0 = constant_history_buf0(state0, horizon)

    # k=8, not the full d_total=18 spectrum: the most contracting directions
    # underflow log|diag(R)| to -inf well before they're needed -- only the
    # leading few exponents are needed for lambda1/lambda2/KY dimension
    # anyway (same numerical-sensitivity note jitcdde's docs make about deep
    # negative exponents needing more frequent rescaling).
    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params=params, dt=dt,
        n_steps=30_000, k=8, renorm_every=10, t_transient=3_000.0,
    )

    exponents = np.array(result.exponents)
    # Reported order of magnitude: 1e-2-1e-3 (see notes/validation_systems.md
    # Tier 4.1) -- treat as qualitative/order-of-magnitude, not digit-matching.
    assert 0.0005 < exponents[0] < 0.05
    assert abs(exponents[1]) < 0.01
    assert np.all(exponents[2:] < 0.0)

    ky = _kaplan_yorke_dimension(exponents)
    assert 1.5 < ky < 3.5


# ---------------------------------------------------------------------------
# Scale benchmark: the actual point of the M4 redesign -- a delayed network
# with d_total well beyond what a dense-jacfwd engine could handle per step.
# ---------------------------------------------------------------------------

def test_delayed_network_benchmark_scale():
    n_nodes = 10
    omega = jnp.linspace(-1.0, 1.0, n_nodes)
    weights = jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)
    model = ModelSpec(
        name="kuramoto", state_variables=(StateVar("theta", 0.0),),
        parameters=(Parameter("omega", 0.0),), cvar=("theta",),
        dfun_str={"theta": "omega + c"},
    )
    dfun = build_jax_dfun(model)
    dt, tau = 1e-2, 0.3
    tau_steps = resolve_tau_steps(tau, dt)
    horizon = tau_steps + 1
    d_total = n_nodes + horizon * n_nodes
    assert d_total > 300  # representative of "beyond dense-jacfwd-practical" scale

    params = {"omega": omega, "G": 1.0}
    step_fn = make_step_fn(
        dfun=dfun, weights=weights, has_delays=True, horizon=horizon,
        n_nodes=n_nodes, cvar_indices=model.cvar_indices, dt=dt,
        coupling_fn=kuramoto_coupling(alpha=0.0), tau_steps=tau_steps, use_heun=True,
    )
    state0 = jnp.linspace(0.0, 2 * jnp.pi, n_nodes, endpoint=False).reshape(1, n_nodes)
    buf0 = constant_history_buf0(state0, horizon)

    t0 = time.perf_counter()
    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params=params, dt=dt,
        n_steps=2_000, k=5, renorm_every=10, t_transient=5.0,
    )
    elapsed = time.perf_counter() - t0

    assert result.exponents.shape == (5,)
    assert np.all(np.isfinite(np.array(result.exponents)))
    assert elapsed < 30.0  # generous CI ceiling; observed ~2s locally
