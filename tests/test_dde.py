"""M4 validation tests: Tier 4 from notes/validation_systems.md, plus
engine-mechanism checks for the DDE Lyapunov engine (src/lyapax/dde.py),
which reuses the vendored ring-buffer simulator (src/lyapax/simulator/step.py)
rather than a second history mechanism -- see notes/milestones.md (M4) for
the design.

Tier 4.2 (linear scalar DDE) is checked first, against the transcendental
characteristic equation's dominant root (Lambert W) -- this isolates a DDE
tangent-propagation bug from Mackey-Glass's nonlinear chaos. The two scalar
benchmarks (systems.mackey_glass/systems.linear_scalar_dde) go through
dde.make_scalar_delayed_step_fn -- the lightweight, ModelSpec-free front
door for non-networked DDEs. test_delayed_network_benchmark_scale exercises
the other (general, ModelSpec/coupling_fn-based) front door directly, for a
real delayed network -- both build a step for the same underlying vendored
ring-buffer simulator. None of these compare against anything in
lyapunov-master/.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import lambertw

from lyapax import systems
from lyapax.core import kaplan_yorke_dimension
from lyapax.coupling import kuramoto_coupling
from lyapax.dde import (
    constant_history_buf0,
    dde_problem,
    lyapunov_spectrum_dde,
    make_scalar_delayed_step_fn,
    resolve_tau_steps,
    scalar_delayed_history0,
)
from lyapax.simulator import ModelSpec, Parameter, StateVar, build_jax_dfun, make_step_fn

# ---------------------------------------------------------------------------
# Tier 4.2 -- linear scalar DDE, dominant exponent vs Lambert W root
# ---------------------------------------------------------------------------

def test_linear_scalar_dde_matches_lambert_w_root():
    # Small a*tau (< 1/e): non-oscillatory decay, real dominant root.
    a, tau, dt = 0.5, 0.3, 1e-2
    tau_steps = resolve_tau_steps(tau, dt)
    rhs = systems.linear_scalar_dde(a=a)
    step_fn = make_scalar_delayed_step_fn(rhs, m=1, tau_steps=tau_steps, dt=dt)
    state0, buf0 = scalar_delayed_history0(jnp.array([1.0]), tau_steps)

    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
        n_steps=20_000, k=1, renorm_every=5, t_transient=10.0,
    )

    expected = float((lambertw(-a * tau, k=0) / tau).real)
    assert abs(float(result.exponents[0]) - expected) < 0.01


def test_dde_problem_call_rejects_conflicting_direct_dt():
    dt = 1e-2
    problem = dde_problem(
        systems.linear_scalar_dde(a=0.5),
        state0=jnp.array([1.0]),
        tau=0.3,
        dt=dt,
    )

    with pytest.raises(ValueError, match="different values"):
        lyapunov_spectrum_dde(problem, n_steps=10, k=1, dt=2 * dt)


def test_linear_scalar_dde_dt_convergence():
    """Same physical tau at two different dt: the LE estimate should be
    dt-stable (cross-cutting test hygiene in notes/validation_systems.md),
    catching integer-step delay rounding bugs (risk #4 in
    notes/milestones.md) rather than genuine dt-dependence."""
    a, tau = 0.5, 0.3
    rhs = systems.linear_scalar_dde(a=a)
    expected = float((lambertw(-a * tau, k=0) / tau).real)

    estimates = []
    for dt in (2e-2, 1e-2):
        tau_steps = resolve_tau_steps(tau, dt)
        step_fn = make_scalar_delayed_step_fn(rhs, m=1, tau_steps=tau_steps, dt=dt)
        state0, buf0 = scalar_delayed_history0(jnp.array([1.0]), tau_steps)
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
    rhs = systems.linear_scalar_dde(a=a)
    step_fn = make_scalar_delayed_step_fn(rhs, m=1, tau_steps=tau_steps, dt=dt)
    state0, buf0 = scalar_delayed_history0(jnp.array([1.0]), tau_steps)

    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
        n_steps=20_000, k=1, renorm_every=10, t_transient=2.0,
    )

    assert abs(float(result.exponents[0]) - (-a)) < 0.02


# ---------------------------------------------------------------------------
# resume -- continuing a DDE run from a previous call's checkpoint
# ---------------------------------------------------------------------------

def test_dde_resume_matches_single_uninterrupted_run():
    a, tau, dt = 0.5, 0.3, 1e-2
    problem = dde_problem(systems.linear_scalar_dde(a=a), state0=jnp.array([1.0]), tau=tau, dt=dt)

    n1, n2, renorm_every = 15_000, 10_000, 5

    whole = lyapunov_spectrum_dde(
        problem, n_steps=n1 + n2, k=1, renorm_every=renorm_every, t_transient=10.0,
    )

    first = lyapunov_spectrum_dde(
        problem, n_steps=n1, k=1, renorm_every=renorm_every, t_transient=10.0,
    )
    second = lyapunov_spectrum_dde(
        problem, n_steps=n2, k=1, renorm_every=renorm_every, resume=first.checkpoint,
    )

    n1_renorm = n1 // renorm_every
    np.testing.assert_allclose(
        np.array(second.times), np.array(whole.times[n1_renorm:]), rtol=1e-12)
    np.testing.assert_allclose(
        np.array(second.exponents), np.array(whole.exponents), atol=1e-8)
    np.testing.assert_allclose(
        np.array(second.history[-1]), np.array(whole.history[-1]), atol=1e-8)


def test_dde_resume_and_t_transient_mutually_exclusive():
    a, tau, dt = 0.5, 0.3, 1e-2
    problem = dde_problem(systems.linear_scalar_dde(a=a), state0=jnp.array([1.0]), tau=tau, dt=dt)
    result = lyapunov_spectrum_dde(problem, n_steps=1_000, k=1, renorm_every=5, t_transient=10.0)

    with pytest.raises(ValueError, match="mutually exclusive"):
        lyapunov_spectrum_dde(
            problem, n_steps=1_000, k=1, renorm_every=5, t_transient=1.0,
            resume=result.checkpoint,
        )


def test_dde_resume_rejects_mismatched_k():
    a, tau, dt = 0.5, 0.3, 1e-2
    problem = dde_problem(systems.linear_scalar_dde(a=a), state0=jnp.array([1.0]), tau=tau, dt=dt)
    result = lyapunov_spectrum_dde(problem, n_steps=1_000, k=1, renorm_every=5, t_transient=10.0)

    with pytest.raises(ValueError, match="tracked dimension"):
        lyapunov_spectrum_dde(
            problem, n_steps=1_000, k=2, renorm_every=5, resume=result.checkpoint,
        )


def test_dde_resume_rejects_dimension_mismatch():
    a, tau, dt = 0.5, 0.3, 1e-2
    problem1 = dde_problem(systems.linear_scalar_dde(a=a), state0=jnp.array([1.0]), tau=tau, dt=dt)
    result1 = lyapunov_spectrum_dde(problem1, n_steps=1_000, k=1, renorm_every=5, t_transient=10.0)

    problem2 = dde_problem(
        systems.linear_scalar_dde(a=a), state0=jnp.array([1.0]), tau=2 * tau, dt=dt)
    with pytest.raises(ValueError, match="shapes"):
        lyapunov_spectrum_dde(
            problem2, n_steps=1_000, renorm_every=5, resume=result1.checkpoint,
        )


def test_dde_resume_rejects_dt_mismatch():
    # tau doubled alongside dt so tau_steps (== round(tau / dt)) -- and
    # therefore the ring-buffer horizon/shape -- stays identical; otherwise
    # the shape check above would fire first and this wouldn't isolate the
    # dt check.
    a, tau, dt = 0.5, 0.3, 1e-2
    problem1 = dde_problem(systems.linear_scalar_dde(a=a), state0=jnp.array([1.0]), tau=tau, dt=dt)
    result1 = lyapunov_spectrum_dde(problem1, n_steps=1_000, k=1, renorm_every=5, t_transient=10.0)

    problem2 = dde_problem(
        systems.linear_scalar_dde(a=a), state0=jnp.array([1.0]), tau=2 * tau, dt=2 * dt)
    with pytest.raises(ValueError, match="resume.dt"):
        lyapunov_spectrum_dde(
            problem2, n_steps=1_000, k=1, renorm_every=5, resume=result1.checkpoint,
        )


def test_transient_floor_prevents_bias_from_short_user_transient():
    """lyapunov_spectrum_dde internally floors t_transient to at least one
    full ring cycle (horizon*dt) -- see the module docstring's discussion
    of why a shorter transient silently under-converges the tangent basis
    for a DDE. Passing t_transient=0.0 explicitly should still land within
    the same tolerance as a well-transiented run, not degrade silently."""
    a, tau, dt = 0.5, 0.3, 1e-2
    tau_steps = resolve_tau_steps(tau, dt)
    rhs = systems.linear_scalar_dde(a=a)
    step_fn = make_scalar_delayed_step_fn(rhs, m=1, tau_steps=tau_steps, dt=dt)
    state0, buf0 = scalar_delayed_history0(jnp.array([1.0]), tau_steps)
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
    rhs = systems.linear_scalar_dde(a=a)
    step_fn = make_scalar_delayed_step_fn(rhs, m=1, tau_steps=tau_steps, dt=dt)

    state0, buf0 = scalar_delayed_history0(jnp.array([0.8]), tau_steps)
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

def test_mackey_glass_qualitative_chaos():
    beta, gamma, n, tau = 0.2, 0.1, 10.0, 17.0
    dt = 1.0
    tau_steps = resolve_tau_steps(tau, dt)
    rhs = systems.mackey_glass(beta=beta, gamma=gamma, n=n)
    step_fn = make_scalar_delayed_step_fn(rhs, m=1, tau_steps=tau_steps, dt=dt)
    state0, buf0 = scalar_delayed_history0(jnp.array([1.2]), tau_steps)

    # k=8, not the full d_total=18 spectrum: the most contracting directions
    # underflow log|diag(R)| to -inf well before they're needed -- only the
    # leading few exponents are needed for lambda1/lambda2/KY dimension
    # anyway (same numerical-sensitivity note jitcdde's docs make about deep
    # negative exponents needing more frequent rescaling).
    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
        n_steps=30_000, k=8, renorm_every=10, t_transient=3_000.0,
    )

    exponents = np.array(result.exponents)
    # Reported order of magnitude: 1e-2-1e-3 (see notes/validation_systems.md
    # Tier 4.1) -- treat as qualitative/order-of-magnitude, not digit-matching.
    assert 0.0005 < exponents[0] < 0.05
    assert abs(exponents[1]) < 0.01
    assert np.all(exponents[2:] < 0.0)

    ky = kaplan_yorke_dimension(exponents)
    assert 1.5 < ky < 3.5


# ---------------------------------------------------------------------------
# Scale benchmark: the actual point of the M4 redesign -- a delayed network
# with d_total well beyond what a dense-jacfwd engine could handle per step.
# Uses the general (ModelSpec/coupling_fn) front door directly, not the
# scalar convenience layer above -- this is a real network, not a
# non-networked system, so the full machinery is the right tool here.
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
        coupling_fn=kuramoto_coupling(alpha=0.0), tau_steps=tau_steps, integrator="heun",
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
