"""Validation tests: Tiers 0-2 from docs/background/validation.md.

None of these compare against anything in lyapunov-master/ -- references
are exact analytic values, structural invariants (constant/derivable
divergence), or published literature figures, per that doc's guidance.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lyapax import systems
from lyapax.core import convergence_drift, lyapunov_spectrum, ode_problem
from lyapax.integrators import rk4_step
from lyapax.utils import simulate_trajectory

# ---------------------------------------------------------------------------
# Tier 0.1 -- linear ODE, exact eigenvalues
# ---------------------------------------------------------------------------

def test_linear_system_distinct_real_eigenvalues():
    A = jnp.diag(jnp.array([-1.0, -2.0, -5.0]))
    rhs = systems.linear_system(A)
    dt = 1e-3
    step = rk4_step(rhs, dt)

    # Even though this system is non-chaotic (all eigenvalues negative, no
    # attractor to relax onto), a transient is still needed here: the
    # *tangent vectors* start at a random orthonormal basis and need time to
    # align with the eigendirections (Oseledets subspaces) before the
    # log-growth readings reflect the true eigenvalues -- see the comment on
    # this in core.py's _advance/transient handling.
    result = lyapunov_spectrum(
        step, state0=jnp.array([0.3, -0.2, 0.5]),
        dt=dt, n_steps=20_000, renorm_every=10, t_transient=5.0,
    )

    np.testing.assert_allclose(np.array(result.exponents), [-1.0, -2.0, -5.0], atol=2e-3)


def test_problem_call_rejects_conflicting_direct_dt():
    dt = 1e-3
    problem = ode_problem(
        systems.linear_system(jnp.array([[-1.0]])),
        state0=jnp.array([0.3]),
        dt=dt,
    )

    with pytest.raises(ValueError, match="different values"):
        lyapunov_spectrum(problem, n_steps=10, dt=2 * dt)


def test_linear_system_complex_conjugate_pair():
    # eigenvalues -0.1 +/- 1j -> both Lyapunov exponents equal Re = -0.1
    A = jnp.array([[-0.1, 1.0], [-1.0, -0.1]])
    rhs = systems.linear_system(A)
    dt = 1e-3
    step = rk4_step(rhs, dt)

    result = lyapunov_spectrum(
        step, state0=jnp.array([1.0, 0.0]),
        dt=dt, n_steps=50_000, renorm_every=10,
    )

    np.testing.assert_allclose(np.array(result.exponents), [-0.1, -0.1], atol=5e-3)


# ---------------------------------------------------------------------------
# Tier 0.2 -- 1D chaotic maps, exact LE = ln(2)
# ---------------------------------------------------------------------------

def test_logistic_map_r4():
    step = systems.logistic_map(r=4.0)

    result = lyapunov_spectrum(
        step, state0=jnp.array([0.4]),
        dt=1.0, n_steps=500_000, renorm_every=1, t_transient=1_000.0,
    )

    assert abs(float(result.exponents[0]) - np.log(2.0)) < 0.02


def test_tent_map():
    step = systems.tent_map()

    result = lyapunov_spectrum(
        step, state0=jnp.array([0.4]),
        dt=1.0, n_steps=500_000, renorm_every=1, t_transient=1_000.0,
    )

    assert abs(float(result.exponents[0]) - np.log(2.0)) < 0.02


# ---------------------------------------------------------------------------
# Tier 0.3 -- Henon map, exact sum(LE) = ln|b|
# ---------------------------------------------------------------------------

def test_henon_map_sum_of_exponents():
    a, b = 1.4, 0.3
    step = systems.henon_map(a=a, b=b)

    result = lyapunov_spectrum(
        step, state0=jnp.array([0.1, 0.1]),
        dt=1.0, n_steps=200_000, renorm_every=1, t_transient=1_000.0,
    )

    total = float(jnp.sum(result.exponents))
    assert abs(total - np.log(b)) < 5e-3
    # loose check against the commonly-cited individual values
    assert result.exponents[0] > 0.3
    assert result.exponents[1] < -1.4


# ---------------------------------------------------------------------------
# Tier 1.1 / Tier 2 -- Lorenz: exact sum + published lambda1
# ---------------------------------------------------------------------------

def test_lorenz_sum_matches_constant_divergence():
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rhs = systems.lorenz(sigma, rho, beta)
    dt = 1e-2
    step = rk4_step(rhs, dt)

    result = lyapunov_spectrum(
        step, state0=jnp.array([1.0, 1.0, 1.0]),
        dt=dt, n_steps=50_000, renorm_every=10, t_transient=100.0,
    )

    expected_sum = -(sigma + 1.0 + beta)
    assert abs(float(jnp.sum(result.exponents)) - expected_sum) < 0.05


def test_lorenz_lambda1_matches_published_value():
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rhs = systems.lorenz(sigma, rho, beta)
    dt = 1e-2
    step = rk4_step(rhs, dt)

    result = lyapunov_spectrum(
        step, state0=jnp.array([1.0, 1.0, 1.0]),
        dt=dt, n_steps=50_000, renorm_every=10, t_transient=100.0,
    )

    assert abs(float(result.exponents[0]) - 0.9056) < 0.08
    assert abs(float(result.exponents[1])) < 0.03


# ---------------------------------------------------------------------------
# Tier 1.2 / Tier 2 -- Rossler: divergence identity + order-of-magnitude lambda1
# ---------------------------------------------------------------------------

def test_rossler_lambda1_order_of_magnitude():
    a, b, c = 0.2, 0.2, 5.7
    rhs = systems.rossler(a, b, c)
    dt = 1e-2
    step = rk4_step(rhs, dt)

    result = lyapunov_spectrum(
        step, state0=jnp.array([1.0, 1.0, 1.0]),
        dt=dt, n_steps=200_000, renorm_every=10, t_transient=200.0,
    )

    assert 0.02 < float(result.exponents[0]) < 0.12
    assert abs(float(result.exponents[1])) < 0.02


def test_rossler_sum_matches_divergence_identity():
    # Tier 1.2: unlike Lorenz, trace(J) = a + (x - c) is state-dependent, so
    # the structural check needs the trajectory's time-average of x:
    # lambda1 + lambda2 + lambda3 = a - c + <x>_t (docs/background/validation.md
    # Tier 1.2). Stronger than the order-of-magnitude check above since it
    # doesn't depend on trusting a literature lambda1 value at all.
    a, b, c = 0.2, 0.2, 5.7
    rhs = systems.rossler(a, b, c)
    dt = 1e-2
    step = rk4_step(rhs, dt)
    state0 = jnp.array([1.0, 1.0, 1.0])
    t_transient = 200.0
    n_steps = 200_000
    renorm_every = 10

    result = lyapunov_spectrum(
        step, state0=state0, dt=dt, n_steps=n_steps,
        renorm_every=renorm_every, t_transient=t_transient,
    )

    # Same transient-length rounding lyapunov_spectrum uses internally, so
    # <x>_t is averaged over the same post-transient window as the LE run.
    n_transient = renorm_every * max(1, round(t_transient / dt / renorm_every))
    traj = simulate_trajectory(step, state0, n_transient + n_steps)
    x_mean = float(jnp.mean(traj[n_transient:, 0]))

    expected_sum = a - c + x_mean
    assert abs(float(jnp.sum(result.exponents)) - expected_sum) < 0.05


# ---------------------------------------------------------------------------
# M2-preview -- top-k should agree with the leading columns of the full run
# ---------------------------------------------------------------------------

def test_partial_spectrum_matches_full_spectrum_leading_columns():
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rhs = systems.lorenz(sigma, rho, beta)
    dt = 1e-2
    step = rk4_step(rhs, dt)

    kwargs = dict(
        state0=jnp.array([1.0, 1.0, 1.0]),
        dt=dt, n_steps=50_000, renorm_every=10, t_transient=100.0, seed=0,
    )
    full = lyapunov_spectrum(step, k=3, **kwargs)
    partial = lyapunov_spectrum(step, k=1, **kwargs)

    # Both are independent finite-time estimates of the same lambda1 (they
    # draw differently-shaped initial tangent bases, so are not required to
    # be numerically identical) -- the tolerance matches the finite-time
    # convergence noise seen in test_lorenz_lambda1_matches_published_value.
    assert abs(float(partial.exponents[0]) - float(full.exponents[0])) < 0.01


# ---------------------------------------------------------------------------
# M6 -- engine-mechanism check: jvp/vmap tangent propagation vs dense jacfwd
# ---------------------------------------------------------------------------

def test_tangent_propagation_matches_dense_jacfwd():
    """Directly validates the jvp/vmap machinery in lyapunov_spectrum's
    _advance (M6), independent of any downstream QR/statistical
    convergence: one raw step's tangent action (dY = J @ Y) computed via
    the vmapped-jvp approach must match a dense jax.jacfwd of the same
    step, to machine precision. Not a public API -- an ad hoc reference
    computed inline, so the library doesn't need a second, dense-jacfwd
    code path just to test the jvp one. Mirrors
    tests/test_dde.py::test_tangent_propagation_matches_dense_jacfwd,
    the analogous check M4 wrote for the DDE engine's jvp/vmap mechanism."""
    A = jnp.array([[-0.5, 1.2, 0.0], [-1.2, -0.5, 0.3], [0.1, -0.2, -0.8]])
    rhs = systems.linear_system(A)
    dt = 1e-2
    step_fn = rk4_step(rhs, dt)

    state0 = jnp.array([0.3, -0.7, 0.5])
    d = state0.shape[0]
    dense_jac = jax.jacfwd(step_fn)(state0)

    key = jax.random.PRNGKey(0)
    Y0, _ = jnp.linalg.qr(jax.random.normal(key, (d, d), dtype=jnp.float64))

    def _single_column(y_col):
        return jax.jvp(step_fn, (state0,), (y_col,))

    _new_state_rep, computed_action = jax.vmap(
        _single_column, in_axes=-1, out_axes=(0, -1)
    )(Y0)
    expected_action = dense_jac @ Y0

    np.testing.assert_allclose(np.array(computed_action), np.array(expected_action), atol=1e-10)


# ---------------------------------------------------------------------------
# convergence_drift -- diagnostic summarizing tail drift in result.history
# ---------------------------------------------------------------------------

def test_convergence_drift_small_for_long_run():
    A = jnp.diag(jnp.array([-1.0, -2.0, -5.0]))
    rhs = systems.linear_system(A)
    dt = 1e-3
    step = rk4_step(rhs, dt)

    result = lyapunov_spectrum(
        step, state0=jnp.array([0.3, -0.2, 0.5]),
        dt=dt, n_steps=20_000, renorm_every=10, t_transient=5.0,
    )

    drift = convergence_drift(result, window=0.1, tol=1e-2)
    assert bool(jnp.all(drift.converged))
    assert bool(jnp.all(drift.relative < 1e-2))
    assert bool(jnp.all(jnp.isfinite(drift.absolute)))


def test_convergence_drift_without_tol_leaves_converged_none():
    A = jnp.diag(jnp.array([-1.0, -2.0, -5.0]))
    step = rk4_step(systems.linear_system(A), 1e-3)
    result = lyapunov_spectrum(
        step, state0=jnp.array([0.3, -0.2, 0.5]),
        dt=1e-3, n_steps=2_000, renorm_every=10,
    )

    drift = convergence_drift(result)
    assert drift.converged is None


def test_convergence_drift_rejects_out_of_range_window():
    A = jnp.diag(jnp.array([-1.0]))
    step = rk4_step(systems.linear_system(A), 1e-3)
    result = lyapunov_spectrum(
        step, state0=jnp.array([0.3]), dt=1e-3, n_steps=1_000, renorm_every=10,
    )

    with pytest.raises(ValueError, match="window"):
        convergence_drift(result, window=0.0)
    with pytest.raises(ValueError, match="window"):
        convergence_drift(result, window=1.5)


def test_convergence_drift_rejects_single_row_history():
    A = jnp.diag(jnp.array([-1.0]))
    step = rk4_step(systems.linear_system(A), 1e-3)
    result = lyapunov_spectrum(
        step, state0=jnp.array([0.3]), dt=1e-3, n_steps=10, renorm_every=10,
    )

    with pytest.raises(ValueError, match="at least 2"):
        convergence_drift(result)


# ---------------------------------------------------------------------------
# resume -- continuing a run from a previous call's checkpoint
# ---------------------------------------------------------------------------

def test_resume_matches_single_uninterrupted_run():
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rhs = systems.lorenz(sigma, rho, beta)
    dt = 1e-2
    problem = ode_problem(rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=dt)

    n1, n2, renorm_every = 3_000, 2_000, 10

    whole = lyapunov_spectrum(
        problem, n_steps=n1 + n2, renorm_every=renorm_every, t_transient=100.0,
    )

    first = lyapunov_spectrum(
        problem, n_steps=n1, renorm_every=renorm_every, t_transient=100.0,
    )
    second = lyapunov_spectrum(
        problem, n_steps=n2, renorm_every=renorm_every, resume=first.checkpoint,
    )

    n1_renorm = n1 // renorm_every
    np.testing.assert_allclose(
        np.array(second.times), np.array(whole.times[n1_renorm:]), rtol=1e-12)
    np.testing.assert_allclose(
        np.array(second.exponents), np.array(whole.exponents), atol=1e-8)
    np.testing.assert_allclose(
        np.array(second.history[-1]), np.array(whole.history[-1]), atol=1e-8)


def test_resume_and_t_transient_mutually_exclusive():
    step = rk4_step(systems.linear_system(jnp.diag(jnp.array([-1.0]))), 1e-3)
    result = lyapunov_spectrum(
        step, state0=jnp.array([0.3]), dt=1e-3, n_steps=100, renorm_every=10,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        lyapunov_spectrum(
            step, state0=jnp.array([0.3]), dt=1e-3, n_steps=100,
            renorm_every=10, t_transient=1.0, resume=result.checkpoint,
        )


def test_resume_rejects_mismatched_k():
    A = jnp.diag(jnp.array([-1.0, -2.0]))
    step = rk4_step(systems.linear_system(A), 1e-3)
    result = lyapunov_spectrum(
        step, state0=jnp.array([0.3, 0.1]), dt=1e-3, n_steps=100,
        renorm_every=10, k=1,
    )

    with pytest.raises(ValueError, match="tracked dimension"):
        lyapunov_spectrum(
            step, state0=jnp.array([0.3, 0.1]), dt=1e-3, n_steps=100,
            renorm_every=10, k=2, resume=result.checkpoint,
        )


def test_resume_rejects_dimension_mismatch():
    step1 = rk4_step(systems.linear_system(jnp.diag(jnp.array([-1.0]))), 1e-3)
    result1 = lyapunov_spectrum(
        step1, state0=jnp.array([0.3]), dt=1e-3, n_steps=100, renorm_every=10,
    )

    step2 = rk4_step(systems.linear_system(jnp.diag(jnp.array([-1.0, -2.0]))), 1e-3)
    with pytest.raises(ValueError, match="dimension"):
        lyapunov_spectrum(
            step2, state0=jnp.array([0.3, 0.1]), dt=1e-3, n_steps=100,
            renorm_every=10, resume=result1.checkpoint,
        )


def test_resume_rejects_dt_mismatch():
    step = rk4_step(systems.linear_system(jnp.diag(jnp.array([-1.0]))), 1e-3)
    result = lyapunov_spectrum(
        step, state0=jnp.array([0.3]), dt=1e-3, n_steps=100, renorm_every=10,
    )

    with pytest.raises(ValueError, match="resume.dt"):
        lyapunov_spectrum(
            step, state0=jnp.array([0.3]), dt=2e-3, n_steps=100,
            renorm_every=10, resume=result.checkpoint,
        )
