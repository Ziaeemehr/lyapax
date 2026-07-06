"""Hermite-interpolated DDE history reads (make_step_fn(..., interpolate=True)).

See notes/stepping_accuracy_review.md (Part B/C) for the design. With
interpolate=True the delayed history is re-read at each integrator
stage's own intra-step time via the cubic Hermite interpolant, so the
integrator's nominal convergence order is actually realized (capped at
~4 by the interpolant itself): heun measures ~2.0 on the scalar linear
DDE, rk4 ~4. This was once misdiagnosed as impossible -- the per-stage
read looked ineffective when first tried, but only because the
ring-buffer write index was then off by one full step, a second,
independent O(dt) bias masking the gain (see the note's Part C for the
full post-mortem). The grid-snapped default (interpolate=False) remains
O(dt) by construction: it freezes one stored sample across the step, and
its tau is rounded to the grid besides.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import lambertw

from lyapax import systems
from lyapax.dde import dde_problem, lyapunov_spectrum_dde

jax.config.update("jax_enable_x64", True)

A, TAU, DT = 0.5, 0.317, 1e-2  # tau deliberately not a multiple of dt


def _lambert_w_exponent(a: float, tau: float) -> float:
    return float((lambertw(-a * tau, k=0) / tau).real)


def test_interpolate_matches_grid_snap_exactly_at_integer_tau_euler():
    """With euler (a single stage at c=0) and tau an exact multiple of dt,
    every interpolated read has theta=0 and lands exactly on a stored grid
    sample, so interpolate=True must reproduce interpolate=False
    bit-for-bit (up to roundoff). Multi-stage integrators are *expected*
    to differ here: their later stages read the history at intra-step
    times the grid-snapped path freezes over -- that difference is the
    accuracy fix, not a bug."""
    rhs = systems.linear_scalar_dde(a=A)
    dt, tau = 1e-2, 0.02  # tau_steps = 2.0 exactly

    problem = dde_problem(rhs, state0=jnp.array([1.0]), tau=tau, dt=dt, integrator="euler")
    problem_i = dde_problem(
        rhs, state0=jnp.array([1.0]), tau=tau, dt=dt, integrator="euler", interpolate=True)

    result = lyapunov_spectrum_dde(problem, n_steps=20_000, k=1, renorm_every=5, t_transient=10.0)
    result_i = lyapunov_spectrum_dde(
        problem_i, n_steps=20_000, k=1, renorm_every=5, t_transient=10.0)

    assert float(result.exponents[0]) == pytest.approx(float(result_i.exponents[0]), abs=1e-10)


def test_interpolate_accepts_non_multiple_tau_without_warning():
    """A tau that isn't a clean multiple of dt triggers resolve_tau_steps's
    rounding warning by default, but interpolate=True should use tau
    exactly (no rounding, no warning)."""
    rhs = systems.linear_scalar_dde(a=A)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here is a test failure
        problem = dde_problem(
            rhs, state0=jnp.array([1.0]), tau=TAU, dt=DT, integrator="heun", interpolate=True)

    assert problem.tau_steps == pytest.approx(TAU / DT)


def test_interpolate_heun_converges_at_second_order():
    """With per-stage Hermite history reads, heun should realize its full
    nominal order ~2 on the scalar linear DDE's exponent (vs the Lambert-W
    analytic answer) -- the regression this guards is the pair of O(dt)
    caps that used to hide it (frozen per-step reads, and the ring-buffer
    write off-by-one), either of which alone drags the order back to ~1."""
    rhs = systems.linear_scalar_dde(a=A)
    expected = _lambert_w_exponent(A, TAU)

    dts = [2e-2, 1e-2, 5e-3, 2.5e-3]
    errors = []
    for dt in dts:
        problem = dde_problem(
            rhs, state0=jnp.array([1.0]), tau=TAU, dt=dt, integrator="heun", interpolate=True)
        result = lyapunov_spectrum_dde(
            problem, n_steps=40_000, k=1, renorm_every=5, t_transient=20.0)
        errors.append(abs(float(result.exponents[0]) - expected))

    for e_coarse, e_fine in zip(errors, errors[1:]):
        assert e_fine < e_coarse

    orders = [
        np.log(e_coarse / e_fine) / np.log(2.0)
        for e_coarse, e_fine in zip(errors, errors[1:])
    ]
    assert all(order == pytest.approx(2.0, abs=0.3) for order in orders)


def test_interpolate_rk4_reaches_hermite_accuracy_floor():
    """rk4 + per-stage Hermite reads converges at ~4th order, so at
    dt=1e-2 its exponent error should already be near the estimation
    noise floor -- orders of magnitude below heun's ~2e-6 at the same dt.
    A loose absolute bound (not an order fit) keeps this cheap and
    non-flaky while still catching any regression back to O(dt) or
    O(dt^2)."""
    rhs = systems.linear_scalar_dde(a=A)
    expected = _lambert_w_exponent(A, TAU)

    problem = dde_problem(
        rhs, state0=jnp.array([1.0]), tau=TAU, dt=DT, integrator="rk4", interpolate=True)
    result = lyapunov_spectrum_dde(
        problem, n_steps=40_000, k=1, renorm_every=5, t_transient=20.0)

    assert abs(float(result.exponents[0]) - expected) < 1e-9


def test_interpolate_rejects_legacy_per_edge_path():
    """interpolate=True is only wired up for the uniform-tau_steps +
    coupling_fn path -- see lyapax.simulator.make_step_fn's docstring."""
    from lyapax.simulator import make_step_fn

    with pytest.raises(ValueError):
        make_step_fn(
            dfun=lambda state, coupling, params: state,
            weights=jnp.ones((1, 1)), has_delays=True, horizon=3, n_nodes=1,
            cvar_indices=(0,), dt=DT, interpolate=True,
        )


def test_interpolate_rejects_sub_step_delay():
    """tau < dt (tau_steps < 1) must be rejected with interpolate=True:
    the per-stage reads at intra-step times up to step+1 (and the stored-
    derivative computation) would touch ring-buffer slots not yet written
    -- see make_step_fn's validation."""
    rhs = systems.linear_scalar_dde(a=A)

    with pytest.raises(ValueError, match="tau_steps >= 1"):
        dde_problem(
            rhs, state0=jnp.array([1.0]), tau=0.005, dt=1e-2,
            integrator="heun", interpolate=True)
