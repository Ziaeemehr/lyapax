"""Hermite-interpolated DDE history reads (make_step_fn(..., interpolate=True)).

See notes/stepping_accuracy_review.md (Part B) for the design and the
empirically-measured accuracy tradeoff: interpolation removes the
integer-step delay rounding (tau no longer snaps to the nearest dt
multiple, and its error no longer aliases non-monotonically as dt
shrinks), but the realized convergence order is capped at O(dt), not the
higher order a textbook Hermite interpolant would give with exact
derivatives -- that cap does not come from the interpolation formula
itself (verified in isolation to be near its expected 4th order). A later
follow-up fixed the analogous "coupling held fixed across one integrator
step" issue for zero-delay coupled networks (recomputing coupling at each
stage's own intra-step state instead), but the same idea applied to the
delayed lookup here -- re-deriving it at each stage's own intra-step time
via the interpolant, instead of once per step -- did not reduce this
O(dt) cap, for reasons not yet understood. See the note for what was
ruled out.
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


def test_interpolate_matches_grid_snap_exactly_at_integer_tau():
    """theta=0 at every read (tau an exact multiple of dt) should make the
    Hermite formula reduce to the plain stored value -- i.e. interpolate=True
    must reproduce interpolate=False bit-for-bit (up to roundoff) whenever
    there's no fractional part to interpolate over."""
    rhs = systems.linear_scalar_dde(a=A)
    dt, tau = 1e-2, 0.02  # tau_steps = 2.0 exactly

    problem = dde_problem(rhs, state0=jnp.array([1.0]), tau=tau, dt=dt, integrator="heun")
    problem_i = dde_problem(
        rhs, state0=jnp.array([1.0]), tau=tau, dt=dt, integrator="heun", interpolate=True)

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


def test_interpolate_converges_monotonically_as_dt_shrinks():
    """The known failure mode this feature fixes: grid-snapped tau_eff can
    land closer to or further from the true tau at a smaller dt (rounding
    aliasing), so its error vs dt is not necessarily monotonic. Hermite
    interpolation should give a clean, monotonically shrinking error at
    consecutively halved dt, unlike the grid-snapped default -- see the
    module docstring for why this is O(dt), not higher order."""
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

    # Order should be close to 1 (not the ~4th order an exact-derivative
    # Hermite interpolant would give -- see the coupling-freezing cap).
    orders = [
        np.log(e_coarse / e_fine) / np.log(2.0)
        for e_coarse, e_fine in zip(errors, errors[1:])
    ]
    assert all(order == pytest.approx(1.0, abs=0.3) for order in orders)


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
