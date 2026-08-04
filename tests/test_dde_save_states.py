"""``save_states``/``decimate`` trajectory capture on ``lyapunov_spectrum_dde``
-- the DDE-engine mirror of ``tests/test_save_states.py``. Uses the same
lightweight linear scalar DDE fixture as ``tests/test_dde.py``'s Tier 4.2
tests (``make_scalar_delayed_step_fn`` + ``scalar_delayed_history0``).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lyapax import systems
from lyapax.dde import (
    lyapunov_spectrum_dde,
    make_scalar_delayed_step_fn,
    resolve_tau_steps,
    scalar_delayed_history0,
)


def _make_scalar_dde(a=0.5, tau=0.3, dt=1e-2):
    tau_steps = resolve_tau_steps(tau, dt)
    rhs = systems.linear_scalar_dde(a=a)
    step_fn = make_scalar_delayed_step_fn(rhs, m=1, tau_steps=tau_steps, dt=dt)
    state0, buf0 = scalar_delayed_history0(jnp.array([1.0]), tau_steps)
    return step_fn, state0, buf0, dt


def test_save_states_false_returns_no_trajectory():
    step_fn, state0, buf0, dt = _make_scalar_dde()
    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
        n_steps=200, k=1, renorm_every=5,
    )
    assert result.trajectory is None
    assert result.save_times is None


def test_save_states_true_shapes():
    step_fn, state0, buf0, dt = _make_scalar_dde()
    n_steps = 200
    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
        n_steps=n_steps, k=1, renorm_every=5, save_states=True,
    )
    assert result.trajectory.shape == (n_steps,) + state0.shape
    assert result.save_times.shape == (n_steps,)
    np.testing.assert_allclose(np.array(result.save_times[-1]), np.array(result.times[-1]))
    np.testing.assert_allclose(np.array(result.save_times[0]), dt)


def test_save_states_matches_direct_step_fn_iteration():
    step_fn, state0, buf0, dt = _make_scalar_dde()
    n_steps = 50
    renorm_every = 5
    params = {}
    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params=params, dt=dt,
        n_steps=n_steps, k=1, renorm_every=renorm_every, save_states=True,
    )

    # Unlike the ODE engine, a DDE always runs a mandatory transient (at
    # least one full ring cycle, horizon * dt) even at the default
    # t_transient=0.0 -- reproduce that floor (see
    # lyapax.dde.lyapunov_spectrum_dde's own transient-length computation)
    # before comparing the accumulation-phase trajectory.
    horizon = buf0.shape[0]
    min_transient = horizon * dt
    n_transient = renorm_every * max(1, round(min_transient / dt / renorm_every))

    expected = []
    carry = (state0, buf0, jnp.int32(0), params)
    for _ in range(n_transient + n_steps):
        carry, new_state = step_fn(carry, None)
        expected.append(new_state)
    expected = jnp.stack(expected[n_transient:])
    np.testing.assert_allclose(
        np.array(result.trajectory), np.array(expected), atol=1e-10)


def test_decimate_is_a_stride_of_the_full_resolution_trajectory():
    step_fn, state0, buf0, dt = _make_scalar_dde()
    n_steps = 200
    params = {}
    full = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params=params, dt=dt,
        n_steps=n_steps, k=1, renorm_every=5, save_states=True, decimate=1,
    )
    decimated = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params=params, dt=dt,
        n_steps=n_steps, k=1, renorm_every=5, save_states=True, decimate=5,
    )
    np.testing.assert_array_equal(
        np.array(decimated.trajectory), np.array(full.trajectory[4::5]))
    np.testing.assert_array_equal(
        np.array(decimated.save_times), np.array(full.save_times[4::5]))


def test_transient_is_never_saved():
    a, tau, dt = 0.5, 0.3, 1e-2
    step_fn, state0, buf0, _ = _make_scalar_dde(a=a, tau=tau, dt=dt)
    params = {}
    renorm_every = 5
    t_transient = 1.0
    decimate = 1

    result = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params=params, dt=dt,
        n_steps=100, k=1, renorm_every=renorm_every,
        t_transient=t_transient, save_states=True, decimate=decimate,
    )

    # DDE transient is floored to at least one full ring cycle
    # (horizon * dt) -- reproduce that rounding independently, same as
    # lyapax.dde.lyapunov_spectrum_dde's own transient-length computation.
    horizon = buf0.shape[0]
    min_transient = horizon * dt
    n_transient = renorm_every * max(
        1, round(max(t_transient, min_transient) / dt / renorm_every))

    carry = (state0, buf0, jnp.int32(0), params)
    for _ in range(n_transient + decimate):
        carry, new_state = step_fn(carry, None)
    np.testing.assert_allclose(
        np.array(result.trajectory[0]), np.array(new_state), atol=1e-9)
    assert not np.allclose(
        np.array(result.trajectory[0]), np.array(state0), atol=1e-3)


def test_decimate_without_save_states_raises():
    step_fn, state0, buf0, dt = _make_scalar_dde()
    with pytest.raises(ValueError, match="save_states"):
        lyapunov_spectrum_dde(
            step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
            n_steps=100, k=1, renorm_every=5, decimate=5,
        )


def test_decimate_below_one_raises():
    step_fn, state0, buf0, dt = _make_scalar_dde()
    with pytest.raises(ValueError, match="decimate"):
        lyapunov_spectrum_dde(
            step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
            n_steps=100, k=1, renorm_every=5, save_states=True, decimate=0,
        )


def test_decimate_must_divide_renorm_every():
    step_fn, state0, buf0, dt = _make_scalar_dde()
    with pytest.raises(ValueError, match="renorm_every"):
        lyapunov_spectrum_dde(
            step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
            n_steps=100, k=1, renorm_every=5, save_states=True, decimate=3,
        )


def test_save_states_does_not_change_exponents():
    step_fn, state0, buf0, dt = _make_scalar_dde()
    kwargs = dict(
        step_fn_or_problem=step_fn, state0=state0, buf0=buf0, params={},
        dt=dt, n_steps=200, k=1, renorm_every=5,
    )
    baseline = lyapunov_spectrum_dde(**kwargs)
    with_states = lyapunov_spectrum_dde(**kwargs, save_states=True, decimate=5)
    np.testing.assert_array_equal(
        np.array(baseline.exponents), np.array(with_states.exponents))
    np.testing.assert_array_equal(
        np.array(baseline.history), np.array(with_states.history))


def test_save_times_offset_correctly_across_resume():
    """A resumed run's save_times must continue the elapsed-time origin of
    the first call (like times/history already do), not reset to zero --
    mirrors tests/test_dde.py::test_dde_resume_matches_single_uninterrupted_run
    but for the trajectory fields."""
    step_fn, state0, buf0, dt = _make_scalar_dde()
    params = {}
    n1, n2, renorm_every, decimate = 100, 100, 5, 5

    whole = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params=params, dt=dt,
        n_steps=n1 + n2, k=1, renorm_every=renorm_every, t_transient=2.0,
        save_states=True, decimate=decimate,
    )

    first = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params=params, dt=dt,
        n_steps=n1, k=1, renorm_every=renorm_every, t_transient=2.0,
        save_states=True, decimate=decimate,
    )
    second = lyapunov_spectrum_dde(
        step_fn, state0=state0, buf0=buf0, params=params, dt=dt,
        n_steps=n2, k=1, renorm_every=renorm_every, resume=first.checkpoint,
        save_states=True, decimate=decimate,
    )

    n1_saves = n1 // decimate
    np.testing.assert_allclose(
        np.array(second.save_times), np.array(whole.save_times[n1_saves:]),
        rtol=1e-12)
    np.testing.assert_allclose(
        np.array(second.trajectory), np.array(whole.trajectory[n1_saves:]),
        atol=1e-10)


def test_save_states_compiles_and_runs_under_jit():
    step_fn, state0, buf0, dt = _make_scalar_dde()

    @jax.jit
    def run(state0, buf0):
        return lyapunov_spectrum_dde(
            step_fn, state0=state0, buf0=buf0, params={}, dt=dt,
            n_steps=200, k=1, renorm_every=5, save_states=True, decimate=5,
        )

    result = run(state0, buf0)
    assert result.trajectory.shape == (40,) + state0.shape
    assert result.save_times.shape == (40,)
