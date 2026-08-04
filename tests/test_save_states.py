"""``save_states``/``decimate`` trajectory capture on ``lyapunov_spectrum``
(the ODE engine). See ``tests/test_dde.py`` for the DDE-engine mirror and
``tests/test_sweep.py`` for the vmapped-sweep behavior.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lyapax import systems
from lyapax.core import lyapunov_spectrum
from lyapax.integrators import rk4_step


def _make_linear_step(rate=-0.5, dt=1e-2):
    A = jnp.array([[rate]])
    return rk4_step(systems.linear_system(A), dt), dt


def test_save_states_false_returns_no_trajectory():
    step, dt = _make_linear_step()
    result = lyapunov_spectrum(
        step, state0=jnp.array([1.0]), dt=dt, n_steps=200, renorm_every=10,
    )
    assert result.trajectory is None
    assert result.save_times is None


def test_save_states_true_default_decimate_shapes():
    step, dt = _make_linear_step()
    n_steps = 200
    result = lyapunov_spectrum(
        step, state0=jnp.array([1.0]), dt=dt, n_steps=n_steps,
        renorm_every=10, save_states=True,
    )
    assert result.trajectory.shape == (n_steps, 1)
    assert result.save_times.shape == (n_steps,)
    np.testing.assert_allclose(np.array(result.save_times[-1]), np.array(result.times[-1]))
    np.testing.assert_allclose(np.array(result.save_times[0]), dt)


def test_save_states_matches_direct_step_fn_iteration():
    step, dt = _make_linear_step()
    n_steps = 50
    state0 = jnp.array([1.0])
    result = lyapunov_spectrum(
        step, state0=state0, dt=dt, n_steps=n_steps, renorm_every=10,
        save_states=True,
    )

    expected = []
    s = state0
    for _ in range(n_steps):
        s = step(s)
        expected.append(s)
    expected = jnp.stack(expected)
    np.testing.assert_allclose(
        np.array(result.trajectory), np.array(expected), atol=1e-10)


def test_decimate_is_a_stride_of_the_full_resolution_trajectory():
    step, dt = _make_linear_step()
    n_steps = 200
    state0 = jnp.array([1.0])
    full = lyapunov_spectrum(
        step, state0=state0, dt=dt, n_steps=n_steps, renorm_every=10,
        save_states=True, decimate=1,
    )
    decimated = lyapunov_spectrum(
        step, state0=state0, dt=dt, n_steps=n_steps, renorm_every=10,
        save_states=True, decimate=5,
    )
    np.testing.assert_array_equal(
        np.array(decimated.trajectory), np.array(full.trajectory[4::5]))
    np.testing.assert_array_equal(
        np.array(decimated.save_times), np.array(full.save_times[4::5]))


def test_transient_is_never_saved():
    rate, dt = -0.5, 1e-2
    step, _ = _make_linear_step(rate=rate, dt=dt)
    state0 = jnp.array([1.0])
    renorm_every = 10
    t_transient = 1.0
    decimate = 1

    result = lyapunov_spectrum(
        step, state0=state0, dt=dt, n_steps=100, renorm_every=renorm_every,
        t_transient=t_transient, save_states=True, decimate=decimate,
    )

    # Independently reproduce the transient-step count and iterate the raw
    # step_fn -- the first saved row must be the state *after* the
    # transient (plus one decimate stride), not near state0.
    n_transient = renorm_every * max(1, round(t_transient / dt / renorm_every))
    s = state0
    for _ in range(n_transient + decimate):
        s = step(s)
    np.testing.assert_allclose(
        np.array(result.trajectory[0]), np.array(s), atol=1e-9)
    assert not np.allclose(
        np.array(result.trajectory[0]), np.array(state0), atol=1e-3)


def test_decimate_without_save_states_raises():
    step, dt = _make_linear_step()
    with pytest.raises(ValueError, match="save_states"):
        lyapunov_spectrum(
            step, state0=jnp.array([1.0]), dt=dt, n_steps=100,
            renorm_every=10, decimate=5,
        )


def test_decimate_below_one_raises():
    step, dt = _make_linear_step()
    with pytest.raises(ValueError, match="decimate"):
        lyapunov_spectrum(
            step, state0=jnp.array([1.0]), dt=dt, n_steps=100,
            renorm_every=10, save_states=True, decimate=0,
        )


def test_decimate_must_divide_renorm_every():
    step, dt = _make_linear_step()
    with pytest.raises(ValueError, match="renorm_every"):
        lyapunov_spectrum(
            step, state0=jnp.array([1.0]), dt=dt, n_steps=100,
            renorm_every=10, save_states=True, decimate=3,
        )


def test_save_states_does_not_change_exponents():
    step, dt = _make_linear_step()
    kwargs = dict(
        step_fn_or_problem=step, state0=jnp.array([1.0]), dt=dt,
        n_steps=200, renorm_every=10,
    )
    baseline = lyapunov_spectrum(**kwargs)
    with_states = lyapunov_spectrum(**kwargs, save_states=True, decimate=5)
    np.testing.assert_array_equal(
        np.array(baseline.exponents), np.array(with_states.exponents))
    np.testing.assert_array_equal(
        np.array(baseline.history), np.array(with_states.history))


def test_save_states_compiles_and_runs_under_jit():
    step, dt = _make_linear_step()

    @jax.jit
    def run(state0):
        return lyapunov_spectrum(
            step, state0, dt, n_steps=200, renorm_every=10,
            save_states=True, decimate=5,
        )

    result = run(jnp.array([1.0]))
    assert result.trajectory.shape == (40, 1)
    assert result.save_times.shape == (40,)
