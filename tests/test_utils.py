"""``lyapax.utils.simulate_trajectory``, including its ``decimate``/
``n_transient`` extension (for ``lyapax.sweep.sweep_trajectory``'s replay
path -- see ``tests/test_sweep.py``).
"""
import jax.numpy as jnp
import numpy as np
import pytest

from lyapax import systems
from lyapax.integrators import rk4_step
from lyapax.utils import simulate_trajectory


def _make_linear_step(rate=-0.5, dt=1e-2):
    A = jnp.array([[rate]])
    return rk4_step(systems.linear_system(A), dt), dt


def test_defaults_match_pre_existing_behavior():
    """decimate=1, n_transient=0 (the defaults) must reproduce exactly what
    simulate_trajectory returned before this feature existed: state0
    followed by n_steps raw steps, one row each."""
    step, dt = _make_linear_step()
    state0 = jnp.array([1.0])
    n_steps = 20

    traj = simulate_trajectory(step, state0, n_steps)

    expected = [state0]
    s = state0
    for _ in range(n_steps):
        s = step(s)
        expected.append(s)
    expected = jnp.stack(expected)

    assert traj.shape == (n_steps + 1, 1)
    np.testing.assert_allclose(np.array(traj), np.array(expected), atol=1e-12)


def test_decimate_keeps_state0_then_strides():
    step, dt = _make_linear_step()
    state0 = jnp.array([1.0])
    n_steps = 20
    decimate = 4

    traj = simulate_trajectory(step, state0, n_steps, decimate=decimate)

    full = []
    s = state0
    for _ in range(n_steps):
        s = step(s)
        full.append(s)
    full = jnp.stack(full)

    assert traj.shape == (n_steps // decimate + 1, 1)
    np.testing.assert_array_equal(np.array(traj[0]), np.array(state0))
    np.testing.assert_array_equal(
        np.array(traj[1:]), np.array(full[decimate - 1::decimate]))


def test_n_transient_skips_states_before_the_window():
    step, dt = _make_linear_step()
    state0 = jnp.array([1.0])
    n_transient = 15
    n_steps = 10

    traj = simulate_trajectory(step, state0, n_steps, n_transient=n_transient)

    s = state0
    for _ in range(n_transient):
        s = step(s)
    post_transient_state = s

    np.testing.assert_allclose(
        np.array(traj[0]), np.array(post_transient_state), atol=1e-12)
    assert not np.allclose(np.array(traj[0]), np.array(state0), atol=1e-3)


def test_decimate_must_divide_n_steps():
    step, dt = _make_linear_step()
    with pytest.raises(ValueError, match="decimate"):
        simulate_trajectory(step, jnp.array([1.0]), n_steps=10, decimate=3)


def test_dt_output_scales_times_by_decimate():
    step, dt = _make_linear_step()
    n_steps = 20
    decimate = 4
    t, traj = simulate_trajectory(
        step, jnp.array([1.0]), n_steps, dt=dt, decimate=decimate)

    assert t.shape[0] == traj.shape[0]
    np.testing.assert_allclose(np.array(t[0]), 0.0)
    np.testing.assert_allclose(np.array(t[1]), decimate * dt)
    np.testing.assert_allclose(np.array(t[-1]), n_steps * dt)
