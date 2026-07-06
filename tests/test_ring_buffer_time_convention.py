"""Deterministic, Lyapunov-free check of the ring buffer's time convention:
see notes/possible_solution_to_open_issues.md's "Suspected Ring-Buffer
Off-by-One" and notes/open_issues.md item 2.

Invariant under test, independent of any dfun/coupling/QR machinery:

    slot (k % horizon) stores the coupling-variable value at physical
    time k * dt, and a delayed read at step t for a tau of tau_steps
    returns the value at physical time (t - tau_steps) * dt (or the
    constant-history value at t=0 if that time is <= 0).

This directly exercises the same low-level primitives ``lyapax.simulator
.step.make_step_fn`` calls, at the write index ``t + 1`` (the fix for the
off-by-one: the newly integrated state is the value at time (t+1)*dt, not
t*dt, so it must be written under slot t + 1).
"""
import jax.numpy as jnp

from lyapax.simulator.step import (
    _read_uniform_delayed_cvar,
    _read_uniform_delayed_cvar_interp,
    _write_ring,
    _write_ring_interp,
)

HORIZON = 8
N_STEPS = 20


def _value_at_time(k: int) -> jnp.ndarray:
    """An arbitrary but deterministic, injective function of the integer
    time index, shaped like a (n_cvar=1, n_nodes=1) coupling-variable
    state -- any wrong slot read is immediately visible as a wrong number,
    not masked by two steps happening to hold equal values."""
    return jnp.array([[100.0 + k]])


def _expected_history(k: int) -> jnp.ndarray:
    """Constant-history convention: the delayed value at any k <= 0 is the
    t=0 value, not an extrapolation."""
    return _value_at_time(max(k, 0))


def test_exact_grid_write_read_recovers_physical_time():
    """Mirrors make_step_fn's own read-then-write order within one
    iteration: a delayed read at step ``t`` must use the buffer as it
    stands *before* that iteration's own write (which lands in slot
    ``t + 1``), matching ``step()``'s ``coupling_at`` closing over the
    pre-write ``buf``. Checking reads only against the buffer state at
    their own time -- not the fully-advanced final buffer -- is required
    because later writes legitimately overwrite old slots."""
    buf = jnp.tile(_value_at_time(0)[None], (HORIZON, 1, 1))
    for t in range(N_STEPS):
        for tau_steps in range(0, HORIZON):
            read = _read_uniform_delayed_cvar(buf, t, tau_steps, HORIZON)
            expected = _expected_history(t - tau_steps)
            assert jnp.allclose(read, expected), (
                f"t={t}, tau_steps={tau_steps}: read {read} != expected {expected}"
            )
        buf = _write_ring(buf, t + 1, _value_at_time(t + 1), HORIZON)


def test_interpolated_write_read_recovers_physical_time_at_integer_offsets():
    dt = 0.1
    buf = jnp.stack(
        [jnp.tile(_value_at_time(0)[None], (HORIZON, 1, 1)),
         jnp.zeros((HORIZON, 1, 1))],
        axis=1,
    )
    # Constant derivative (1 per step) so the exact-grid Hermite read (theta=0)
    # must reduce to the plain stored value regardless of the derivative term.
    deriv = jnp.array([[1.0 / dt]])
    for t in range(N_STEPS):
        for tau_steps in range(0, HORIZON - 1):
            read = _read_uniform_delayed_cvar_interp(buf, t, float(tau_steps), HORIZON, dt)
            expected = _expected_history(t - tau_steps)
            assert jnp.allclose(read, expected, atol=1e-10), (
                f"t={t}, tau_steps={tau_steps}: read {read} != expected {expected}"
            )
        buf = _write_ring_interp(buf, t + 1, _value_at_time(t + 1), deriv, HORIZON)
