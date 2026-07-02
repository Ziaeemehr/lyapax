"""Core Benettin/QR Lyapunov-spectrum engine (M1).

Scope: single-node (or otherwise uncoupled) systems given as a one-step
map ``state -> new_state``. Networked/coupled systems (M3) and delayed
systems (M4) reuse the same renormalization idea but need extra tangent
bookkeeping (a delay ring buffer's tangent, in the DDE case) — see
notes/milestones.md.

Method: propagate a (d, k) matrix of tangent vectors alongside the
trajectory using the exact Jacobian of ``step_fn`` at each step
(``jax.jacfwd`` — dense; matrix-free jvp-based propagation is deferred to
M6 per notes/milestones.md, "start dense, don't add matrix-free machinery
before a concrete network size needs it"). Every ``renorm_every`` steps,
QR-decompose the tangent matrix, accumulate ``log|diag(R)|``, and replace
the tangent matrix with the orthonormal factor ``Q`` (Benettin's method).
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

StepFn = Callable[[jnp.ndarray], jnp.ndarray]


class LyapunovResult(NamedTuple):
    exponents: jnp.ndarray
    """(k,) final Lyapunov-exponent estimates, sorted descending."""

    history: jnp.ndarray
    """(n_renorm, k) running estimate at each renormalization point, in the
    same column order as ``exponents`` — use to check convergence."""

    times: jnp.ndarray
    """(n_renorm,) elapsed time (in units of ``dt``) at each row of
    ``history``, measured from the end of the transient."""


def lyapunov_spectrum(
        step_fn: StepFn,
        state0: jnp.ndarray,
        dt: float,
        n_steps: int,
        k: int | None = None,
        renorm_every: int = 1,
        t_transient: float = 0.0,
        seed: int = 0,
) -> LyapunovResult:
    """
    Compute the (partial or full) Lyapunov spectrum of ``step_fn`` along the
    trajectory started from ``state0``, via the Benettin/QR method.

    Parameters
    ----------
    step_fn : one fixed-time-step (or one map-iterate) update,
        ``state (d,) -> new_state (d,)``. Must be a pure, differentiable
        JAX function of ``state`` alone — close over any parameters.
    state0 : (d,) initial state (pre-transient).
    dt : time represented by one call to ``step_fn``. For discrete maps,
        pass ``dt=1.0`` and interpret the exponents as per-iterate.
    n_steps : number of steps to run *after* the transient. Must be a
        multiple of ``renorm_every``.
    k : number of leading exponents to track (``k <= d``). Defaults to the
        full spectrum (``k = d``). Cost scales with ``k``, not ``d`` — this
        is the "only the first few largest exponents" case.
    renorm_every : QR-renormalize every this many steps. Larger values
        reduce QR overhead but risk tangent-vector overflow/underflow for
        fast-growing/shrinking directions (see notes/milestones.md, risk
        #2) — keep small enough that
        ``exp(|lambda_max| * renorm_every * dt)`` stays well within
        float64 range.
    t_transient : time to integrate (discarding tangent tracking) before
        starting the Lyapunov accumulation.
    seed : PRNG seed for the initial random tangent-vector basis.

    Returns
    -------
    LyapunovResult
    """
    state0 = jnp.asarray(state0)
    d = state0.shape[0]
    if k is None:
        k = d
    if not (1 <= k <= d):
        raise ValueError(f"k must be in [1, {d}]; got {k}.")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0; got {n_steps}.")
    if n_steps % renorm_every != 0:
        raise ValueError(
            f"n_steps ({n_steps}) must be a multiple of renorm_every "
            f"({renorm_every})."
        )

    key = jax.random.PRNGKey(seed)
    Y0 = jax.random.normal(key, (d, k), dtype=state0.dtype)
    Y0, _ = jnp.linalg.qr(Y0)

    def _advance(state, Y, n_substeps):
        """Propagate state and tangent matrix jointly for n_substeps raw
        steps, then QR once. Used for both the transient (alignment) phase
        and each accumulation block, so a long transient still gets
        periodic renormalization and can't overflow (risk #2 in
        notes/milestones.md)."""
        def _inner(carry_inner, _):
            state_i, Y_i = carry_inner
            jac = jax.jacfwd(step_fn)(state_i)
            new_state = step_fn(state_i)
            new_Y = jac @ Y_i
            return (new_state, new_Y), None

        (state, Y), _ = jax.lax.scan(_inner, (state, Y), None, length=n_substeps)
        Q, R = jnp.linalg.qr(Y)
        return state, Q, R

    if t_transient > 0.0:
        # Chunk the transient at the same cadence as the main run: the
        # transient isn't just letting the *state* reach the attractor, it's
        # also letting the tangent vectors align to the Oseledets subspaces
        # (see M1 test notes) -- without this, a random initial Y0 biases
        # the estimate for a linear/weakly-mixing system noticeably.
        n_transient = renorm_every * max(1, round(t_transient / dt / renorm_every))

        def _transient_block(carry, _):
            state, Y = carry
            state, Q, _R = _advance(state, Y, renorm_every)
            return (state, Q), None

        (state0, Y0), _ = jax.lax.scan(
            _transient_block, (state0, Y0), None, length=n_transient // renorm_every)

    n_renorm = n_steps // renorm_every

    def _renorm_block(carry, _):
        state, Y = carry
        state, Q, R = _advance(state, Y, renorm_every)
        log_growth = jnp.log(jnp.abs(jnp.diag(R)))
        return (state, Q), log_growth

    (_final_state, _final_Y), log_growth_per_block = jax.lax.scan(
        _renorm_block, (state0, Y0), None, length=n_renorm)

    cum_log_growth = jnp.cumsum(log_growth_per_block, axis=0)  # (n_renorm, k)
    block_times = (jnp.arange(1, n_renorm + 1) * renorm_every) * dt
    history = cum_log_growth / block_times[:, None]

    order = jnp.argsort(-history[-1])
    history = history[:, order]
    exponents = history[-1]

    return LyapunovResult(exponents=exponents, history=history, times=block_times)
