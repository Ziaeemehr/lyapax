"""Core Benettin/QR Lyapunov-spectrum engine (M1).

Scope: single-node (or otherwise uncoupled) systems given as a one-step
map ``state -> new_state``. Networked/coupled systems (M3) and delayed
systems (M4) reuse the same renormalization idea but need extra tangent
bookkeeping (a delay ring buffer's tangent, in the DDE case) — see
notes/milestones.md.

Method: propagate a (d, k) matrix of tangent vectors alongside the
trajectory. Every ``renorm_every`` steps, QR-decompose the tangent matrix,
accumulate ``log|diag(R)|``, and replace the tangent matrix with the
orthonormal factor ``Q`` (Benettin's method).

Tangent propagation is ``jax.jvp``-based (M6), not ``jax.jacfwd``-based: one
``jax.jvp`` call per tracked column, batched via ``jax.vmap`` — cost is O(k)
forward-mode passes per raw step, not O(d) for a dense Jacobian. M1-M5 used
dense ``jax.jacfwd`` here deliberately ("start dense, don't add matrix-free
machinery before a concrete need" — see notes/milestones.md); M4's DDE
engine (``lyapax.dde.lyapunov_spectrum_dde``) already needed and validated
this exact jvp/vmap pattern for a much larger augmented ``(state, buf)``
dimension, so M6 carries it back here for the plain (no ring buffer) case,
where it matters whenever ``k < d`` (the partial-spectrum case ``jacfwd``
can't exploit, since it always computes all ``d`` columns regardless of
how many are actually tracked).
"""
from __future__ import annotations

import warnings
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

StepFn = Callable[[jnp.ndarray], jnp.ndarray]


def _warn_if_not_x64(state0: jnp.ndarray) -> None:
    """Lyapunov exponents are long-horizon averages of log-growth rates;
    JAX's default float32 silently degrades them (see notes/milestones.md,
    risk #1). Warn once per call site rather than raising, since a caller
    computing genuinely short/coarse estimates may accept the precision
    loss -- but the default (x64 disabled) is very rarely what a caller
    actually wants here.
    """
    if not jax.config.jax_enable_x64 and jnp.asarray(state0).dtype in (
            jnp.float32, jnp.complex64):
        warnings.warn(
            "lyapax: jax_enable_x64 is not set, so state0 is float32 -- "
            "Lyapunov exponent estimates accumulate log-growth over many "
            "steps and are known to degrade under float32 (see "
            "notes/milestones.md, risk #1). Call "
            "`jax.config.update('jax_enable_x64', True)` before importing "
            "jax.numpy arrays, or pass a float64 state0 once x64 is "
            "enabled.",
            stacklevel=3,
        )


class LyapunovResult(NamedTuple):
    exponents: jnp.ndarray
    """(k,) final Lyapunov-exponent estimates, sorted descending."""

    history: jnp.ndarray
    """(n_renorm, k) running estimate at each renormalization point, in the
    same column order as ``exponents`` — use to check convergence.

    Columns are ordered once, by the *final* row (``history[-1]`` ==
    ``exponents``), then that column order is applied to every row. Near-
    degenerate exponents can cross over during the run, so an early row's
    per-column values are not guaranteed to be individually sorted or to
    track the same Oseledets direction throughout — only the last row is."""

    times: jnp.ndarray
    """(n_renorm,) elapsed time (in units of ``dt``) at each row of
    ``history``, measured from the end of the transient."""


def _run_renorm_scan(
        renorm_block, carry0, n_renorm: int, renorm_every: int, dt: float,
) -> LyapunovResult:
    """Shared tail between ``lyapunov_spectrum`` (ODE) and
    ``lyapax.dde.lyapunov_spectrum_dde``: scan ``renorm_block`` (which must
    return ``(new_carry, log_growth (k,))`` per call, log-growth already
    ``log|diag(R)|`` from one QR-renormalization block) ``n_renorm`` times,
    then turn the per-block log-growth into cumulative running exponent
    estimates. Not part of the public API -- the only thing that differs
    between the ODE and DDE engines is what one renorm block's tangent
    propagation looks like, not this bookkeeping.
    """
    _final_carry, log_growth_per_block = jax.lax.scan(
        renorm_block, carry0, None, length=n_renorm)

    cum_log_growth = jnp.cumsum(log_growth_per_block, axis=0)  # (n_renorm, k)
    block_times = (jnp.arange(1, n_renorm + 1) * renorm_every) * dt
    history = cum_log_growth / block_times[:, None]

    order = jnp.argsort(-history[-1])
    history = history[:, order]
    exponents = history[-1]

    return LyapunovResult(exponents=exponents, history=history, times=block_times)


def _check_finite(result: LyapunovResult) -> None:
    """Raise if any running estimate is non-finite (``check_finite=True``).

    Only meaningful for eager (non-``jit``-wrapped) calls: it forces
    ``result.history`` to a concrete value, which fails under tracing.
    """
    if not bool(jnp.all(jnp.isfinite(result.history))):
        raise FloatingPointError(
            "non-finite Lyapunov exponent estimate encountered -- a QR "
            "diagonal entry underflowed/overflowed (log(R_ii) was 0, inf, "
            "or NaN), most likely from renorm_every being too large for "
            "this system's growth rate, or a diverging/NaN-producing "
            "step_fn. See renorm_every's docstring for the tangent-"
            "overflow tradeoff, or pass check_finite=False to disable "
            "this check."
        )


def lyapunov_spectrum(
        step_fn: StepFn,
        state0: jnp.ndarray,
        dt: float,
        n_steps: int,
        k: int | None = None,
        renorm_every: int = 1,
        t_transient: float = 0.0,
        seed: int = 0,
        check_finite: bool = False,
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
    check_finite : if True, raise ``FloatingPointError`` when any running
        estimate in ``history`` is non-finite (a QR diagonal underflowed to
        0 or overflowed to inf, e.g. from ``renorm_every`` too large).
        Off by default and only usable when this function is called
        eagerly (not wrapped in ``jax.jit``).

    Returns
    -------
    LyapunovResult
    """
    state0 = jnp.asarray(state0)
    _warn_if_not_x64(state0)
    d = state0.shape[0]
    if k is None:
        k = d
    if not (1 <= k <= d):
        raise ValueError(f"k must be in [1, {d}]; got {k}.")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0; got {n_steps}.")
    if renorm_every < 1:
        raise ValueError(f"renorm_every must be >= 1; got {renorm_every}.")
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

            def _single_column(y_col):
                return jax.jvp(step_fn, (state_i,), (y_col,))

            new_state_rep, new_Y = jax.vmap(
                _single_column, in_axes=-1, out_axes=(0, -1)
            )(Y_i)
            new_state = new_state_rep[0]
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

    def _renorm_block(carry, _):
        state, Y = carry
        state, Q, R = _advance(state, Y, renorm_every)
        log_growth = jnp.log(jnp.abs(jnp.diag(R)))
        return (state, Q), log_growth

    n_renorm = n_steps // renorm_every
    result = _run_renorm_scan(_renorm_block, (state0, Y0), n_renorm, renorm_every, dt)
    if check_finite:
        _check_finite(result)
    return result
