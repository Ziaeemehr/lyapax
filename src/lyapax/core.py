"""Core Benettin/QR Lyapunov-spectrum engine.

Scope: single-node (or otherwise uncoupled) systems given as a one-step
map ``state -> new_state``. Networked/coupled systems (``lyapax.network``)
and delayed systems (``lyapax.dde``) reuse the same renormalization idea
but need extra tangent bookkeeping (a delay ring buffer's tangent, in the
DDE case).

Method: propagate a (d, k) matrix of tangent vectors alongside the
trajectory. Every ``renorm_every`` steps, QR-decompose the tangent matrix,
accumulate ``log|diag(R)|``, and replace the tangent matrix with the
orthonormal factor ``Q`` (Benettin's method).

Tangent propagation is ``jax.jvp``-based, not ``jax.jacfwd``-based: one
``jax.jvp`` call per tracked column, batched via ``jax.vmap`` — cost is O(k)
forward-mode passes per raw step, not O(d) for a dense Jacobian, which
matters whenever ``k < d`` (the partial-spectrum case ``jacfwd`` can't
exploit, since it always computes all ``d`` columns regardless of how many
are actually tracked). See :ref:`matrix-free-tangent` for the design
rationale.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .integrators import ode_step

StepFn = Callable[[jnp.ndarray], jnp.ndarray]


def _warn_if_not_x64(state0: jnp.ndarray) -> None:
    """Lyapunov exponents are long-horizon averages of log-growth rates;
    JAX's default float32 silently degrades them (see
    :ref:`precision-requirements`). Warn once per call site rather than
    raising, since a caller
    computing genuinely short/coarse estimates may accept the precision
    loss -- but the default (x64 disabled) is very rarely what a caller
    actually wants here.
    """
    if not jax.config.jax_enable_x64 and jnp.asarray(state0).dtype in (
            jnp.float32, jnp.complex64):
        warnings.warn(
            "lyapax: jax_enable_x64 is not set, so state0 is float32 -- "
            "Lyapunov exponent estimates accumulate log-growth over many "
            "steps and are known to degrade under float32 (see the "
            "'Precision requirements' section of the lyapax docs). Call "
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


@dataclass(frozen=True)
class ODEProblem:
    """Owns the ``(step_fn, state0, dt)`` triple that ``lyapunov_spectrum``
    otherwise asks the caller to keep in sync by hand -- in particular, the
    ``dt`` baked into ``step_fn`` (e.g. via ``ode_step``) and the ``dt``
    passed separately to ``lyapunov_spectrum`` must agree, and nothing
    checks that when they're two independent arguments. Build one with
    ``ode_problem`` (plain ODE) or ``network_problem`` (coupled network),
    or construct directly if you already have a step_fn.

    :param step_fn: one fixed-time-step update, ``state (d,) -> new_state (d,)``.
    :param state0: ``(d,)`` initial state (pre-transient).
    :param dt: time represented by one call to ``step_fn``.
    """
    step_fn: StepFn
    state0: jnp.ndarray
    dt: float


def ode_problem(
        rhs,
        state0: jnp.ndarray,
        dt: float,
        integrator: str | Callable = "rk4",
) -> ODEProblem:
    """
    Build an ``ODEProblem`` for a plain (uncoupled) ODE -- bundles
    ``ode_step``'s ``step_fn`` together with ``state0`` and ``dt`` so
    ``lyapunov_spectrum(problem, n_steps=...)`` never needs ``dt`` (or
    ``state0``) passed a second time.

    :param rhs: right-hand side, ``state (d,) -> dstate (d,)``.
    :param state0: ``(d,)`` initial state (pre-transient).
    :param dt: fixed step size.
    :param integrator: ``"euler"``, ``"heun"``, ``"rk4"``, ``"rk6"``, or a callable
        ``(rhs, dt) -> step_fn`` -- see ``lyapax.integrators.ode_step``.
    """
    step_fn = ode_step(rhs, dt, integrator=integrator)
    return ODEProblem(step_fn=step_fn, state0=jnp.asarray(state0), dt=dt)


def lyapunov_spectrum(
        step_fn_or_problem: StepFn | ODEProblem,
        state0: jnp.ndarray | int | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
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
    step_fn_or_problem : either an ``ODEProblem`` (from ``ode_problem`` /
        ``network_problem``) -- in which case ``state0`` and ``dt`` are
        read off it and the second positional argument is ``n_steps``
        (``lyapunov_spectrum(problem, n_steps)`` or
        ``lyapunov_spectrum(problem, n_steps=...)``) -- or a plain
        ``step_fn``, ``state (d,) -> new_state (d,)``, in which case
        ``state0``, ``dt``, ``n_steps`` are all given explicitly (the
        original, lower-level call form). Must be a pure, differentiable
        JAX function of ``state`` alone — close over any parameters.
    state0 : (d,) initial state (pre-transient). Ignored (and may be
        omitted) when an ``ODEProblem`` is passed.
    dt : time represented by one call to ``step_fn``. For discrete maps,
        pass ``dt=1.0`` and interpret the exponents as per-iterate. When
        an ``ODEProblem`` is passed, this may be omitted; if provided, it
        must match the ``dt`` stored on the problem.
    n_steps : number of steps to run *after* the transient. Must be a
        multiple of ``renorm_every``.
    k : number of leading exponents to track (``k <= d``). Defaults to the
        full spectrum (``k = d``). Cost scales with ``k``, not ``d`` — this
        is the "only the first few largest exponents" case.
    renorm_every : QR-renormalize every this many steps. Larger values
        reduce QR overhead but risk tangent-vector overflow/underflow for
        fast-growing/shrinking directions (see
        :ref:`choosing-renorm-every`) — keep small enough that
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
    if isinstance(step_fn_or_problem, ODEProblem):
        problem = step_fn_or_problem
        if dt is not None and float(dt) != float(problem.dt):
            raise ValueError(
                "dt was passed both directly and via ODEProblem with "
                f"different values: dt={dt!r}, problem.dt={problem.dt!r}."
            )
        if n_steps is None:
            if state0 is None:
                raise TypeError(
                    "lyapunov_spectrum(problem, ...) requires n_steps."
                )
            n_steps = state0  # lyapunov_spectrum(problem, n_steps) form
        step_fn = problem.step_fn
        state0, dt = problem.state0, problem.dt
    else:
        step_fn = step_fn_or_problem
        if state0 is None or dt is None or n_steps is None:
            raise TypeError(
                "lyapunov_spectrum(step_fn, state0, dt, n_steps, ...) "
                "requires state0, dt, and n_steps when step_fn is a plain "
                "step function."
            )

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
        periodic renormalization and can't overflow (the ``renorm_every``
        overflow risk)."""
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
