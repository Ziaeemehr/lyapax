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

**Differentiating an exponent w.r.t. a system parameter** (``notes/
open_issues.md`` item 6.1): ``lyapunov_spectrum`` is built entirely from
``jax.lax.scan`` (static trip count) and ``jnp.linalg.qr``, both of which
support reverse-mode AD, so ``jax.grad``/``jax.jacrev`` do not raise here
(unlike ``lyapax.adaptive``'s diffrax-backed integrator, whose internal
dynamic-trip-count ``while_loop`` blocks reverse-mode outright). But *not
raising* is not the same as *useful*: for a genuinely chaotic trajectory,
``step_fn`` closes over the parameter being differentiated, so
``jax.grad``/``jax.jacfwd`` differentiate through the entire unrolled state
trajectory, and the result inherits that trajectory's own exponential
sensitivity to perturbation — the returned "gradient" grows roughly like
``exp(lambda_max * horizon)`` and is numerically meaningless well before it
overflows (confirmed empirically: unusable already within a few hundred
steps for the Lorenz system, worse for longer runs; see
``tests/test_differentiability.py``). This is a known phenomenon in
chaotic sensitivity analysis, not a lyapax-specific bug — naive
trajectory-unrolling gradients of long-time-averaged chaotic quantities are
fundamentally unreliable, which is why shadowing-based methods (e.g.
least-squares shadowing) exist in that literature. **Practical guidance:**
``jax.grad``/``jax.jacfwd`` through this engine are reliable for
non-chaotic or short-horizon systems (e.g. tuning a parameter that keeps
the trajectory on a stable fixed point/limit cycle, or a genuinely short
run) — do not trust a gradient computed through a long chaotic trajectory
without independently checking it (e.g. against a finite-difference
estimate) first.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple

import jax
import jax.numpy as jnp

from .integrators import ode_step

if TYPE_CHECKING:
    from .dde import DDECheckpoint

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


class LyapunovCheckpoint(NamedTuple):
    state: jnp.ndarray
    """(d,) trajectory state at the end of the checkpointed run."""

    Y: jnp.ndarray
    """(d, k) QR-orthonormalized tangent basis at the end of the
    checkpointed run, in its raw (not display-sorted) column order -- see
    ``cum_log_growth``."""

    cum_log_growth: jnp.ndarray
    """(k,) total accumulated log-growth (sum of ``log|diag(R)|`` over
    every renormalization block since accumulation started) in the same raw
    column order as ``Y`` -- *not* ``LyapunovResult.history``'s column
    order, which is re-sorted by the final row on every call and can
    reorder differently between two otherwise-identical resumed runs if
    exponents are near-degenerate. Keeping this in ``Y``'s raw order is
    what makes resuming exact: it is the same "never reordered until the
    very end" bookkeeping ``lyapunov_spectrum`` already does internally
    within a single call, just carried across two calls instead of within
    one."""

    elapsed_time: jnp.ndarray
    """Scalar: total elapsed accumulation time since the end of the
    transient, across every resumed call so far -- same units as
    ``LyapunovResult.times``."""


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

    checkpoint: LyapunovCheckpoint | DDECheckpoint | None = None
    """Enough state to continue this run with another
    ``lyapunov_spectrum(..., resume=result.checkpoint)`` /
    ``lyapunov_spectrum_dde(..., resume=result.checkpoint)`` call, picking
    up exactly where this one left off (no re-transient, no discontinuity
    in ``history``/``times``) -- the "run a fixed n_steps, eyeball
    ``history``/``convergence_drift``, resume if not converged yet"
    workflow. Always set by both ``lyapunov_spectrum`` (the ODE engine,
    ``LyapunovCheckpoint``) and ``lyapax.dde.lyapunov_spectrum_dde`` (the
    DDE engine, ``lyapax.dde.DDECheckpoint`` -- also carries the delay ring
    buffer's state, since a DDE's Markovian state is ``(state, buf)``
    together, not ``state`` alone)."""


def _run_renorm_scan(
        renorm_block, carry0, n_renorm: int, renorm_every: int, dt: float,
        cum_log_growth0: jnp.ndarray | float = 0.0,
        elapsed_time0: jnp.ndarray | float = 0.0,
):
    """Shared tail between ``lyapunov_spectrum`` (ODE) and
    ``lyapax.dde.lyapunov_spectrum_dde``: scan ``renorm_block`` (which must
    return ``(new_carry, log_growth (k,))`` per call, log-growth already
    ``log|diag(R)|`` from one QR-renormalization block) ``n_renorm`` times,
    then turn the per-block log-growth into cumulative running exponent
    estimates. Not part of the public API -- the only thing that differs
    between the ODE and DDE engines is what one renorm block's tangent
    propagation looks like, not this bookkeeping.

    ``cum_log_growth0``/``elapsed_time0`` offset the cumulative sum/elapsed
    time before turning them into ``history``/``times`` -- zero (the
    default) for a fresh run, or a previous call's raw-order final
    cumulative log-growth/elapsed time when resuming (``lyapunov_spectrum``'s
    ``resume=`` path). Returns ``(final_carry, final_cum_log_growth,
    result)`` -- the extra two (over just ``result``) are what
    ``lyapunov_spectrum`` needs to build the next ``LyapunovCheckpoint``;
    ``lyapunov_spectrum_dde`` ignores them (no resume support yet).
    """
    final_carry, log_growth_per_block = jax.lax.scan(
        renorm_block, carry0, None, length=n_renorm)

    # (n_renorm, k), raw (not display-sorted) column order
    cum_log_growth = jnp.cumsum(log_growth_per_block, axis=0) + cum_log_growth0
    block_times = elapsed_time0 + (jnp.arange(1, n_renorm + 1) * renorm_every) * dt
    history = cum_log_growth / block_times[:, None]

    order = jnp.argsort(-history[-1])
    history = history[:, order]
    exponents = history[-1]

    result = LyapunovResult(exponents=exponents, history=history, times=block_times)
    return final_carry, cum_log_growth[-1], result


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


class ConvergenceDrift(NamedTuple):
    absolute: jnp.ndarray
    """(k,) ``|history[-1] - history[-1 - window_rows]|`` per exponent --
    the raw change in the running estimate over the tail window."""

    relative: jnp.ndarray
    """(k,) ``absolute / |history[-1]|`` per exponent. Blows up for
    exponents near zero (e.g. a conservative system's zero exponent, or any
    near-degenerate pair) -- a near-zero reference makes "relative to it"
    not meaningful; use ``absolute`` for those columns instead."""

    converged: jnp.ndarray | None
    """(k,) bool, ``relative <= tol`` per exponent -- ``None`` if ``tol``
    was not given to ``convergence_drift``."""


def convergence_drift(
        result: LyapunovResult, window: float = 0.1, tol: float | None = None,
) -> ConvergenceDrift:
    """
    Summarize how much each exponent's running estimate has moved over the
    tail of the run, by comparing ``result.history[-1]`` (the final
    estimate) to the estimate ``window`` fraction of the run's
    renormalization points earlier.

    ``lyapunov_spectrum``/``lyapunov_spectrum_dde`` always run a fixed
    ``n_steps`` -- there is no built-in stopping criterion, adaptive or
    otherwise. This is a diagnostic to help a caller judge, after the fact,
    whether that fixed-length run was long enough: a large drift means the
    estimate was probably still moving and ``n_steps`` should be increased;
    a small, stable drift is evidence (not proof) of convergence. Pairs
    with ``result.checkpoint``/``resume=``: if not converged, continue the
    same run with ``lyapunov_spectrum(..., resume=result.checkpoint)``
    instead of restarting from scratch.

    :param result: output of ``lyapunov_spectrum``/``lyapunov_spectrum_dde``.
    :param window: fraction, in ``(0, 1]``, of ``result.history``'s rows
        making up the tail comparison window. Default ``0.1`` compares the
        final estimate against the estimate from 10% of the run ago.
    :param tol: if given, ``converged`` is ``relative <= tol`` per
        exponent; otherwise ``converged`` is ``None``.

    :return: ``ConvergenceDrift(absolute, relative, converged)``.

    Usage
    -----
    >>> import jax
    >>> jax.config.update("jax_enable_x64", True)
    >>> import jax.numpy as jnp
    >>> from lyapax import lyapunov_spectrum, ode_problem, systems
    >>> from lyapax.core import convergence_drift
    >>> rhs = systems.lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    >>> problem = ode_problem(rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=1e-2)
    >>> result = lyapunov_spectrum(
    ...     problem, n_steps=50_000, renorm_every=10, t_transient=100.0,
    ... )
    >>> drift = convergence_drift(result, window=0.1, tol=1e-2)
    >>> bool(drift.converged[0])  # doctest: +SKIP
    True
    """
    if not (0.0 < window <= 1.0):
        raise ValueError(f"window must be in (0, 1]; got {window}.")
    n_renorm = result.history.shape[0]
    if n_renorm < 2:
        raise ValueError(
            "convergence_drift needs at least 2 renormalization points in "
            f"result.history to measure drift over; got {n_renorm}. Use a "
            "smaller renorm_every or a larger n_steps."
        )
    window_rows = min(max(1, round(window * n_renorm)), n_renorm - 1)

    latest = result.history[-1]
    reference = result.history[-1 - window_rows]
    absolute = jnp.abs(latest - reference)
    relative = absolute / jnp.abs(latest)
    converged = None if tol is None else relative <= tol
    return ConvergenceDrift(absolute=absolute, relative=relative, converged=converged)


def kaplan_yorke_dimension(
        exponents: jnp.ndarray, d_total: int | None = None,
) -> float:
    """
    Kaplan-Yorke (Lyapunov) dimension: an estimate of an attractor's
    fractal dimension from its Lyapunov spectrum (Kaplan & Yorke, 1979),
    ``j + sum(exponents[:j]) / |exponents[j]|`` where ``j`` is the largest
    prefix length with a non-negative partial sum. Pure post-processing of
    ``LyapunovResult.exponents`` -- no tangent-propagation or QR involved.

    :param exponents: ``(k,)`` exponents, sorted descending -- exactly
        ``LyapunovResult.exponents``'s own order, so
        ``kaplan_yorke_dimension(result.exponents)`` is the usual call.
    :param d_total: the full system dimension, if ``exponents`` is only a
        *partial* spectrum (``k < d_total``). If given, and the partial
        sum never goes negative within the tracked ``k`` exponents, raises
        ``ValueError`` instead of silently returning ``k`` -- the true
        crossing point lies beyond what was tracked, so ``k`` would
        understate the answer; track more exponents (increase ``k``)
        instead. Omit (the default) when ``exponents`` is already the full
        spectrum -- the "sum stays non-negative" case then correctly
        returns ``len(exponents)`` (the attractor fills the whole tracked
        phase space, e.g. a conservative system).

    :return: the Kaplan-Yorke dimension, in ``[0, len(exponents)]``.

    Usage
    -----
    >>> import jax.numpy as jnp
    >>> from lyapax.core import kaplan_yorke_dimension
    >>> float(kaplan_yorke_dimension(jnp.array([0.906, 0.0, -14.57])))  # Lorenz
    2.0621...
    """
    exponents = jnp.asarray(exponents)
    n = exponents.shape[0]
    cumsum = jnp.cumsum(exponents)
    j = 0
    for i in range(1, n + 1):
        if float(cumsum[i - 1]) >= 0.0:
            j = i
        else:
            break
    if j == n:
        if d_total is not None and d_total > n:
            raise ValueError(
                f"kaplan_yorke_dimension: the cumulative sum of all {n} "
                "tracked exponents never goes negative, so the "
                f"Kaplan-Yorke crossing point lies beyond the tracked "
                f"partial spectrum (d_total={d_total} > k={n}) -- track "
                "more exponents (a larger k) to find it. Omit d_total "
                "only when `exponents` is already the full spectrum."
            )
        return float(j)
    if j == 0:
        return 0.0
    return j + float(cumsum[j - 1]) / abs(float(exponents[j]))


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
        resume: LyapunovCheckpoint | None = None,
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
        starting the Lyapunov accumulation. Mutually exclusive with
        ``resume`` (raises ``ValueError`` if both are given and nonzero) --
        a resumed run is already transient-aligned.
    seed : PRNG seed for the initial random tangent-vector basis. Unused
        when ``resume`` is given (the tangent basis comes from the
        checkpoint instead).
    check_finite : if True, raise ``FloatingPointError`` when any running
        estimate in ``history`` is non-finite (a QR diagonal underflowed to
        0 or overflowed to inf, e.g. from ``renorm_every`` too large).
        Off by default and only usable when this function is called
        eagerly (not wrapped in ``jax.jit``).
    resume : a previous call's ``result.checkpoint``, to continue that run
        instead of starting a fresh one -- skips the random tangent-basis
        init and the transient, and ``history``/``times`` in the returned
        ``LyapunovResult`` continue the cumulative running estimate (not
        reset to zero), so concatenating two calls' ``history`` reads as
        one continuous convergence curve. ``k`` must match the checkpoint's
        tracked dimension if given explicitly (defaults to it otherwise).
        See :ref:`16_convergence_drift.py
        <sphx_glr_auto_examples_16_convergence_drift.py>` for the
        run-inspect-resume workflow this enables.

    Returns
    -------
    LyapunovResult

    Examples
    --------
    >>> import jax
    >>> jax.config.update("jax_enable_x64", True)
    >>> import jax.numpy as jnp
    >>> from lyapax import lyapunov_spectrum, ode_problem, systems
    >>> rhs = systems.lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    >>> problem = ode_problem(rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=1e-2)
    >>> result = lyapunov_spectrum(
    ...     problem, n_steps=50_000, renorm_every=10, t_transient=100.0,
    ... )
    >>> result.exponents  # doctest: +SKIP
    Array([ 0.906,  0.   , -14.57], dtype=float64)
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
    if resume is not None:
        if t_transient > 0.0:
            raise ValueError(
                "t_transient and resume are mutually exclusive -- a "
                "resumed run continues from an already-transient-aligned "
                "checkpoint; pass t_transient=0.0 (the default) when "
                "resume is given."
            )
        if resume.state.shape[0] != d:
            raise ValueError(
                f"resume.state has dimension {resume.state.shape[0]}, but "
                f"state0 has dimension {d} -- resume must come from a "
                "checkpoint of a run on the same system."
            )
        resume_k = resume.Y.shape[1]
        if k is None:
            k = resume_k
        elif k != resume_k:
            raise ValueError(
                f"k={k} does not match resume.Y's tracked dimension "
                f"({resume_k}) -- resuming must track the same number of "
                "exponents as the checkpointed run."
            )
    elif k is None:
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

    if resume is not None:
        state0, Y0 = resume.state, resume.Y
    else:
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

    if t_transient > 0.0 and resume is None:
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
    cum_log_growth0 = resume.cum_log_growth if resume is not None else 0.0
    elapsed_time0 = resume.elapsed_time if resume is not None else 0.0
    final_carry, final_cum_log_growth, result = _run_renorm_scan(
        _renorm_block, (state0, Y0), n_renorm, renorm_every, dt,
        cum_log_growth0=cum_log_growth0, elapsed_time0=elapsed_time0,
    )
    final_state, final_Y = final_carry
    checkpoint = LyapunovCheckpoint(
        state=final_state, Y=final_Y,
        cum_log_growth=final_cum_log_growth, elapsed_time=result.times[-1],
    )
    result = result._replace(checkpoint=checkpoint)
    if check_finite:
        _check_finite(result)
    return result
