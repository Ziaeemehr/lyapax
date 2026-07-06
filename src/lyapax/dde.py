"""Fixed-delay DDE Lyapunov engine, on top of the vendored ring buffer (M4).

Reuses the vendored ring-buffer simulator (``lyapax.simulator.step``) as-is
rather than a second, parallel history mechanism: the genuinely missing
piece was never "how do we store delayed history" (the vendored
``_write_ring``/``_read_delayed_coupling`` already do that correctly), it
was that ``lyapax.core.lyapunov_spectrum`` only differentiates a flat
``state``, not the ``(state, buf)`` carry a DDE actually needs -- a
sensitivity to the delayed value (``d f/d x(t-tau)``) is just as real as
the sensitivity to the current value, and dropping it silently gives wrong
exponents, not merely imprecise ones.

Method, following Farmer's (1982) approach to DDE Lyapunov spectra (the
same method the established ``jitcdde_lyap`` package implements for
adaptive-step DDEs via Hermite-interpolated history): augment the system
with explicit tangent dynamics for both ``state`` and ``buf``, propagate
them alongside the primal trajectory, periodically re-orthonormalize via
QR. Where this differs from ``jitcdde_lyap``: our scope is fixed-step with
an integer-step delay, so a delayed value always lands exactly on a stored
grid point -- no interpolation needed, so a plain finite-size ring buffer
suffices and the tangent state is already finite-dimensional (a plain
``jnp.linalg.qr``, not jitcdde's continuous-function inner product over a
Hermite interpolant).

Tangent propagation is ``jax.jvp``-based, not ``jax.jacfwd``-based like
``lyapax.core``: cost is O(k) forward passes per raw step (k = tracked
exponents), not O(d_total) for the full augmented ``(state, buf)``
dimension -- the earlier M4 attempt (state-augmentation fed through a
dense jacfwd) didn't scale to real coupled networks, since d_total grows
with both network size and ring-buffer depth. See notes/milestones.md
(M4) for the full design discussion and a design-review pass that caught
a subtle bug in an earlier draft of this engine (closing over the
ring-buffer step counter ``t`` instead of threading it through the scan
carry -- see the comment on ``t`` below).

Scope note: ``lyapunov_spectrum_dde`` itself has no opinion on delay
structure -- it differentiates through whatever carry ``step_fn`` produces,
so it already works correctly with a genuine per-edge (heterogeneous)
delay matrix via the vendored step's legacy, hardcoded-linear coupling
path (``coupling_fn=None``, ``delay_steps=<(n_nodes,n_nodes) matrix>``,
verified directly against an asymmetric 2-node case). The real scope limit
lives in ``lyapax.simulator.make_step_fn``, not here: a *custom*
``coupling_fn`` (``lyapax.coupling``'s plain-callable style, e.g. for a
delayed sigmoidal/Kuramoto network) is currently only wired up for
zero-delay and *uniform*-delay (single global ``tau_steps``) branches --
combining a custom ``coupling_fn`` with per-edge delays needs an
edge-aware ``coupling_fn`` signature, a separate design fork left for M5.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .core import LyapunovResult, _check_finite, _run_renorm_scan, _warn_if_not_x64
from .network import Network
from .simulator import make_step_fn

CarryStepFn = Callable
"""step(carry, _) -> (new_carry, new_state), carry = (state, buf, t, params)
-- exactly what lyapax.simulator.make_step_fn(...) returns."""

DelayedRHS = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
"""rhs_delayed(state_now, state_delayed) -> dstate, both shape (m,) --
m=1 for a scalar DDE (e.g. Mackey-Glass), m>1 for a small non-networked
vector DDE. See make_scalar_delayed_step_fn."""


def resolve_tau_steps(tau: float, dt: float, warn_tol: float = 1e-6) -> int:
    """Round a physical delay ``tau`` to an integer number of ``dt`` steps.

    Integer-step only, no sub-step interpolation -- see risk #4 in
    notes/milestones.md for the accuracy tradeoff this implies (characterize
    it with a convergence-vs-dt test at fixed physical ``tau``). Every
    downstream Lyapunov spectrum is computed for the *rounded* delay
    ``tau_eff(tau_steps, dt)``, not the exact ``tau`` passed in -- use
    ``tau_eff`` to see what delay was actually used, and decrease ``dt``
    to shrink the gap.

    warn_tol : relative tolerance (of ``tau``) above which a mismatch
        between ``tau`` and the rounded ``tau_eff`` triggers a warning.
    """
    tau_steps = max(1, round(tau / dt))
    if tau > 0 and abs(tau_steps * dt - tau) / tau > warn_tol:
        warnings.warn(
            f"resolve_tau_steps: requested tau={tau} rounds to "
            f"tau_steps={tau_steps} (tau_eff={tau_steps * dt}) at dt={dt} "
            "-- the Lyapunov spectrum will be computed for tau_eff, not "
            "tau. Decrease dt to shrink this gap.",
            stacklevel=2,
        )
    return tau_steps


def tau_eff(tau_steps: int, dt: float) -> float:
    """The effective (rounded-to-grid) delay actually used by a DDE run
    built with integer ``tau_steps``, in the same units as ``dt``. See
    ``resolve_tau_steps``."""
    return tau_steps * dt


def constant_history_buf0(cvar_state0: jnp.ndarray, horizon: int) -> jnp.ndarray:
    """Initial ring buffer under the constant-history DDE convention: the
    coupling-variable state is assumed equal to ``cvar_state0`` for all
    ``t <= 0``.

    :param cvar_state0: ``(n_cvar, n_nodes)`` coupling-variable state at t=0.
    :returns: ``(horizon, n_cvar, n_nodes)`` buf0, ready for
        ``carry0 = (state0, buf0, jnp.int32(0), params)``.
    """
    cvar_state0 = jnp.asarray(cvar_state0)
    return jnp.tile(cvar_state0[None, :, :], (horizon, 1, 1))


def make_scalar_delayed_step_fn(
        rhs_delayed: DelayedRHS,
        m: int,
        tau_steps: int,
        dt: float,
        integrator: str | Callable = "heun",
) -> CarryStepFn:
    """
    Build a carry-based step function for a simple, non-networked
    fixed-delay DDE directly from a plain ``rhs_delayed`` -- no
    ``ModelSpec``/``coupling_fn`` ceremony needed. Mirrors
    ``lyapax.integrators.rk4_step``'s role on the ODE side: a lightweight
    front door for the "just one system, no network" case, sitting next to
    ``lyapax.simulator.make_step_fn`` (the general, ``ModelSpec``-based
    front door used for real delayed networks, e.g.
    ``tests/test_dde.py``'s benchmark test) -- both build a step for the
    *same* underlying vendored ring-buffer simulator, so there is still
    only one DDE mechanism, just two ways to construct a step for it (see
    notes/milestones.md, M4, on why forcing every DDE through the network
    machinery -- even a single scalar equation -- was a real usability gap
    the ODE side never had).

    Internally: a trivial "coupled to your own delayed history" 1-node
    self-loop, with an identity coupling_fn (the delayed state passes
    through unchanged; any scaling/nonlinearity belongs in ``rhs_delayed``
    itself, not a coupling formula -- there's no real network here).

    :param rhs_delayed: ``(state_now, state_delayed) -> dstate``, both shape
        ``(m,)`` -- see ``DelayedRHS``.
    :param m: state dimension (e.g. 1 for a scalar DDE like Mackey-Glass).
    :param tau_steps: integer delay in units of ``dt`` (see
        ``resolve_tau_steps``).
    :param integrator: ``"euler"``, ``"heun"``, ``"rk4"``, or a callable --
        see ``lyapax.simulator.make_step_fn``.
    """
    def dfun(state, coupling, params):
        return rhs_delayed(state[:, 0], coupling[:, 0])[:, None]

    def _identity_coupling(cvar_state, weights, params):
        return cvar_state

    weights = jnp.ones((1, 1))
    return make_step_fn(
        dfun=dfun, weights=weights, has_delays=True, horizon=tau_steps + 1,
        n_nodes=1, cvar_indices=tuple(range(m)), dt=dt,
        coupling_fn=_identity_coupling, tau_steps=tau_steps, integrator=integrator,
    )


def scalar_delayed_history0(state0_now: jnp.ndarray, tau_steps: int):
    """(state0, buf0) for make_scalar_delayed_step_fn, from a flat ``(m,)``
    initial condition, under the constant-history DDE convention -- the
    scalar-case counterpart to constant_history_buf0, hiding the
    ``(n_sv, n_nodes)``/``(horizon, n_cvar, n_nodes)`` reshaping the
    general (networked) path needs.
    """
    state0_now = jnp.asarray(state0_now)
    state0 = state0_now[:, None]  # (m, 1)
    buf0 = constant_history_buf0(state0, tau_steps + 1)  # (horizon, m, 1)
    return state0, buf0


@dataclass(frozen=True)
class DDEProblem:
    """Owns the carry-style plumbing (``step_fn``, ring buffer, delay
    length) that ``lyapunov_spectrum_dde`` otherwise asks the caller to
    assemble and pass by hand -- ``buf0``, ``tau_steps``, and the carry
    layout become properties of one object instead of four separate
    positional arguments. Build one with ``dde_problem`` (single/scalar
    DDE) or ``network_dde_problem`` (coupled, uniformly-delayed network),
    or construct directly if you already have a carry ``step_fn``.

    :param step_fn: see ``lyapunov_spectrum_dde``.
    :param state0: ``(n_sv, n_nodes)`` initial state (pre-transient).
    :param buf0: ``(horizon, n_cvar, n_nodes)`` initial ring buffer.
    :param params: closed over for tangent propagation.
    :param dt: fixed step size.
    :param tau_steps: integer delay in units of ``dt`` (see
        ``resolve_tau_steps``) -- carried for reference/inspection
        (``problem.tau_steps``, ``tau_eff(problem.tau_steps, problem.dt)``);
        ``lyapunov_spectrum_dde`` itself only needs ``buf0``'s shape.
    """
    step_fn: CarryStepFn
    state0: jnp.ndarray
    buf0: jnp.ndarray
    params: dict
    dt: float
    tau_steps: int


def dde_problem(
        rhs_delayed: DelayedRHS,
        state0: jnp.ndarray,
        tau: float,
        dt: float,
        history: jnp.ndarray | None = None,
        integrator: str | Callable = "heun",
) -> DDEProblem:
    """
    Build a ``DDEProblem`` for a simple, non-networked scalar/vector
    fixed-delay DDE directly from ``rhs_delayed`` -- the parallel-to-ODE
    front door for ``lyapunov_spectrum_dde``, hiding the ring-buffer/carry
    construction that ``make_scalar_delayed_step_fn`` +
    ``scalar_delayed_history0`` otherwise ask the caller to wire up by hand.

    :param rhs_delayed: ``(state_now, state_delayed) -> dstate``, both shape
        ``(m,)`` -- see ``DelayedRHS``.
    :param state0: ``(m,)`` initial state (pre-transient, pre-history).
    :param tau: physical delay, rounded to an integer number of ``dt``
        steps -- see ``resolve_tau_steps``.
    :param dt: fixed step size.
    :param history: optional ``(tau_steps + 1, m)`` initial ring buffer,
        for a non-constant history. Defaults to the constant-history
        convention (``state0`` held for all ``t <= 0``), see
        ``scalar_delayed_history0``.
    :param integrator: ``"euler"``, ``"heun"``, ``"rk4"``, or a callable --
        see ``lyapax.simulator.make_step_fn``.
    """
    state0 = jnp.asarray(state0)
    m = state0.shape[0]
    tau_steps = resolve_tau_steps(tau, dt)
    step_fn = make_scalar_delayed_step_fn(rhs_delayed, m, tau_steps, dt, integrator=integrator)
    state0_2d, default_buf0 = scalar_delayed_history0(state0, tau_steps)
    buf0 = jnp.asarray(history) if history is not None else default_buf0
    return DDEProblem(
        step_fn=step_fn, state0=state0_2d, buf0=buf0, params={}, dt=dt,
        tau_steps=tau_steps,
    )


def network_dde_problem(
        dfun: Callable,
        network: Network,
        coupling: Callable,
        params: dict,
        state0: jnp.ndarray,
        dt: float,
        tau: float | None = None,
        history: jnp.ndarray | None = None,
        integrator: str | Callable = "heun",
) -> DDEProblem:
    """
    Build a ``DDEProblem`` for a coupled, uniformly-delayed network -- the
    DDE counterpart of ``lyapax.network.network_step`` (see that function
    and ``Network`` for the topology/coupling/integrator split this keeps
    parallel between the ODE and DDE construction paths).

    Only a single global delay is supported (a uniform ``tau_steps`` read
    via one ring-buffer lookup per node) -- per-edge heterogeneous delays
    combined with a custom ``coupling`` callable are not yet supported, see
    ``lyapax.simulator.make_step_fn``'s docstring.

    :param dfun: ``(state, coupling, params) -> dstate``.
    :param network: topology; exactly one of ``tau`` or an int
        ``network.delay_steps`` must give the uniform delay.
    :param coupling: ``(cvar_state, weights, params) -> coupling``.
    :param params: closed over for tangent propagation.
    :param state0: ``(n_sv, n_nodes)`` initial state.
    :param dt: fixed step size.
    :param tau: physical delay, rounded via ``resolve_tau_steps``. Mutually
        exclusive with an int ``network.delay_steps``.
    :param history: optional ``(tau_steps + 1, n_cvar, n_nodes)`` initial
        ring buffer. Defaults to the constant-history convention, see
        ``constant_history_buf0``.
    :param integrator: ``"euler"``, ``"heun"``, ``"rk4"``, or a callable --
        see ``lyapax.simulator.make_step_fn``.
    """
    if tau is not None:
        if network.delay_steps is not None:
            raise ValueError(
                "network_dde_problem: pass either tau or network.delay_steps, "
                "not both."
            )
        tau_steps = resolve_tau_steps(tau, dt)
    elif isinstance(network.delay_steps, int):
        tau_steps = network.delay_steps
    else:
        raise ValueError(
            "network_dde_problem needs a single global delay: pass tau=... "
            "or set network.delay_steps to an int (uniform tau_steps); a "
            "per-edge delay matrix is not supported with a custom coupling."
        )

    state0 = jnp.asarray(state0)
    horizon = tau_steps + 1
    step_fn = make_step_fn(
        dfun=dfun, weights=network.weights, has_delays=True, horizon=horizon,
        n_nodes=network.n_nodes, cvar_indices=network.cvar_indices, dt=dt,
        coupling_fn=coupling, tau_steps=tau_steps, integrator=integrator,
    )
    if history is not None:
        buf0 = jnp.asarray(history)
    else:
        cvar_idx = jnp.array(network.cvar_indices, dtype=jnp.int32)
        buf0 = constant_history_buf0(state0[cvar_idx], horizon)
    return DDEProblem(
        step_fn=step_fn, state0=state0, buf0=buf0, params=params, dt=dt,
        tau_steps=tau_steps,
    )


def lyapunov_spectrum_dde(
        step_fn_or_problem: CarryStepFn | DDEProblem,
        state0: jnp.ndarray | int | None = None,
        buf0: jnp.ndarray | None = None,
        params: dict | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        k: int | None = None,
        renorm_every: int = 1,
        t_transient: float = 0.0,
        seed: int = 0,
        check_finite: bool = False,
) -> LyapunovResult:
    """
    Compute the (partial or full) Lyapunov spectrum of a fixed-delay DDE,
    via the Benettin/QR method generalized to a ``(state, buf)`` tangent
    pair (see module docstring).

    Parameters
    ----------
    step_fn_or_problem : either a ``DDEProblem`` (from ``dde_problem`` /
        ``network_dde_problem``) -- in which case ``state0``, ``buf0``,
        ``params``, and ``dt`` are read off it and the second positional
        argument is ``n_steps`` (``lyapunov_spectrum_dde(problem, n_steps)``
        or ``lyapunov_spectrum_dde(problem, n_steps=...)``) -- or a plain
        carry ``step_fn``, ``step(carry, _) -> (new_carry, new_state)``,
        carry = ``(state, buf, t, params)``, in which case ``state0``,
        ``buf0``, ``params``, ``dt``, ``n_steps`` are all given explicitly
        (the original, lower-level call form).
    state0 : (n_sv, n_nodes) initial state (pre-transient). Ignored (and
        may be omitted) when a ``DDEProblem`` is passed.
    buf0 : (horizon, n_cvar, n_nodes) initial ring buffer (see
        ``constant_history_buf0``). Ignored when a ``DDEProblem`` is passed.
    params : closed over for tangent propagation (not differentiated --
        only ``state``/``buf`` are). Ignored when a ``DDEProblem`` is passed.
    dt, n_steps, renorm_every, seed : see ``lyapax.core.lyapunov_spectrum``
        (same meaning).
    k : number of leading exponents to track. Defaults to the *full*
        augmented spectrum, ``k = d_total = state0.size + buf0.size`` --
        note this is the ring-buffer-augmented dimension, not just the
        physical state dimension, and grows with ``horizon`` (delay
        length). Per-step cost and tangent-matrix memory are ``O(k)``, and
        QR is ``O(d_total * k^2)``, so for large ``horizon`` (long delays
        or many network nodes) pass an explicit small ``k`` rather than
        relying on the full-spectrum default.
    t_transient : time to integrate (discarding tangent tracking) before
        starting the Lyapunov accumulation. Internally rounded up to cover
        at least one full ring cycle (``horizon * dt``): until every buffer
        slot has been written at least once from real dynamics, delayed-
        direction tangent information is incomplete, and (for large
        ``horizon``) a random initial tangent basis is disproportionately
        concentrated in the buffer's trivial "just shifted forward
        unchanged" directions -- the same class of silent-bias risk M1's
        transient fix addresses for the plain ODE case, amplified here.
    check_finite : see ``lyapax.core.lyapunov_spectrum`` (same meaning);
        the augmented ``(state, buf)`` tangent dimension makes underflow
        more likely for large ``horizon``, so this is worth enabling while
        tuning ``renorm_every`` for a new DDE system.

    Returns
    -------
    LyapunovResult
    """
    if isinstance(step_fn_or_problem, DDEProblem):
        problem = step_fn_or_problem
        if n_steps is None:
            if state0 is None:
                raise TypeError(
                    "lyapunov_spectrum_dde(problem, ...) requires n_steps."
                )
            n_steps = state0  # lyapunov_spectrum_dde(problem, n_steps) form
        step_fn = problem.step_fn
        state0, buf0, params, dt = problem.state0, problem.buf0, problem.params, problem.dt
    else:
        step_fn = step_fn_or_problem
        if n_steps is None or state0 is None or buf0 is None or params is None or dt is None:
            raise TypeError(
                "lyapunov_spectrum_dde(step_fn, state0, buf0, params, dt, "
                "n_steps, ...) requires all of state0, buf0, params, dt, "
                "n_steps when step_fn is a plain carry step function."
            )

    state0 = jnp.asarray(state0)
    buf0 = jnp.asarray(buf0)
    _warn_if_not_x64(state0)
    horizon = buf0.shape[0]
    state_shape = state0.shape
    buf_shape = buf0.shape
    d_state = state0.size
    d_buf = buf0.size
    d_total = d_state + d_buf

    if k is None:
        k = d_total
    if not (1 <= k <= d_total):
        raise ValueError(f"k must be in [1, {d_total}]; got {k}.")
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
    Y0_flat, _ = jnp.linalg.qr(jax.random.normal(key, (d_total, k), dtype=state0.dtype))
    Y_state0 = Y0_flat[:d_state].reshape(state_shape + (k,))
    Y_buf0 = Y0_flat[d_state:].reshape(buf_shape + (k,))

    def _flatten(Y_state, Y_buf):
        return jnp.concatenate(
            [Y_state.reshape(d_state, k), Y_buf.reshape(d_buf, k)], axis=0)

    def _unflatten(Y_flat):
        return (Y_flat[:d_state].reshape(state_shape + (k,)),
                Y_flat[d_state:].reshape(buf_shape + (k,)))

    def _advance(state, buf, t, Y_state, Y_buf, n_substeps):
        """Propagate (state, buf) and their tangent pair jointly for
        n_substeps raw steps, then QR once -- t lives in the scanned carry
        and is re-read/incremented every raw step (not closed over once per
        block): closing over a stale t was verified to silently diverge
        from the correct answer during design review (no crash/NaN, just a
        wrong number), since t drives the ring-buffer's modular read/write
        indices and must match what step_fn itself sees at each raw step."""
        def _raw_step(carry_inner, _):
            state_i, buf_i, t_i, Y_state_i, Y_buf_i = carry_inner

            def f(state, buf):
                (new_state, new_buf, _t2, _params2), _ = step_fn(
                    (state, buf, t_i, params), None)
                return new_state, new_buf

            def _single_column(Y_state_col, Y_buf_col):
                return jax.jvp(f, (state_i, buf_i), (Y_state_col, Y_buf_col))

            (new_state_rep, new_buf_rep), (dY_state, dY_buf) = jax.vmap(
                _single_column, in_axes=(-1, -1), out_axes=((0, 0), (-1, -1))
            )(Y_state_i, Y_buf_i)
            new_state, new_buf = new_state_rep[0], new_buf_rep[0]

            return (new_state, new_buf, t_i + 1, dY_state, dY_buf), None

        (state, buf, t, Y_state, Y_buf), _ = jax.lax.scan(
            _raw_step, (state, buf, t, Y_state, Y_buf), None, length=n_substeps)

        Q_flat, R = jnp.linalg.qr(_flatten(Y_state, Y_buf))
        Q_state, Q_buf = _unflatten(Q_flat)
        return state, buf, t, Q_state, Q_buf, R

    t0 = jnp.int32(0)

    # Unconditional (unlike core.py's ODE transient, which is skippable via
    # t_transient=0.0): a DDE always needs at least one full ring cycle
    # (horizon * dt) before every buffer slot has been written from real
    # dynamics -- see the t_transient docstring above. Floored, not
    # skipped, even when the caller passes t_transient=0.0 exactly.
    min_transient = horizon * dt
    n_transient = renorm_every * max(
        1, round(max(t_transient, min_transient) / dt / renorm_every))

    def _transient_block(carry, _):
        state, buf, t, Y_state, Y_buf = carry
        state, buf, t, Y_state, Y_buf, _R = _advance(
            state, buf, t, Y_state, Y_buf, renorm_every)
        return (state, buf, t, Y_state, Y_buf), None

    (state0, buf0, t0, Y_state0, Y_buf0), _ = jax.lax.scan(
        _transient_block, (state0, buf0, t0, Y_state0, Y_buf0), None,
        length=n_transient // renorm_every)

    def _renorm_block(carry, _):
        state, buf, t, Y_state, Y_buf = carry
        state, buf, t, Y_state, Y_buf, R = _advance(
            state, buf, t, Y_state, Y_buf, renorm_every)
        log_growth = jnp.log(jnp.abs(jnp.diag(R)))
        return (state, buf, t, Y_state, Y_buf), log_growth

    n_renorm = n_steps // renorm_every
    result = _run_renorm_scan(
        _renorm_block, (state0, buf0, t0, Y_state0, Y_buf0), n_renorm, renorm_every, dt)
    if check_finite:
        _check_finite(result)
    return result
