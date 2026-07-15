"""Adaptive-step ODE integration via ``diffrax`` (M9, ``notes/milestones.md``).

ODE-only: ``diffrax`` has no DDE support
(`patrick-kidger/diffrax#406 <https://github.com/patrick-kidger/diffrax/issues/406>`_,
open since April 2024, re-checked 2026-07-07) -- ``lyapax.dde`` stays
fixed-step, see that module and ``notes/open_issues.md`` item 5.

Not imported by ``lyapax``'s top-level ``__init__``, since ``diffrax`` is an
optional dependency (the ``adaptive`` extra) -- import this module directly:
``from lyapax.adaptive import diffrax_adaptive_step``.

Design: ``diffrax_adaptive_step(...)`` returns a builder with the same
``(rhs, dt) -> step_fn`` signature as the fixed-step builders in
``lyapax.integrators`` (``rk4_step``, etc.), so it drops straight into
``lyapax.ode_problem(rhs, state0, dt, integrator=diffrax_adaptive_step(...))``
with no changes to ``lyapax.core`` -- ``core.py``'s ``_advance`` already
treats ``step_fn`` as an opaque ``state -> new_state`` map called once per
raw step and differentiated with ``jax.jvp``; whether that one call does a
single fixed-size update or an internal accept/reject/step-size-adaptation
loop is invisible to it. ``dt`` keeps its usual lyapax meaning here too --
the fixed sampling/renormalization interval ``lyapunov_spectrum`` advances by
per raw step -- and is *not* the adaptive integrator's own internal step
size, which varies within ``[t, t + dt]`` under ``rtol``/``atol`` control.

The internal accept/reject loop is a ``jax.lax.while_loop`` over diffrax's
low-level ``solver.step`` primitive (the same primitive ``diffrax.diffeqsolve``
is built on), mirroring ``diffrax._integrate.loop``'s own step/clip-to-boundary
structure closely enough to inherit its correctness, but scoped to a single
``dt`` advance with no ``SaveAt``/dense-output machinery. See
``tests/test_adaptive_diffrax.py`` for the isolated primitive check this is
built on, and two findings that shaped this implementation:

- **``scan_kind="lax"`` is required.** A diffrax Runge-Kutta solver's
  default construction (``scan_kind=None``, internally ``"checkpointed"``)
  iterates its stages via an ``equinox``-checkpointed ``while_loop`` wrapped
  in a ``custom_vjp`` that rejects forward-mode ``jax.jvp`` outright
  (``TypeError``). ``scan_kind="lax"`` uses a plain ``jax.lax.scan``
  instead, which is jvp-compatible. Enforced here, not left to the caller.
- **Reverse-mode differentiation (``jax.grad``/``jax.jacrev``) does not
  work end-to-end through this integrator**, even though the Lyapunov
  engine's own tangent propagation is forward-mode (``jax.jvp``, see
  ``lyapax.core``'s module docstring): the *outer* while_loop here has a
  data-dependent trip count (however many accept/reject iterations a given
  ``rtol``/``atol``/system needs), and JAX's reverse-mode AD cannot replay
  a ``lax.while_loop`` with a dynamic trip count backward at all -- a
  fundamental JAX limitation, not a diffrax quirk or a precision issue.
  Differentiating a Lyapunov exponent w.r.t. a handful of system parameters
  (``notes/open_issues.md`` item 6's goal) still works, but requires
  ``jax.jacfwd``/``jax.jvp`` (forward-mode over the parameter), not
  ``jax.grad``. Confirmed empirically: ``jax.grad`` raises
  ``ValueError: Reverse-mode differentiation does not work for
  lax.while_loop ... with dynamic start/stop values``; ``jax.jacfwd`` of the
  same function matches a central finite-difference reference to ~1e-8.

diffrax's own step-size controller (``PIDController``) already applies
``jax.lax.stop_gradient`` internally to its step-size-adjustment factor and
initial-step guess (see its source) -- the "stop_gradient on the step-size
decision" concern from the design note is therefore already handled by using
``PIDController``, not something this module adds on top.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

try:
    import diffrax
    import equinox as eqx
except ImportError:  # pragma: no cover - exercised via the ImportError path below
    diffrax = None
    eqx = None

RHS = Callable[[jnp.ndarray], jnp.ndarray]
Step = Callable[[jnp.ndarray], jnp.ndarray]


def _require_diffrax() -> None:
    if diffrax is None:
        raise ImportError(
            "adaptive ODE integration requires diffrax, which is not "
            "installed. Install it with the 'adaptive' extra "
            "(`pip install lyapax[adaptive]`) or directly "
            "(`pip install diffrax`)."
        )


def diffrax_adaptive_step(
        solver=None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        dt0: float | None = None,
        max_steps: int = 4096,
) -> Callable[[RHS, float], Step]:
    """
    Build a ``lyapax.ode_step``-compatible adaptive integrator:
    ``(rhs, dt) -> step_fn`` where ``step_fn(state) -> new_state`` advances
    by exactly ``dt`` using an adaptive-step diffrax solver internally.

    :param solver: a diffrax solver instance, e.g. ``diffrax.Dopri5()``,
        constructed with ``scan_kind="lax"``. Defaults to
        ``diffrax.Dopri5(scan_kind="lax")``. Raises ``ValueError`` if given
        a solver without ``scan_kind="lax"`` -- see the module docstring.
    :param rtol, atol: ``diffrax.PIDController`` step-size tolerances.
    :param dt0: initial internal step-size guess. Defaults to diffrax's own
        heuristic initial-step selection (``PIDController.init``).
    :param max_steps: safety cap on internal accept+reject iterations per
        ``dt`` advance, so a misbehaving ``rhs``/tolerance combination loops
        a bounded number of times inside ``jax.lax.while_loop`` rather than
        running away. Unlike ``diffrax.diffeqsolve`` (which returns
        whatever was reached when ``max_steps`` is hit unless
        ``throw=True``), exhausting ``max_steps`` here always raises via
        ``equinox.error_if`` -- returning a state from an earlier time than
        the requested ``t1`` while ``lyapunov_spectrum`` advances its
        elapsed-time denominator by the full ``dt`` regardless would
        silently mis-associate exponent with elapsed time, which
        ``check_finite=True`` cannot detect (a truncated step is still
        finite). ``equinox.error_if`` raises immediately in eager mode and
        under ``jax.jit``/``jax.vmap``; see its docstring for the
        ``EQX_ON_ERROR`` environment variable if you need to disable this
        check or substitute NaN instead. Raise ``max_steps`` or loosen
        ``rtol``/``atol`` if this triggers on a well-posed system.

    Usage
    -----
    >>> import lyapax
    >>> from lyapax.adaptive import diffrax_adaptive_step  # doctest: +SKIP
    >>> problem = lyapax.ode_problem(       # doctest: +SKIP
    ...     rhs, state0, dt=0.1, integrator=diffrax_adaptive_step(rtol=1e-8, atol=1e-10),
    ... )
    >>> result = lyapax.lyapunov_spectrum(problem, n_steps=...)  # doctest: +SKIP
    """
    _require_diffrax()
    if solver is None:
        solver = diffrax.Dopri5(scan_kind="lax")
    elif getattr(solver, "scan_kind", None) != "lax":
        raise ValueError(
            "diffrax solver must be constructed with scan_kind='lax' -- "
            "the default ('checkpointed') wraps its internal stage loop in "
            "a custom_vjp that rejects jax.jvp (see lyapax.adaptive's "
            "module docstring and tests/test_adaptive_diffrax.py)."
        )
    controller = diffrax.PIDController(rtol=rtol, atol=atol)

    def builder(rhs: RHS, dt: float) -> Step:
        term = diffrax.ODETerm(lambda t, y, args: rhs(y))
        error_order = solver.error_order(term)

        def step(state: jnp.ndarray) -> jnp.ndarray:
            t0 = jnp.asarray(0.0, dtype=state.dtype)
            t1 = jnp.asarray(dt, dtype=state.dtype)
            h0 = None if dt0 is None else jnp.asarray(dt0, dtype=state.dtype)

            tnext, controller_state0 = controller.init(
                term, t0, t1, state, h0, args=None, func=solver.func,
                error_order=error_order,
            )
            tnext = jnp.minimum(tnext, t1)
            solver_state0 = solver.init(term, t0, tnext, state, args=None)

            init_carry = (t0, tnext, state, solver_state0, controller_state0,
                          False, 0)

            def cond_fn(carry):
                tprev, _tnext, _y, _ss, _cs, _mj, n_steps = carry
                return (tprev < t1) & (n_steps < max_steps)

            def body_fn(carry):
                tprev, tnext, y, solver_state, controller_state, made_jump, n_steps = carry
                y1, y_error, _dense_info, new_solver_state, _result = solver.step(
                    term, tprev, tnext, y, args=None,
                    solver_state=solver_state, made_jump=made_jump,
                )
                (keep_step, new_tprev, new_tnext, new_made_jump,
                 new_controller_state, _ctrl_result) = controller.adapt_step_size(
                    tprev, tnext, y, y1, args=None, y_error=y_error,
                    error_order=error_order, controller_state=controller_state,
                )
                # Mirrors diffrax._integrate.loop's own clip-to-boundary
                # bookkeeping: clip the *next* proposed step, not the one
                # just taken, so the step just completed isn't itself
                # distorted by an artificial boundary truncation.
                new_tprev = jnp.minimum(new_tprev, t1)
                clip = new_tnext >= t1
                tclip = jnp.where(
                    keep_step, t1, new_tprev + 0.5 * (t1 - new_tprev))
                new_tnext = jnp.where(clip, tclip, new_tnext)

                y_next = jnp.where(keep_step, y1, y)
                solver_state_next = jax.tree_util.tree_map(
                    lambda a, b: jnp.where(keep_step, a, b),
                    new_solver_state, solver_state,
                )

                return (new_tprev, new_tnext, y_next, solver_state_next,
                        new_controller_state, new_made_jump, n_steps + 1)

            final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
            tprev_final, _tnext, y_final, _ss, _cs, _mj, n_steps_final = final_carry
            y_final = eqx.error_if(
                y_final,
                (tprev_final < t1) & (n_steps_final >= max_steps),
                "lyapax.adaptive.diffrax_adaptive_step: max_steps exhausted "
                "before completing this dt advance -- the returned state is "
                "from an earlier time than requested, which would silently "
                "corrupt the caller's elapsed-time accounting. Raise "
                "max_steps or loosen rtol/atol.",
            )
            return y_final

        return step

    # Duck-typed marker checked by lyapax.simulator.step.make_step_fn (the
    # DDE path) to reject this integrator with a clear error rather than a
    # bare arity-mismatch TypeError -- see that check and M9.3.
    builder._lyapax_adaptive_ode_only = True
    return builder
