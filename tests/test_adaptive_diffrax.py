"""M9.2's isolated diffrax-primitive check, run before any Benettin-loop
wiring: does ``jax.jvp`` propagate a correct tangent through diffrax's
low-level ``solver.step`` (the primitive ``diffeqsolve`` is built on,
and the atomic unit M9's adaptive integrator needs to wrap per raw step)?

Answer, found empirically here: not with a solver's default construction.
``AbstractRungeKutta.step`` iterates its stages via an internal
``equinox``-checkpointed ``while_loop`` (``scan_kind=None`` -> "checkpointed"),
which is wrapped in a ``custom_vjp`` that explicitly rejects forward-mode
autodiff (``TypeError: can't apply forward-mode autodiff (jvp) to a
custom_vjp function``). Constructing the solver with ``scan_kind="lax"``
switches that internal loop to a plain ``jax.lax.scan``, which *does*
support ``jvp`` -- confirmed against the exact tangent action of a linear
system (``d/dy0[expm(A*dt) @ y0] = expm(A*dt)``). This is the concrete
reason M9's adaptive-ODE wrapper must construct diffrax solvers with
``scan_kind="lax"`` explicitly, not rely on diffrax's default.
"""
import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm

diffrax = pytest.importorskip("diffrax")

A = jnp.array([[-0.5, 1.0], [-1.0, -0.5]])
DT = 0.1


def _rhs(t, y, args):
    return A @ y


def _one_step(solver, term, y0):
    y1, _y_err, _dense_info, _solver_state, _result = solver.step(
        term, 0.0, DT, y0, args=None,
        solver_state=solver.init(term, 0.0, DT, y0, args=None),
        made_jump=False,
    )
    return y1


def test_default_scan_kind_rejects_jvp():
    """Documents the surprise: default construction is jvp-incompatible.

    If a future diffrax release changes this, this test starts failing
    (not erroring), which is exactly the signal that M9.2's scan_kind="lax"
    workaround may no longer be necessary.
    """
    term = diffrax.ODETerm(_rhs)
    solver = diffrax.Dopri5()
    y0 = jnp.array([1.0, 0.3])
    v = jnp.array([0.4, -1.2])
    with pytest.raises(TypeError, match="forward-mode autodiff"):
        jax.jvp(lambda y: _one_step(solver, term, y), (y0,), (v,))


def test_scan_kind_lax_jvp_matches_exact_linear_tangent():
    term = diffrax.ODETerm(_rhs)
    solver = diffrax.Dopri5(scan_kind="lax")
    y0 = jnp.array([1.0, 0.3])
    v = jnp.array([0.4, -1.2])

    y1, tangent = jax.jvp(lambda y: _one_step(solver, term, y), (y0,), (v,))

    exact_fundamental = expm(A * DT)
    exact_y1 = exact_fundamental @ y0
    exact_tangent = exact_fundamental @ v

    # Dopri5's own truncation error at this dt, not machine precision --
    # this is a single fixed step of an order-5 method, so O(dt^6)-ish.
    assert jnp.max(jnp.abs(y1 - exact_y1)) < 1e-8
    assert jnp.max(jnp.abs(tangent - exact_tangent)) < 1e-8
