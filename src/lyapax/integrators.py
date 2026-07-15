"""General-purpose fixed-step ODE integrators.

Independent of the vendored (Euler/Heun) integrators in
``lyapax.simulator.step`` - those are tied to the coupling/ring-buffer
plumbing for network models (M3+). These are plain ``state -> new_state``
maps for standalone benchmark systems (Lorenz, Rössler, ...), where RK4's
better accuracy at a given dt matters for getting a clean Lyapunov-exponent
estimate (see :ref:`Tier 2 <validation-tier-2>` of the validation guide).
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

RHS = Callable[[jnp.ndarray], jnp.ndarray]
Step = Callable[[jnp.ndarray], jnp.ndarray]


def euler_step(rhs: RHS, dt: float) -> Step:
    def step(state: jnp.ndarray) -> jnp.ndarray:
        return state + dt * rhs(state)
    return step


def heun_step(rhs: RHS, dt: float) -> Step:
    """Fixed-step 2nd-order Heun (explicit trapezoidal)."""
    def step(state: jnp.ndarray) -> jnp.ndarray:
        k1 = rhs(state)
        k2 = rhs(state + dt * k1)
        return state + 0.5 * dt * (k1 + k2)
    return step


def rk4_step(rhs: RHS, dt: float) -> Step:
    """Classic fixed-step 4th-order Runge-Kutta."""
    def step(state: jnp.ndarray) -> jnp.ndarray:
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return step


RK6_STAGE_C = (0.0, 3 / 50, 1439 / 15000, 1439 / 10000, 4973 / 10000, 389 / 400, 1999 / 2000, 1.0)
"""Butcher ``c_i`` (fractional position within the step) for each of
``rk6_combine``'s 8 stages -- verified via each stage's row-sum
consistency condition (``sum_j a_ij == c_i``) against the source tableau,
see ``rk6_combine``'s docstring. Exposed so a coupled/delayed caller can
recompute a state- or time-dependent input (e.g. network coupling, or a
DDE's delayed history lookup) at the *correct* intra-step point for each
stage, instead of freezing it at the step's start -- see
:ref:`dde-rk-stage-order`."""


def rk6_combine(
        f: Callable[[jnp.ndarray, float], jnp.ndarray],
        state: jnp.ndarray,
        dt: float,
) -> jnp.ndarray:
    """One fixed-step 6th-order Runge-Kutta update, ``state -> new_state``,
    against a derivative evaluator ``f(y, c) -> dy``, where ``c`` is that
    stage's Butcher node (``RK6_STAGE_C``) -- the fractional position
    within the step, ``0`` at the first stage, ``1`` at the last. Purely
    autonomous callers (no coupling, no explicit time-dependence) can
    ignore ``c``; ``lyapax.simulator.step``'s coupled-network RK6 uses it
    to recompute coupling (or a delayed history lookup) fresh at each
    stage's own intra-step state/time, rather than freezing it once per
    step (see :ref:`dde-rk-stage-order`) -- this and
    ``rk6_step`` share the same tableau instead of duplicating ~30 lines
    of coefficients twice.

    The 8-stage tableau is the ``a_ij`` coefficients (stages 1-8) of
    Verner's "efficient order 6(5)" embedded pair -- the coefficients
    underlying ``Vern6`` in Julia's OrdinaryDiffEq.jl -- with the 6th-order
    solution taken as the FSAL point these stages build towards (``c=1``,
    i.e. what would be stage 9's *argument*), rather than a separately
    weighted sum over 9 evaluated stages: an earlier draft of this
    function used the ``b1..b9`` weights given as comments in the Julia
    source directly, which turned out to be a red herring -- symbolically
    expanding that combination's stability function against ``e^z`` (exact
    rational arithmetic, ``state' = z*state``) showed it matches only
    through ``z^5``, i.e. order 5, not 6. The 8-stage FSAL combination
    below was verified the same way and matches ``e^z`` through ``z^6``
    (mismatching first at ``z^7``, the expected order-6 signature), and
    separately confirmed with an empirical convergence-order check against
    a genuinely nonlinear system with a known exact solution -- see
    ``tests/test_integrators.py``.
    """
    c = RK6_STAGE_C
    k1 = f(state, c[0])
    k2 = f(state + dt * (3 / 50 * k1), c[1])
    k3 = f(state + dt * (519479 / 27000000 * k1 + 2070721 / 27000000 * k2), c[2])
    k4 = f(state + dt * (1439 / 40000 * k1 + 4317 / 40000 * k3), c[3])
    k5 = f(state + dt * (
        109225017611 / 82828840000 * k1
        - 417627820623 / 82828840000 * k3
        + 43699198143 / 10353605000 * k4
    ), c[4])
    k6 = f(state + dt * (
        -8036815292643907349452552172369 / 191934985946683241245914401600 * k1
        + 246134619571490020064824665 / 1543816496655405117602368 * k3
        - 13880495956885686234074067279 / 113663489566254201783474344 * k4
        + 755005057777788994734129 / 136485922925633667082436 * k5
    ), c[5])
    k7 = f(state + dt * (
        -1663299841566102097180506666498880934230261
        / 30558424506156170307020957791311384232000 * k1
        + 130838124195285491799043628811093033 / 631862949514135618861563657970240 * k3
        - 3287100453856023634160618787153901962873 / 20724314915376755629135711026851409200 * k4
        + 2771826790140332140865242520369241 / 396438716042723436917079980147600 * k5
        - 1799166916139193 / 96743806114007800 * k6
    ), c[6])
    k8 = f(state + dt * (
        -832144750039369683895428386437986853923637763
        / 15222974550069600748763651844667619945204887 * k1
        + 818622075710363565982285196611368750 / 3936576237903728151856072395343129 * k3
        - 9818985165491658464841194581385463434793741875
        / 61642597962658994069869370923196463581866011 * k4
        + 31796692141848558720425711042548134769375 / 4530254033500045975557858016006308628092 * k5
        - 14064542118843830075 / 766928748264306853644 * k6
        - 1424670304836288125 / 2782839104764768088217 * k7
    ), c[7])
    return state + dt * (
        382735282417 / 11129397249634 * k1
        + 5535620703125000 / 21434089949505429 * k4
        + 13867056347656250 / 32943296570459319 * k5
        + 626271188750 / 142160006043 * k6
        - 51160788125000 / 289890548217 * k7
        + 163193540017 / 946795234 * k8
    )


def rk6_step(rhs: RHS, dt: float) -> Step:
    """Fixed-step 6th-order Runge-Kutta, for when RK4's O(dt^4) local error
    is the accuracy bottleneck (e.g. wanting a much larger dt at the same
    error, or a much smaller error at the same dt, than RK4 gives). See
    ``rk6_combine`` for the tableau, provenance, and correctness checks.
    """
    def step(state: jnp.ndarray) -> jnp.ndarray:
        return rk6_combine(lambda y, _c: rhs(y), state, dt)
    return step


_ODE_INTEGRATORS: dict[str, Callable[[RHS, float], Step]] = {
    "euler": euler_step,
    "heun": heun_step,
    "rk4": rk4_step,
    "rk6": rk6_step,
}


def get_integrator(name: str) -> Callable[[RHS, float], Step]:
    """Look up one of the built-in fixed-step ODE integrator builders by
    name (``"euler"``, ``"heun"``, ``"rk4"``, or ``"rk6"``), each mapping
    ``(rhs, dt) -> step_fn``.
    """
    try:
        return _ODE_INTEGRATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown integrator {name!r}; choose one of "
            f"{sorted(_ODE_INTEGRATORS)}, or pass a callable "
            "(rhs, dt) -> step_fn directly."
        ) from None


def ode_step(
        rhs: RHS,
        dt: float,
        integrator: str | Callable[[RHS, float], Step] = "rk4",
) -> Step:
    """
    Build a single-step ``state -> new_state`` map for a plain (uncoupled)
    ODE, with the integration method as an explicit, swappable argument.

    :param rhs: right-hand side, ``state (d,) -> dstate (d,)``.
    :param dt: fixed step size.
    :param integrator: one of ``"euler"``, ``"heun"``, ``"rk4"``, ``"rk6"``,
        or a callable ``(rhs, dt) -> step_fn`` (e.g. ``rk4_step`` itself, or
        a user-supplied builder with the same signature).
    """
    builder = get_integrator(integrator) if isinstance(integrator, str) else integrator
    return builder(rhs, dt)
