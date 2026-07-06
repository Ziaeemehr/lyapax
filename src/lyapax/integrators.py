"""General-purpose fixed-step ODE integrators.

Independent of the vendored (Euler/Heun) integrators in
``lyapax.simulator.step`` — those are tied to the coupling/ring-buffer
plumbing for network models (M3+). These are plain ``state -> new_state``
maps for standalone benchmark systems (Lorenz, Rössler, ...), where RK4's
better accuracy at a given dt matters for getting a clean Lyapunov-exponent
estimate (see notes/validation_systems.md, Tier 2).
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


_ODE_INTEGRATORS: dict[str, Callable[[RHS, float], Step]] = {
    "euler": euler_step,
    "heun": heun_step,
    "rk4": rk4_step,
}


def get_integrator(name: str) -> Callable[[RHS, float], Step]:
    """Look up one of the built-in fixed-step ODE integrator builders by
    name (``"euler"``, ``"heun"``, or ``"rk4"``), each mapping
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
    :param integrator: one of ``"euler"``, ``"heun"``, ``"rk4"``, or a
        callable ``(rhs, dt) -> step_fn`` (e.g. ``rk4_step`` itself, or a
        user-supplied builder with the same signature).
    """
    builder = get_integrator(integrator) if isinstance(integrator, str) else integrator
    return builder(rhs, dt)
