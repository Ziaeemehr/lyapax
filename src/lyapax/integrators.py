"""General-purpose fixed-step ODE integrators.

Independent of the vendored (Euler/Heun) integrators in
``lyapax.vendored.step`` — those are tied to the coupling/ring-buffer
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


def rk4_step(rhs: RHS, dt: float) -> Step:
    """Classic fixed-step 4th-order Runge-Kutta."""
    def step(state: jnp.ndarray) -> jnp.ndarray:
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return step
