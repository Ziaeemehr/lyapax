"""Benchmark systems for validating the Lyapunov engine.

Plain flat-vector JAX functions, deliberately independent of the vendored
ModelSpec/coupling machinery in ``lyapax.vendored`` — these are single-node,
uncoupled textbook systems, so routing them through the string-expression
codegen built for networked models would only add indirection. Each entry
here corresponds to a tier in notes/validation_systems.md; see that doc for
the reference Lyapunov values and citations.
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

RHS = Callable[[jnp.ndarray], jnp.ndarray]
Step = Callable[[jnp.ndarray], jnp.ndarray]


# ---------------------------------------------------------------------------
# Tier 0 — exact analytic references
# ---------------------------------------------------------------------------

def linear_system(A: jnp.ndarray) -> RHS:
    """rhs(x) = A @ x. Exact Lyapunov spectrum = Re(eigvals(A)) (Tier 0.1)."""
    A = jnp.asarray(A)

    def rhs(x: jnp.ndarray) -> jnp.ndarray:
        return A @ x
    return rhs


def logistic_map(r: float) -> Step:
    """x_{n+1} = r x (1-x). At r=4, exact LE = ln(2) (Tier 0.2)."""
    def step(x: jnp.ndarray) -> jnp.ndarray:
        return r * x * (1.0 - x)
    return step


def tent_map() -> Step:
    """Exact LE = ln(2) (Tier 0.2)."""
    def step(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x < 0.5, 2.0 * x, 2.0 * (1.0 - x))
    return step


def henon_map(a: float = 1.4, b: float = 0.3) -> Step:
    """Constant Jacobian determinant -b => sum(LE) = ln|b| exactly (Tier 0.3)."""
    def step(state: jnp.ndarray) -> jnp.ndarray:
        x, y = state[0], state[1]
        return jnp.array([1.0 - a * x * x + y, b * x])
    return step


# ---------------------------------------------------------------------------
# Tier 1/2 — chaotic flows (structural invariants + published values)
# ---------------------------------------------------------------------------

def lorenz(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> RHS:
    """Constant trace(J) = -(sigma+1+beta) => sum(LE) exactly known (Tier 1.1);
    lambda1 has a well-documented published value (Tier 2)."""
    def rhs(state: jnp.ndarray) -> jnp.ndarray:
        x, y, z = state[0], state[1], state[2]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return jnp.array([dx, dy, dz])
    return rhs


def rossler(a: float = 0.2, b: float = 0.2, c: float = 5.7) -> RHS:
    """trace(J) = a + x - c (state-dependent) => sum(LE) = a - c + <x>_t
    (Tier 1.2); lambda1 has a published order-of-magnitude value (Tier 2)."""
    def rhs(state: jnp.ndarray) -> jnp.ndarray:
        x, y, z = state[0], state[1], state[2]
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return jnp.array([dx, dy, dz])
    return rhs
