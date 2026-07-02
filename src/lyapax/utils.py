"""Small helpers for demos/examples -- not part of the core LE engine."""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def simulate_trajectory(
        step_fn: Callable[[jnp.ndarray], jnp.ndarray],
        state0: jnp.ndarray,
        n_steps: int,
) -> jnp.ndarray:
    """Run ``step_fn`` for ``n_steps`` iterations via ``lax.scan`` (fast,
    compiled -- unlike a Python for-loop calling a JAX function repeatedly).

    Returns the trajectory including the initial state: shape
    ``(n_steps + 1, d)``.
    """
    def body(state, _):
        new_state = step_fn(state)
        return new_state, new_state

    _, traj = jax.lax.scan(body, state0, None, length=n_steps)
    return jnp.concatenate([state0[None, :], traj], axis=0)
