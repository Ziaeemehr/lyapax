"""Small helpers for demos/examples -- not part of the core LE engine."""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def simulate_trajectory(
        step_fn: Callable[[jnp.ndarray], jnp.ndarray],
        state0: jnp.ndarray,
        n_steps: int,
        dt: float | None = None,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Run ``step_fn`` for ``n_steps`` iterations via ``lax.scan`` (fast,
    compiled -- unlike a Python for-loop calling a JAX function repeatedly).

    Returns the trajectory including the initial state: shape
    ``(n_steps + 1, d)``. If ``dt`` is given, also returns the matching
    time axis as ``(t, traj)``, ``t = arange(n_steps + 1) * dt`` -- ``dt``
    is otherwise opaque to ``step_fn`` (baked into its closure), so this is
    the only place that can hand it back to the caller.
    """
    def body(state, _):
        new_state = step_fn(state)
        return new_state, new_state

    _, traj = jax.lax.scan(body, state0, None, length=n_steps)
    traj = jnp.concatenate([state0[None, :], traj], axis=0)
    if dt is None:
        return traj
    return jnp.arange(n_steps + 1) * dt, traj
