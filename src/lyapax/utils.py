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
        decimate: int = 1,
        n_transient: int = 0,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Run ``step_fn`` for ``n_steps`` iterations via ``lax.scan`` (fast,
    compiled -- unlike a Python for-loop calling a JAX function repeatedly).

    Returns the trajectory including the initial state: shape
    ``(n_steps // decimate + 1, d)`` (``(n_steps + 1, d)`` at the default
    ``decimate=1``). If ``dt`` is given, also returns the matching time
    axis as ``(t, traj)``, ``t[0] = 0`` and ``t[i] = i * decimate * dt``
    for ``i >= 1`` (``t = arange(n_steps + 1) * dt`` at ``decimate=1``) --
    ``dt`` is otherwise opaque to ``step_fn`` (baked into its closure), so
    this is the only place that can hand it back to the caller.

    ``n_transient`` : run this many steps first, discarding them (not
    emitted, not counted in ``n_steps``/the returned shape) -- "the
    initial state" the docstring above refers to becomes the state after
    the transient, not literally ``state0``, mirroring
    ``lyapax.core.LyapunovResult.trajectory``'s "transient is never
    saved" convention. Lets a replay (e.g.
    ``lyapax.sweep.sweep_trajectory``) land on the same post-transient
    state as a ``save_states=True`` Lyapunov run without a second,
    separately-tracked skip loop.

    ``decimate`` : keep every this many steps after the transient (must
    divide ``n_steps``). ``decimate=1`` (the default) keeps every step,
    reproducing this function's original behavior exactly.
    """
    if decimate < 1:
        raise ValueError(f"decimate must be >= 1; got {decimate}.")
    if n_steps % decimate != 0:
        raise ValueError(
            f"n_steps ({n_steps}) must be a multiple of decimate "
            f"({decimate})."
        )

    if n_transient:
        def _skip(state, _):
            return step_fn(state), None

        state0, _ = jax.lax.scan(_skip, state0, None, length=n_transient)

    def body(state, _):
        new_state = step_fn(state)
        return new_state, new_state

    if decimate == 1:
        _, traj = jax.lax.scan(body, state0, None, length=n_steps)
    else:
        def _chunk(state, _):
            state, _ = jax.lax.scan(body, state, None, length=decimate)
            return state, state

        _, traj = jax.lax.scan(_chunk, state0, None, length=n_steps // decimate)

    traj = jnp.concatenate([state0[None, :], traj], axis=0)
    if dt is None:
        return traj
    n_saves = n_steps // decimate
    t = jnp.concatenate([
        jnp.zeros((1,), dtype=traj.dtype),
        jnp.arange(1, n_saves + 1) * decimate * dt,
    ])
    return t, traj
