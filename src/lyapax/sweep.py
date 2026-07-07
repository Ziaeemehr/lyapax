"""Batched parameter/initial-condition sweeps via ``jax.vmap``.

Mirrors vbi's ``JaxSweeper`` pattern (batch a simulation over a grid of
parameter values) without reusing its code, per the vendoring decision.
The surface needed here is much smaller than a
general sweeper: ``lyapax.simulator.make_step_fn``'s carry already threads
``params`` through as data rather than closing over it (see that function's
docstring), so a parameter
sweep is just ``jax.vmap`` applied to a function that calls
``lyapax.core.lyapunov_spectrum`` once, with ``params`` (and optionally
``state0``) as the vmapped argument. No new tangent-propagation or
QR machinery -- this is a batching wrapper around the existing engine, not
a third one.

The one piece of the network path that still closed over ``params`` at
construction time was ``lyapax.network.make_network_step_fn`` (a thin
adapter, not the tangent engine); ``make_parametrized_network_step_fn``
(in that module) is the call-time-``params`` sibling this needs.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .core import LyapunovResult, lyapunov_spectrum

StepFlatParametrized = Callable[[jnp.ndarray, dict], jnp.ndarray]


def sweep_lyapunov_spectrum(
        step_fn: StepFlatParametrized,
        state0: jnp.ndarray,
        params_batch: dict,
        dt: float,
        n_steps: int,
        k: int | None = None,
        renorm_every: int = 1,
        t_transient: float = 0.0,
        seed: int = 0,
        state0_batch: jnp.ndarray | None = None,
) -> LyapunovResult:
    """
    Batched ``lyapunov_spectrum`` over a grid of parameter values (and,
    optionally, a matching grid of initial conditions) via one ``jax.vmap``
    call, rather than a Python loop making one call per grid point (as in
    e.g. ``examples/05_kuramoto_sync.py``/``09_kuramoto_delayed_network.py``).

    step_fn : ``(state, params) -> new_state`` -- ``params`` must be a
        call-time argument, *not* closed over. See
        ``lyapax.network.make_parametrized_network_step_fn`` for the
        network-coupling case; a hand-written function with this signature
        works too (same "plain callable, no registry" extension point as
        ``lyapax.coupling``).
    state0 : (d,) initial state, shared across the whole sweep, unless
        ``state0_batch`` is given.
    params_batch : pytree matching one ``params`` dict's structure, but
        every leaf carries an extra leading batch axis of size
        ``n_sweep`` -- e.g. ``{"G": jnp.linspace(0, 4, 13), "omega":
        jnp.broadcast_to(omega, (13, n_nodes))}`` to sweep ``G`` at fixed
        ``omega``.
    dt, n_steps, k, renorm_every, t_transient, seed : see
        ``lyapax.core.lyapunov_spectrum`` — shared (not swept) across the
        whole grid.
    state0_batch : optional ``(n_sweep, d)`` initial conditions, one row
        per grid point, if the initial condition should vary alongside
        ``params_batch`` rather than staying fixed at ``state0``.

    Returns
    -------
    LyapunovResult, every field with an extra leading batch axis of size
    ``n_sweep`` -- e.g. ``result.exponents.shape == (n_sweep, k)``.
    """
    def _run_one(params, state0_i):
        return lyapunov_spectrum(
            lambda state: step_fn(state, params), state0_i, dt, n_steps,
            k=k, renorm_every=renorm_every, t_transient=t_transient, seed=seed,
        )

    if state0_batch is None:
        return jax.vmap(_run_one, in_axes=(0, None))(params_batch, state0)
    return jax.vmap(_run_one, in_axes=(0, 0))(params_batch, state0_batch)
