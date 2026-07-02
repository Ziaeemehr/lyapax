"""M3: coupled, zero-delay networks — wires a dfun + a coupling callable
into the flat ``state -> new_state`` shape ``lyapax.core.lyapunov_spectrum``
expects.

Not the vendored ring-buffer ``step(carry, _)`` from
``lyapax.vendored.step`` — that shape carries a delay buffer needed from
M4 onward. For a zero-delay network the buffer is dead weight, and
``lyapax.core`` wants a plain ``state -> new_state`` map over a flat
vector, so this module reshapes to/from ``(n_sv, n_nodes)`` around a
coupling + integrator step instead of routing through the carry tuple.
M4 will need to reconcile this with the vendored ring-buffer step (most
likely by giving that one a ``coupling_fn`` parameter too, replacing its
hardcoded linear-only branch — tracked in notes/milestones.md).
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from .coupling import CouplingFn

StepFlat = Callable[[jnp.ndarray], jnp.ndarray]


def _heun(state, dfun, coupling, dt, params):
    k1 = dfun(state, coupling, params)
    k2 = dfun(state + dt * k1, coupling, params)
    return state + 0.5 * dt * (k1 + k2)


def _euler(state, dfun, coupling, dt, params):
    return state + dt * dfun(state, coupling, params)


def make_network_step_fn(
        dfun: Callable,
        weights: jnp.ndarray,
        cvar_indices: tuple[int, ...],
        params: dict,
        dt: float,
        coupling_fn: CouplingFn,
        use_heun: bool = True,
) -> StepFlat:
    """
    Build a flat ``state_flat (n_sv * n_nodes,) -> new_state_flat`` step
    function for use with ``lyapax.core.lyapunov_spectrum``.

    dfun : ``(state, coupling, params) -> dstate``, e.g. from
        ``lyapax.vendored.build_jax_dfun(model_spec)``. ``state`` and
        ``coupling`` are ``(n_sv, n_nodes)`` / ``(n_cvar, n_nodes)``.
    weights : (n_nodes, n_nodes), ``weights[tgt, src]``.
    cvar_indices : indices into the state-variable axis selecting which
        rows of ``state`` feed the coupling (``ModelSpec.cvar_indices``).
    params : closed over for this step function; not swept here (M6).
    coupling_fn : see ``lyapax.coupling`` — any callable with signature
        ``(cvar_state, weights, params) -> coupling`` works, including a
        user-defined one.
    """
    n_nodes = weights.shape[0]
    cvar_idx = jnp.array(cvar_indices, dtype=jnp.int32)
    integrate = _heun if use_heun else _euler

    def step_flat(state_flat: jnp.ndarray) -> jnp.ndarray:
        n_sv = state_flat.shape[0] // n_nodes
        state = state_flat.reshape((n_sv, n_nodes))
        cvar_state = state[cvar_idx]
        coupling = coupling_fn(cvar_state, weights, params)
        new_state = integrate(state, dfun, coupling, dt, params)
        return new_state.reshape(-1)

    return step_flat
