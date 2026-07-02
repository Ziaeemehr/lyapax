"""M3: coupled, zero-delay networks — wires a dfun + a coupling callable
into the flat ``state -> new_state`` shape ``lyapax.core.lyapunov_spectrum``
expects.

Built on the vendored ring-buffer step (``lyapax.simulator.make_step_fn``,
``has_delays=False``) via a thin carry-to-flat adapter, not a second,
independent Euler/Heun implementation. M3 originally built its own (this
module used to have local ``_euler``/``_heun`` copies, byte-for-byte
identical to the vendored ones) because at the time there was no way to
feed a carry-based step into anything -- ``lyapax.core.lyapunov_spectrum``
wants a flat ``state -> new_state`` map, and M4's carry-based Lyapunov
engine (``lyapax.dde.lyapunov_spectrum_dde``) didn't exist yet. Unified
here (M5 cleanup, see notes/milestones.md) now that there's a real adapter
to write instead of a second copy of the integrators to maintain.

Why this still doesn't need ``lyapax.dde``'s carry-aware tangent engine:
for ``has_delays=False`` the vendored step's ``buf`` never changes (it's a
structural no-op) and ``t`` is never read, so closing over a fixed
placeholder ``buf``/``t`` and only threading ``state`` through
``lyapunov_spectrum``'s ``jacfwd`` is exactly equivalent to differentiating
the whole carry -- there is no delay-buffer tangent information to lose.
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from .coupling import CouplingFn
from .simulator import make_step_fn

StepFlat = Callable[[jnp.ndarray], jnp.ndarray]


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
        ``lyapax.simulator.build_jax_dfun(model_spec)``. ``state`` and
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
    n_cvar = len(cvar_indices)
    carry_step = make_step_fn(
        dfun=dfun, weights=weights, has_delays=False, horizon=1,
        n_nodes=n_nodes, cvar_indices=cvar_indices, dt=dt,
        coupling_fn=coupling_fn, use_heun=use_heun,
    )
    buf0 = jnp.zeros((1, n_cvar, n_nodes))

    def step_flat(state_flat: jnp.ndarray) -> jnp.ndarray:
        n_sv = state_flat.shape[0] // n_nodes
        state = state_flat.reshape((n_sv, n_nodes))
        (new_state, _new_buf, _t, _params), _ = carry_step(
            (state, buf0, jnp.int32(0), params), None)
        return new_state.reshape(-1)

    return step_flat
