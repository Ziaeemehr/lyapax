"""Coupled, zero-delay networks — wires a dfun + a coupling callable
into the flat ``state -> new_state`` shape ``lyapax.core.lyapunov_spectrum``
expects.

Built on the vendored ring-buffer step (``lyapax.simulator.make_step_fn``,
``has_delays=False``) via a thin carry-to-flat adapter, not a second,
independent Euler/Heun implementation -- the adapter is the single place
that bridges the carry-based step to the flat map
``lyapax.core.lyapunov_spectrum`` wants, so there is only one copy of the
integrators to maintain.

Why this still doesn't need ``lyapax.dde``'s carry-aware tangent engine:
for ``has_delays=False`` the vendored step's ``buf`` never changes (it's a
structural no-op) and ``t`` is never read, so closing over a fixed
placeholder ``buf``/``t`` and only threading ``state`` through
``lyapunov_spectrum``'s ``jacfwd`` is exactly equivalent to differentiating
the whole carry -- there is no delay-buffer tangent information to lose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from .core import ODEProblem
from .coupling import CouplingFn
from .simulator import make_step_fn

StepFlat = Callable[[jnp.ndarray], jnp.ndarray]


StepFlatParametrized = Callable[[jnp.ndarray, dict], jnp.ndarray]


@dataclass(frozen=True)
class Network:
    """Topology for a coupled network: what M3+ call ``weights``/
    ``cvar_indices`` (and, for delayed networks, ``delay_steps``), gathered
    into one named object instead of separate positional arguments repeated
    across ``network_step``/``network_dde_problem`` call sites.

    :param weights: ``(n_nodes, n_nodes)``, ``weights[tgt, src]``.
    :param cvar_indices: indices into the state-variable axis selecting
        which rows of ``state`` feed the coupling.
    :param delay_steps: ``None`` for a zero-delay network, a number (``int``,
        or ``float`` when interpolating -- see
        ``lyapax.simulator.make_step_fn``) for a single global delay (in
        units of ``dt``), or an ``(n_nodes, n_nodes)`` integer matrix for
        per-edge delays. Only relevant to the DDE path
        (``network_dde_problem``); ignored by ``network_step``.
    """
    weights: jnp.ndarray
    cvar_indices: tuple[int, ...]
    delay_steps: int | float | jnp.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return self.weights.shape[0]

    @property
    def n_cvar(self) -> int:
        return len(self.cvar_indices)


def make_parametrized_network_step_fn(
        dfun: Callable,
        weights: jnp.ndarray,
        cvar_indices: tuple[int, ...],
        dt: float,
        coupling_fn: CouplingFn,
        integrator: str | Callable = "heun",
) -> StepFlatParametrized:
    """
    Same wiring as ``make_network_step_fn``, except ``params`` is a plain
    call-time argument, ``(state_flat, params) -> new_state_flat``, instead
    of being closed over at construction time -- for M6's ``jax.vmap``
    parameter sweeps (``lyapax.sweep.sweep_lyapunov_spectrum``), where
    ``params`` needs to be data the vmapped function can batch over, not a
    baked-in Python constant. Works because ``lyapax.simulator.make_step_fn``
    already threads ``params`` through the scanned carry rather than
    closing over it (see that function's docstring) -- this adapter was
    the only piece still closing params in early, so it's the only piece
    that needed to change.

    dfun, weights, cvar_indices, coupling_fn, integrator : see
        ``make_network_step_fn``.
    """
    n_nodes = weights.shape[0]
    n_cvar = len(cvar_indices)
    carry_step = make_step_fn(
        dfun=dfun, weights=weights, has_delays=False, horizon=1,
        n_nodes=n_nodes, cvar_indices=cvar_indices, dt=dt,
        coupling_fn=coupling_fn, integrator=integrator,
    )
    buf0 = jnp.zeros((1, n_cvar, n_nodes))

    def step_flat(state_flat: jnp.ndarray, params: dict) -> jnp.ndarray:
        n_sv = state_flat.shape[0] // n_nodes
        state = state_flat.reshape((n_sv, n_nodes))
        (new_state, _new_buf, _t, _params), _ = carry_step(
            (state, buf0, jnp.int32(0), params), None)
        return new_state.reshape(-1)

    return step_flat


def make_network_step_fn(
        dfun: Callable,
        weights: jnp.ndarray,
        cvar_indices: tuple[int, ...],
        params: dict,
        dt: float,
        coupling_fn: CouplingFn,
        integrator: str | Callable = "heun",
) -> StepFlat:
    """
    Build a flat ``state_flat (n_sv * n_nodes,) -> new_state_flat`` step
    function for use with ``lyapax.core.lyapunov_spectrum``.

    :param dfun: ``(state, coupling, params) -> dstate``, e.g. from
        ``lyapax.simulator.build_jax_dfun(model_spec)``. ``state`` and
        ``coupling`` are ``(n_sv, n_nodes)`` / ``(n_cvar, n_nodes)``.
    :param weights: ``(n_nodes, n_nodes)``, ``weights[tgt, src]``.
    :param cvar_indices: indices into the state-variable axis selecting which
        rows of ``state`` feed the coupling (``ModelSpec.cvar_indices``).
    :param params: closed over for this step function. To sweep over params via
        ``jax.vmap`` instead, use ``make_parametrized_network_step_fn`` +
        ``lyapax.sweep.sweep_lyapunov_spectrum`` (M6).
    :param coupling_fn: see ``lyapax.coupling`` — any callable with signature
        ``(cvar_state, weights, params) -> coupling`` works, including a
        user-defined one.
    :param integrator: ``"euler"``, ``"heun"``, ``"rk4"``, ``"rk6"``, or a callable
        ``(state, dfun, coupling, dt, params) -> new_state`` -- see
        ``lyapax.simulator.make_step_fn``.
    """
    step_flat_p = make_parametrized_network_step_fn(
        dfun, weights, cvar_indices, dt, coupling_fn, integrator)
    return lambda state_flat: step_flat_p(state_flat, params)


def network_step(
        dfun: Callable,
        network: Network,
        coupling: CouplingFn,
        params: dict,
        dt: float,
        integrator: str | Callable = "heun",
) -> StepFlat:
    """
    Build a flat ``state_flat -> new_state_flat`` step for a zero-delay
    coupled network, naming the four concepts ``make_network_step_fn``
    otherwise mixes into one long argument list separately: model dynamics
    (``dfun``), topology (``network``), the coupling rule (``coupling``),
    and the numerical method (``integrator``).

    A thin wrapper -- it calls straight through to ``make_network_step_fn``,
    reading ``weights``/``cvar_indices`` off ``network``. Prefer this over
    ``make_network_step_fn`` in new code; the latter remains for direct,
    lower-level use.

    :param dfun: ``(state, coupling, params) -> dstate``, e.g. from
        ``lyapax.simulator.build_jax_dfun(model_spec)``.
    :param network: topology, see ``Network``.
    :param coupling: any ``lyapax.coupling``-style callable,
        ``(cvar_state, weights, params) -> coupling``.
    :param params: closed over for this step function.
    :param dt: fixed step size.
    :param integrator: ``"euler"``, ``"heun"``, ``"rk4"``, ``"rk6"``, or a callable --
        see ``lyapax.simulator.make_step_fn``.
    """
    return make_network_step_fn(
        dfun=dfun, weights=network.weights, cvar_indices=network.cvar_indices,
        params=params, dt=dt, coupling_fn=coupling, integrator=integrator,
    )


def network_step_parametrized(
        dfun: Callable,
        network: Network,
        coupling: CouplingFn,
        dt: float,
        integrator: str | Callable = "heun",
) -> StepFlatParametrized:
    """``network_step``'s counterpart for ``jax.vmap`` parameter sweeps --
    see ``make_parametrized_network_step_fn``, which this wraps."""
    return make_parametrized_network_step_fn(
        dfun=dfun, weights=network.weights, cvar_indices=network.cvar_indices,
        dt=dt, coupling_fn=coupling, integrator=integrator,
    )


def network_problem(
        dfun: Callable,
        network: Network,
        coupling: CouplingFn,
        params: dict,
        state0: jnp.ndarray,
        dt: float,
        integrator: str | Callable = "heun",
) -> ODEProblem:
    """
    Build an ``ODEProblem`` for a zero-delay coupled network -- bundles
    ``network_step``'s ``step_fn`` together with ``state0`` and ``dt``, the
    same ``lyapunov_spectrum(problem, n_steps=...)`` recipe ``ode_problem``
    gives the plain-ODE case (see ``ODEProblem``).

    :param dfun: ``(state, coupling, params) -> dstate``.
    :param network: topology, see ``Network``.
    :param coupling: any ``lyapax.coupling``-style callable,
        ``(cvar_state, weights, params) -> coupling``.
    :param params: closed over for this step function.
    :param state0: initial state, either already flat
        (``n_sv * n_nodes,``) or in the network's natural ``(n_sv,
        n_nodes)`` shape -- flattened here either way, since
        ``network_step``'s step function operates on flat state.
    :param dt: fixed step size.
    :param integrator: ``"euler"``, ``"heun"``, ``"rk4"``, ``"rk6"``, or a callable --
        see ``lyapax.simulator.make_step_fn``.
    """
    step_fn = network_step(dfun, network, coupling, params, dt, integrator=integrator)
    state0 = jnp.asarray(state0).reshape(-1)
    return ODEProblem(step_fn=step_fn, state0=state0, dt=dt)
