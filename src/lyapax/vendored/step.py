"""dfun codegen + the ring-buffer step(carry, _) factory.

Adapted from vbi/simulator/backend/jax_/codegen.py and
vbi/simulator/backend/jax_/simulator.py — see NOTICE.md in this directory
for provenance and what was dropped (stochastic noise, state clipping,
sigmoidal/kuramoto coupling, monitors/stimuli).

This is the piece that unifies ODE and DDE: ``has_delays=False`` reads
coupling from the instantaneous state; ``has_delays=True`` reads it from a
ring buffer of past coupling-variable states via a flat-index gather. Both
paths produce the same ``step(carry, _) -> (new_carry, new_state)`` shape
consumed by ``jax.lax.scan``, and — critically for the Lyapunov engine in
M1/M4 — both are plain, differentiable JAX functions of ``carry``.
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from .model_spec import ModelSpec


def build_jax_dfun(spec: ModelSpec) -> Callable:
    """
    Compile dfun_str expressions into a JAX-traceable function.

    Generated signature:
        fn(state, coupling, params) -> jnp.ndarray  shape (n_sv, n_nodes)

    where:
        state    : (n_sv, n_nodes)
        coupling : (n_cvar, n_nodes) - one row per coupling variable
        params   : dict[str, scalar | jnp.ndarray]

    Compiled once via exec() on the spec's own strings, same as the vbi
    original — nothing about the generated function blocks jax.jacfwd /
    jax.jvp.
    """
    sv = spec.sv_names
    param_names = spec.param_names
    cvar_names = spec.cvar

    lines = [
        "import jax.numpy as _jnp",
        "from jax.numpy import pi, exp, log, sin, cos, tanh, sqrt",
        "def _dfun_jax(state, coupling, params):",
    ]
    for i, name in enumerate(sv):
        lines.append(f"    {name} = state[{i}]")
    for name in param_names:
        lines.append(f"    {name} = params['{name}']")
    for i, cname in enumerate(cvar_names):
        lines.append(f"    c_{cname} = coupling[{i}]")
    lines.append("    c = coupling[0] if coupling.shape[0] > 0 else 0.0")
    sv_exprs = ", ".join(f"({spec.dfun_str[name]})" for name in sv)
    lines.append(f"    return _jnp.stack([{sv_exprs}])")

    src = "\n".join(lines)
    globs: dict = {}
    exec(compile(src, "<dfun_jax>", "exec"), globs)
    return globs["_dfun_jax"]


# ---------------------------------------------------------------------------
# Coupling primitives
# ---------------------------------------------------------------------------

def _instant_coupling(cvar_state: jnp.ndarray, weights: jnp.ndarray,
                      G: float, a: float, b: float) -> jnp.ndarray:
    """No-delay path. cvar_state: (n_cvar, N) -> coupling: (n_cvar, N)."""
    return G * a * jnp.einsum("ts,cs->ct", weights, cvar_state) + b


def _write_ring(buf: jnp.ndarray, step: int, cvar_state: jnp.ndarray,
                horizon: int) -> jnp.ndarray:
    """buf: (horizon, n_cvar, n_nodes) -> updated buf."""
    return buf.at[step % horizon].set(cvar_state)


def _read_delayed_coupling(
        buf: jnp.ndarray, step: int,
        delay_steps: jnp.ndarray, weights: jnp.ndarray,
        G: float, a: float, b: float,
        horizon: int, n_nodes: int) -> jnp.ndarray:
    """
    Returns coupling (n_cvar, n_nodes), reading each source node's delayed
    coupling-variable state out of the ring buffer via a flat-index gather
    (avoids a Python loop / dynamic 2-D gather inside traced code).
    """
    idx_time = (step - delay_steps) % horizon        # (N, N)
    src_idx = jnp.arange(n_nodes, dtype=jnp.int32)
    flat_idx = idx_time * n_nodes + src_idx[None, :]  # (N, N)

    n_cvar = buf.shape[1]
    cvars = []
    for cv in range(n_cvar):
        buf_cv = buf[:, cv, :]                        # (horizon, N)
        delayed = buf_cv.reshape(-1)[flat_idx]        # (N, N)
        cvars.append(G * a * jnp.sum(weights * delayed, axis=1) + b)
    return jnp.stack(cvars)                           # (n_cvar, N)


# ---------------------------------------------------------------------------
# Integrators (pure JAX, traceable)
# ---------------------------------------------------------------------------

def _euler(state, dfun, coupling, dt, params):
    return state + dt * dfun(state, coupling, params)


def _heun(state, dfun, coupling, dt, params):
    k1 = dfun(state, coupling, params)
    k2 = dfun(state + dt * k1, coupling, params)
    return state + 0.5 * dt * (k1 + k2)


# ---------------------------------------------------------------------------
# step(carry, _) factory
# ---------------------------------------------------------------------------

def make_step_fn(
        dfun: Callable,
        weights: jnp.ndarray,
        delay_steps: jnp.ndarray,
        has_delays: bool,
        horizon: int,
        n_nodes: int,
        cvar_indices: tuple[int, ...],
        dt: float,
        G_default: float,
        coup_a: float,
        coup_b: float,
        use_heun: bool = True,
) -> Callable:
    """
    Returns step(carry, _) -> (new_carry, new_state).

    carry = (state, buf, step_int32, params)

    ``params`` travels inside the carry (not closed over) so that, later,
    jax.vmap over swept parameters and jax.jacfwd/jvp w.r.t. state both see
    it as data rather than a baked-in constant.

    ``buf`` is a no-op array when ``has_delays=False`` — the ODE and DDE
    cases share this exact function; only the coupling branch differs.
    """
    cvar_idx = jnp.array(list(cvar_indices), dtype=jnp.int32)
    integrate = _heun if use_heun else _euler

    def _coupling(buf, step, state, params):
        G = params.get("G", G_default)
        if has_delays:
            return _read_delayed_coupling(
                buf, step, delay_steps, weights, G, coup_a, coup_b,
                horizon, n_nodes)
        cvar_state = state[cvar_idx]
        return _instant_coupling(cvar_state, weights, G, coup_a, coup_b)

    def step(carry, _):
        state, buf, t, params = carry
        coup = _coupling(buf, t, state, params)
        new_state = integrate(state, dfun, coup, dt, params)

        if has_delays:
            new_buf = _write_ring(buf, t, new_state[cvar_idx], horizon)
        else:
            new_buf = buf

        return (new_state, new_buf, t + 1, params), new_state

    return step
