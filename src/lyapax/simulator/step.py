"""dfun codegen + the ring-buffer step(carry, _) factory.

Adapted from ``vbi/simulator/backend/jax_/codegen.py`` and
``vbi/simulator/backend/jax_/simulator.py`` — see NOTICE.md in this directory
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

from ..integrators import rk6_combine
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

    Per-edge delay (delay_steps is a full (N, N) matrix): the coupling
    formula (G*a*sum+b) is baked in here rather than exposed as a separate
    "gather" step, because different edges read different time offsets --
    there's no single (n_cvar, n_nodes) "the delayed cvar state" independent
    of which edge is asking, so a plain coupling_fn(cvar_state, weights,
    params) signature (one reading per node) can't express this in general.
    Left untouched for M5 (per-edge delay + coupling_fn needs an edge-aware
    signature, a separate design fork) -- see _read_uniform_delayed_cvar
    below for the M4 case where this collapses back to one reading per node.
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


def _read_uniform_delayed_cvar(
        buf: jnp.ndarray, step: int, tau_steps: int, horizon: int) -> jnp.ndarray:
    """M4: a single global delay (not per-edge), so the delayed reading is
    the same for every node -- one O(1) modular ring-buffer read, no
    flat-index gather needed. Returns cvar_state (n_cvar, n_nodes), the same
    shape lyapax.coupling's plain-callable coupling_fn expects for the
    zero-delay case, so any existing coupling_fn works unmodified against
    either instantaneous or uniformly-delayed cvar_state."""
    return buf[(step - tau_steps) % horizon]


# ---------------------------------------------------------------------------
# Integrators (pure JAX, traceable)
# ---------------------------------------------------------------------------

def _euler(state, dfun, coupling, dt, params):
    return state + dt * dfun(state, coupling, params)


def _heun(state, dfun, coupling, dt, params):
    k1 = dfun(state, coupling, params)
    k2 = dfun(state + dt * k1, coupling, params)
    return state + 0.5 * dt * (k1 + k2)


def _rk4(state, dfun, coupling, dt, params):
    """Classic RK4, with ``coupling`` held fixed across the four stages
    (same simplifying convention ``_heun`` already uses: coupling is read
    once per step, not re-evaluated at the RK substeps)."""
    k1 = dfun(state, coupling, params)
    k2 = dfun(state + 0.5 * dt * k1, coupling, params)
    k3 = dfun(state + 0.5 * dt * k2, coupling, params)
    k4 = dfun(state + dt * k3, coupling, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk6(state, dfun, coupling, dt, params):
    """Fixed-step 6th-order Runge-Kutta, ``coupling`` held fixed across all
    nine stages (same convention as ``_rk4``/``_heun``). See
    ``lyapax.integrators.rk6_combine`` for the tableau, provenance, and
    correctness checks -- this just plugs the coupled-network derivative
    ``dfun(y, coupling, params)`` into that shared 9-stage combination."""
    return rk6_combine(lambda y: dfun(y, coupling, params), state, dt)


_STEP_INTEGRATORS: dict[str, Callable] = {
    "euler": _euler,
    "heun": _heun,
    "rk4": _rk4,
    "rk6": _rk6,
}


# ---------------------------------------------------------------------------
# step(carry, _) factory
# ---------------------------------------------------------------------------

def make_step_fn(
        dfun: Callable,
        weights: jnp.ndarray,
        has_delays: bool,
        horizon: int,
        n_nodes: int,
        cvar_indices: tuple[int, ...],
        dt: float,
        delay_steps: jnp.ndarray | None = None,
        G_default: float = 1.0,
        coup_a: float = 1.0,
        coup_b: float = 0.0,
        integrator: str | Callable = "heun",
        coupling_fn: Callable | None = None,
        tau_steps: int | None = None,
) -> Callable:
    """
    Returns step(carry, _) -> (new_carry, new_state).

    carry = (state, buf, step_int32, params)

    ``params`` travels inside the carry (not closed over) so that, later,
    jax.vmap over swept parameters and jax.jacfwd/jvp w.r.t. state both see
    it as data rather than a baked-in constant.

    ``buf`` is a no-op array when ``has_delays=False`` — the ODE and DDE
    cases share this exact function; only the coupling branch differs.

    integrator : ``"euler"``, ``"heun"``, ``"rk4"``, ``"rk6"``, or a callable
        ``(state, dfun, coupling, dt, params) -> new_state``. ``coupling``
        is read once per step and held fixed across any internal stages
        (see ``_rk4``'s docstring).
    coupling_fn : optional ``lyapax.coupling``-style plain callable,
        ``(cvar_state, weights, params) -> coupling`` (see
        ``lyapax.coupling.CouplingFn``). When given, replaces the
        hardcoded ``G_default``/``coup_a``/``coup_b`` linear formula for
        *both* the zero-delay and (uniform-delay-only, see ``tau_steps``)
        delayed branches, unifying this step with M3's coupling design
        (notes/milestones.md, M4). ``G_default``/``coup_a``/``coup_b`` are
        ignored when ``coupling_fn`` is given. Default ``None`` preserves
        the exact original (pre-M4) behavior.
    tau_steps : required alongside ``coupling_fn`` when ``has_delays=True``
        -- a single global delay (in steps), read via
        ``_read_uniform_delayed_cvar`` (O(1) ring-buffer read). Per-edge
        heterogeneous delays (the general ``delay_steps`` matrix) are not
        supported together with a custom ``coupling_fn`` yet -- that
        combination needs an edge-aware coupling_fn signature, left for a
        future milestone (M5); use the legacy ``coupling_fn=None`` path
        (which does support per-edge ``delay_steps``, just with the
        hardcoded linear formula) until then.
    """
    cvar_idx = jnp.array(list(cvar_indices), dtype=jnp.int32)
    integrate = _STEP_INTEGRATORS[integrator] if isinstance(integrator, str) else integrator

    if coupling_fn is not None:
        def _coupling(buf, step, state, params):
            if has_delays:
                cvar_state = _read_uniform_delayed_cvar(buf, step, tau_steps, horizon)
            else:
                cvar_state = state[cvar_idx]
            return coupling_fn(cvar_state, weights, params)
    else:
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
