"""
vmap parameter sweeps: the same Kuramoto transition, one XLA call (M6)
============================================================================

``plot_05_kuramoto_sync.py`` swept the coupling strength ``G`` with a plain
Python ``for`` loop, one full ``lyapunov_spectrum`` call per ``G`` value.
M6 adds ``lyapax.sweep.sweep_lyapunov_spectrum``, which does the same sweep
as a single ``jax.vmap``-batched call instead -- one XLA program that
computes every grid point together, rather than ``len(G_values)`` separate
Python-level dispatches (each of which re-enters the JAX/XLA call machinery
even when, as ``plot_07_speed_and_accuracy.py`` found, the compiled
executable itself is cached across same-shape calls).

**Why this was possible with no new tangent/QR math.**
``lyapax.simulator.make_step_fn``'s carry already threads ``params`` through
as data, not a closed-over Python constant (see that function's docstring
-- anticipated from M0/M4 specifically for this). The one place that still
closed over ``params`` was the thin ``lyapax.network.make_network_step_fn``
adapter -- its new sibling, ``make_parametrized_network_step_fn``, takes
``params`` as a call-time argument instead, and
``sweep_lyapunov_spectrum`` is then just ``jax.vmap`` applied to a function
that calls the *existing*, unmodified ``lyapunov_spectrum`` once. See the
M6 entry in notes/milestones.md.

**Correctness, not just speed.** The plot below reproduces
``plot_05``'s figure using the vmap sweep instead of the Python loop; the
values are compared point-by-point against a fresh Python-loop run of the
exact same system below and printed as a max-abs-diff (should be at or
near machine precision -- this is the same computation, just batched).
"""
# %%
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import time

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from lyapax.core import lyapunov_spectrum
from lyapax.coupling import kuramoto_coupling
from lyapax.network import make_network_step_fn, make_parametrized_network_step_fn
from lyapax.simulator import ModelSpec, StateVar, Parameter, build_jax_dfun
from lyapax.sweep import sweep_lyapunov_spectrum

# %%
n_nodes = 6
omega = jnp.linspace(-1.0, 1.0, n_nodes)
weights = jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)

model = ModelSpec(
    name="kuramoto",
    state_variables=(StateVar("theta", default_init=0.0),),
    parameters=(Parameter("omega", 0.0),),
    cvar=("theta",),
    dfun_str={"theta": "omega + c"},
)
dfun = build_jax_dfun(model)

dt = 1e-2
state0 = jnp.linspace(0.0, 2 * jnp.pi, n_nodes, endpoint=False)
G_values = jnp.linspace(0.0, 4.0, 13)
n_sweep = G_values.shape[0]

sweep_kwargs = dict(dt=dt, n_steps=5_000, renorm_every=10, k=2, t_transient=50.0)

# %%
# --- one vmap-batched call over the whole G grid ---
step_p = make_parametrized_network_step_fn(
    dfun, weights, model.cvar_indices, dt, kuramoto_coupling(alpha=0.0))
params_batch = {
    "omega": jnp.broadcast_to(omega, (n_sweep, n_nodes)),
    "G": G_values,
}

t0 = time.perf_counter()
swept = sweep_lyapunov_spectrum(step_p, state0, params_batch, **sweep_kwargs)
jax.block_until_ready(swept)
t_vmap = time.perf_counter() - t0
print(f"vmap sweep ({n_sweep} points): {t_vmap:.2f}s")

# %%
# --- reference: the plot_05 Python loop, same system, same G values ---
t0 = time.perf_counter()
lambda1_loop, lambda2_loop = [], []
for G in G_values:
    params = {"omega": omega, "G": float(G)}
    step = make_network_step_fn(
        dfun, weights, model.cvar_indices, params, dt,
        coupling_fn=kuramoto_coupling(alpha=0.0),
    )
    result = lyapunov_spectrum(step, state0=state0, **sweep_kwargs)
    lambda1_loop.append(float(result.exponents[0]))
    lambda2_loop.append(float(result.exponents[1]))
t_loop = time.perf_counter() - t0
print(f"Python loop ({n_sweep} points): {t_loop:.2f}s  ({t_loop / t_vmap:.1f}x slower than vmap)")

lambda1_swept = np.array(swept.exponents[:, 0])
lambda2_swept = np.array(swept.exponents[:, 1])
max_diff = max(
    np.max(np.abs(lambda1_swept - np.array(lambda1_loop))),
    np.max(np.abs(lambda2_swept - np.array(lambda2_loop))),
)
print(f"max |vmap - loop| over the whole grid: {max_diff:.2e} (same computation, just batched)")

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(G_values, lambda1_swept, "o-", label=r"$\lambda_1$ (vmap sweep)")
ax.plot(G_values, lambda2_swept, "o-", label=r"$\lambda_2$ (vmap sweep)")
ax.axhline(0.0, color="gray", lw=0.5)
ax.set_xlabel("coupling strength G")
ax.set_ylabel("Lyapunov exponent")
ax.set_title(f"Kuramoto sync transition via jax.vmap ({t_vmap:.1f}s vs {t_loop:.1f}s looped)")
ax.legend()
fig.tight_layout()
plt.show()
