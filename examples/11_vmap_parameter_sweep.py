"""
vmap parameter sweeps: the same Kuramoto transition, one XLA call
=======================================================================

:ref:`05_kuramoto_sync.py <sphx_glr_auto_examples_05_kuramoto_sync.py>` swept
the coupling strength ``G`` with a plain
Python ``for`` loop, one full ``lyapunov_spectrum`` call per ``G`` value.
``lyapax.sweep.sweep_lyapunov_spectrum`` does the same sweep as a single
``jax.vmap``-batched call instead -- one XLA program that computes every
grid point together, rather than ``len(G_values)`` separate Python-level
dispatches (each of which re-enters the JAX/XLA call machinery even when,
as
:ref:`07_speed_and_accuracy.py <sphx_glr_auto_examples_07_speed_and_accuracy.py>`
found, the compiled executable itself
is cached across same-shape calls).

**Why sweeping needs a different step function.** ``jax.vmap`` can only
batch over things passed to it as *data*, not over values baked into a
closure at construction time -- and both ``network_problem`` and the
lower-level ``make_network_step_fn`` it wraps close ``params`` (e.g. the
coupling strength ``G``) over when the step is built, exactly like the
non-swept examples use it. ``network_step_parametrized`` is the same
``Network``-based wiring with ``params`` taken as a call-time argument
instead (``step(state, params) -> new_state``), so a batch of ``params``
values can be threaded through a single ``jax.vmap`` call around the
*existing*, unmodified ``lyapunov_spectrum`` -- no new tangent-propagation
or QR code needed, since batching a computation is orthogonal to what the
computation itself does.

**Correctness, not just speed.** The plot below reproduces
:ref:`05_kuramoto_sync.py <sphx_glr_auto_examples_05_kuramoto_sync.py>`'s
figure using the vmap sweep instead of the Python loop; the
values are compared point-by-point against a fresh Python-loop run of the
exact same system below and printed as a max-abs-diff (should be at or
near machine precision -- this is the same computation, just batched).
"""
# %%
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from lyapax.core import lyapunov_spectrum
from lyapax.coupling import kuramoto_coupling
from lyapax.network import Network, network_problem, network_step_parametrized
from lyapax.simulator import ModelSpec, Parameter, StateVar, build_jax_dfun
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
network = Network(weights=weights, cvar_indices=model.cvar_indices)

dt = 1e-2
state0 = jnp.linspace(0.0, 2 * jnp.pi, n_nodes, endpoint=False)
G_values = jnp.linspace(0.0, 4.0, 13)
n_sweep = G_values.shape[0]

sweep_kwargs = dict(dt=dt, n_steps=5_000, renorm_every=10, k=2, t_transient=50.0)

# %%
# --- one vmap-batched call over the whole G grid ---
step_p = network_step_parametrized(
    dfun, network, kuramoto_coupling(alpha=0.0), dt)
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
# --- reference: the 05_kuramoto_sync.py Python loop, same system, same G values ---
loop_kwargs = {k: v for k, v in sweep_kwargs.items() if k != "dt"}
t0 = time.perf_counter()
lambda1_loop, lambda2_loop = [], []
for G in G_values:
    params = {"omega": omega, "G": float(G)}
    problem = network_problem(
        dfun, network, kuramoto_coupling(alpha=0.0),
        params=params, state0=state0, dt=dt,
    )
    result = lyapunov_spectrum(problem, **loop_kwargs)
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
