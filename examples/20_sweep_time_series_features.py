"""
Time series from a vmap sweep: a second feature, from the same run
=======================================================================

:ref:`11_vmap_parameter_sweep.py
<sphx_glr_auto_examples_11_vmap_parameter_sweep.py>` batches
``lyapunov_spectrum`` over a grid of Kuramoto coupling strengths ``G`` with
``jax.vmap``, but only returns exponents -- no way to look at *how* each
simulation actually behaved without re-running it. ``save_states=True`` on
``lyapax.sweep.sweep_lyapunov_spectrum`` fixes that: it emits the state
time series from the same scan that computes the exponents (no extra
dynamics cost), batched across the sweep exactly like every other field.

This demo derives a second, independent feature from that saved time
series -- the Kuramoto order parameter ``R(t) = |mean_i(exp(i*theta_i(t)))|``,
which is 0 for incoherent phases and 1 for perfect phase-lock -- and shows
it tells the same synchronization story as the Lyapunov spectrum from
:ref:`05_kuramoto_sync.py <sphx_glr_auto_examples_05_kuramoto_sync.py>`:
``lambda_2`` (the sync-signaling exponent -- ``lambda_1`` stays pinned at
the marginal rotation mode, see that example) turns negative right where
``mean(R)`` rises toward 1.

**Memory.** A full-resolution trajectory is ``n_sweep * n_steps * n_nodes``
floats -- this demo prints that figure with and without ``decimate`` to
make the tradeoff concrete, and uses ``batch_size`` to bound how much of
the sweep sits on-device at once (see ``sweep_lyapunov_spectrum``'s
``batch_size``/``trajectory_to_host`` docstring for when a wide, fine-
resolution sweep needs those). Finally, ``lyapax.sweep.sweep_trajectory``
-- the replay-only counterpart, no tangent propagation -- is cross-checked
against the trajectory this run already saved, printed as a max-abs-diff.
"""
# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from lyapax.coupling import kuramoto_coupling
from lyapax.network import Network, network_step_parametrized
from lyapax.simulator import ModelSpec, Parameter, StateVar, build_jax_dfun
from lyapax.sweep import sweep_lyapunov_spectrum, sweep_trajectory

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

n_steps = 5_000
renorm_every = 10
decimate = 10
n_saves = n_steps // decimate

step_p = network_step_parametrized(dfun, network, kuramoto_coupling(alpha=0.0), dt)
params_batch = {"omega": jnp.broadcast_to(omega, (n_sweep, n_nodes)), "G": G_values}

nbytes_full = n_sweep * n_steps * n_nodes * 8
nbytes_decimated = n_sweep * n_saves * n_nodes * 8
print(f"trajectory footprint: {nbytes_full / 1e6:.1f} MB full resolution, "
      f"{nbytes_decimated / 1e6:.1f} MB at decimate={decimate}")

# %%
# --- one vmap-batched call: exponents AND the state time series they came from ---
swept = sweep_lyapunov_spectrum(
    step_p, state0, params_batch, dt=dt, n_steps=n_steps,
    renorm_every=renorm_every, k=2, t_transient=50.0,
    save_states=True, decimate=decimate, batch_size=4,
)
print(f"swept.trajectory.shape = {swept.trajectory.shape}  "
      f"(n_sweep={n_sweep}, n_saves={n_saves}, n_nodes={n_nodes})")

# %%
# --- a feature the exponents alone don't give: the Kuramoto order parameter ---
# R(t) = |mean_i exp(i*theta_i(t))| -- 0 for incoherent phases, 1 for perfect lock.
order_parameter = jnp.abs(jnp.mean(jnp.exp(1j * swept.trajectory), axis=-1))  # (n_sweep, n_saves)
mean_R = jnp.mean(order_parameter, axis=-1)  # (n_sweep,)

lambda2 = np.array(swept.exponents[:, 1])
mean_R = np.array(mean_R)

# %%
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(G_values, lambda2, "o-", color="C0", label=r"$\lambda_2$ (Lyapunov spectrum)")
ax1.axhline(0.0, color="gray", lw=0.5)
ax1.set_xlabel("coupling strength G")
ax1.set_ylabel(r"$\lambda_2$", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")

ax2 = ax1.twinx()
ax2.plot(G_values, mean_R, "s-", color="C1", label=r"mean $R(t)$ (order parameter)")
ax2.set_ylabel(r"mean $R(t)$", color="C1")
ax2.tick_params(axis="y", labelcolor="C1")
ax2.set_ylim(0.0, 1.05)

ax1.set_title("Two features from one sweep: spectrum and time series")
fig.tight_layout()
plt.show()

# %%
# --- cross-check: the replay-only path (no tangent propagation) reproduces
# the trajectory the LE run already saved ---
replayed_t, replayed_traj = sweep_trajectory(
    step_p, state0, params_batch, dt=dt, n_steps=n_steps,
    t_transient=50.0, renorm_every=renorm_every, decimate=decimate,
)
max_diff = float(jnp.max(jnp.abs(replayed_traj - swept.trajectory)))
print(f"max |sweep_trajectory - swept.trajectory|: {max_diff:.2e} "
      "(same states, one via the LE run, one via a direct replay)")
