"""
Kuramoto ODE: Lyapunov exponents and state time series, in one figure
=======================================================================

The simplest version of what ``save_states=True`` gives you: run
``lyapax.sweep.sweep_lyapunov_spectrum`` (the ``jax.vmap``-batched ODE
engine) on a small Kuramoto coupling-strength sweep, and get back both the
Lyapunov exponents *and* the phase time series they were computed from,
from one call. This demo just plots the two side by side for a single
point in the sweep -- see :ref:`20_sweep_time_series_features.py
<sphx_glr_auto_examples_20_sweep_time_series_features.py>` for deriving a
further feature (the order parameter) from the saved states across the
*whole* sweep.

Kept small on purpose: 4 oscillators, 3 coupling values, 2,000 steps --
runs in well under a second.
"""
# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from lyapax.coupling import kuramoto_coupling
from lyapax.network import Network, network_step_parametrized
from lyapax.simulator import ModelSpec, Parameter, StateVar, build_jax_dfun
from lyapax.sweep import sweep_lyapunov_spectrum

# %%
n_nodes = 4
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
G_values = jnp.array([0.5, 1.5, 3.0])
n_sweep = G_values.shape[0]

n_steps = 2_000
renorm_every = 10
decimate = 2  # keep every 2nd step -> 1,000 saved rows per run

step_p = network_step_parametrized(dfun, network, kuramoto_coupling(alpha=0.0), dt)
params_batch = {"omega": jnp.broadcast_to(omega, (n_sweep, n_nodes)), "G": G_values}

# %%
# --- one sweep call: exponents AND the state time series they came from ---
swept = sweep_lyapunov_spectrum(
    step_p, state0, params_batch, dt=dt, n_steps=n_steps,
    renorm_every=renorm_every, k=2, t_transient=5.0,
    save_states=True, decimate=decimate,
)
print(f"swept.exponents.shape  = {swept.exponents.shape}")
print(f"swept.trajectory.shape = {swept.trajectory.shape}  (n_sweep, n_saves, n_nodes)")
for G, lambdas in zip(G_values, swept.exponents):
    print(f"  G={float(G):.1f}  lambda = {[f'{float(lam):+.3f}' for lam in lambdas]}")

# %%
# --- one point of the sweep, in detail: phase time series and LE convergence ---
i = 0  # G=0.5
G = float(G_values[i])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

ax1.plot(swept.save_times[i], swept.trajectory[i])
ax1.set_ylabel(r"$\theta_i(t)$")
ax1.set_title(f"saved state time series (G={G:.1f})")

ax2.plot(swept.times[i], swept.history[i, :, 0], label=r"$\lambda_1$")
ax2.plot(swept.times[i], swept.history[i, :, 1], label=r"$\lambda_2$")
ax2.axhline(0.0, color="gray", lw=0.5)
ax2.set_xlabel("time since end of transient")
ax2.set_ylabel("running Lyapunov exponent estimate")
ax2.set_title("convergence to swept.exponents[i]")
ax2.legend()

fig.tight_layout()
plt.show()
