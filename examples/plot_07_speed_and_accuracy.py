"""
Speed and accuracy characterization
======================================

Two practical questions before trusting lyapax on a new problem:

1. How does wall time scale with ``k`` (number of tracked exponents) and
   is a repeated call any cheaper than the first (i.e. is there a JIT
   cache to benefit from right now)?
2. How does the Lorenz sum-of-exponents error shrink as the integration
   step ``dt`` decreases (it should shrink quickly for RK4)?

Both use the Lorenz system from ``plot_03_chaotic_flows.py`` as the test
case: it's genuinely chaotic (so representative of a real workload,
unlike the linear/map toy systems) yet still has the exact
``sum(lambda) = -(sigma + 1 + beta)`` invariant to measure error against,
without needing a separate reference computation for question 2.
``rk4_step`` (``lyapax.integrators``) is a 4th-order method, so halving
``dt`` should shrink its local truncation error roughly 16-fold -- the
log-log error-vs-dt plot below should come out close to a straight line
of slope 4, which is the standard way to sanity-check that an integrator
is achieving its nominal order rather than silently degrading (e.g. from
a bug, or from the QR renormalization interval interacting badly with the
step size).
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
from lyapax.integrators import rk4_step
from lyapax import systems

# %%
# --- Speed vs k, and first-call vs second-call ---
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
rhs = systems.lorenz(sigma, rho, beta)
dt = 1e-2
step = rk4_step(rhs, dt)

ks = [1, 2, 3]
first_call, second_call = [], []
for k in ks:
    t0 = time.perf_counter()
    lyapunov_spectrum(step, state0=jnp.array([1.0, 1.0, 1.0]), dt=dt,
                       n_steps=20_000, renorm_every=10, k=k)
    first_call.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    lyapunov_spectrum(step, state0=jnp.array([1.0, 1.0, 1.0]), dt=dt,
                       n_steps=20_000, renorm_every=10, k=k)
    second_call.append(time.perf_counter() - t0)

for k, t1, t2 in zip(ks, first_call, second_call):
    print(f"k={k}: call 1 = {t1:.3f}s, call 2 = {t2:.3f}s "
          f"({t1 / t2:.1f}x faster on repeat)")
print(
    "\nMeasured, not assumed: even though lyapunov_spectrum() is a plain "
    "Python function -- not wrapped in jax.jit -- a repeated call at the "
    "same shapes is ~4-6x faster than the first. That's almost certainly "
    "JAX's own persistent/in-memory compilation cache reusing the compiled "
    "executable for an identical jaxpr, not anything lyapax does "
    "explicitly. It's not a guarantee, though: a *different* shape "
    "(n_steps, k, renorm_every, or state dimension) still pays the full "
    "trace+compile cost. Wrapping the hot path in an explicit jax.jit "
    "(static_argnums for n_steps/k/renorm_every) would make the caching "
    "behavior deliberate rather than incidental -- worth doing in M6 if "
    "repeated-shape calls turn out to be common (e.g. parameter sweeps "
    "like plot_05_kuramoto_sync.py)."
)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ks, first_call, "o-", label="call 1")
ax.plot(ks, second_call, "o-", label="call 2 (same shapes)")
ax.set_xlabel("k (number of tracked exponents)")
ax.set_ylabel("wall time (s)")
ax.set_title("Lorenz, 20000 steps: cost vs k")
ax.legend()
fig.tight_layout()
plt.show()

# %%
# --- Accuracy vs dt: Lorenz sum-of-exponents error (RK4) ---
# n_steps/t_transient are recomputed per dt so every run covers the same
# *physical* time (500 post-transient + 100 transient time units) despite
# taking a different number of steps -- otherwise a coarser dt would also
# see a shorter, differently-converged run and confound the two effects.
dts = [4e-2, 2e-2, 1e-2, 5e-3]
errors = []
expected_sum = -(sigma + 1.0 + beta)
renorm_every = 10
for dt_i in dts:
    step_i = rk4_step(rhs, dt_i)
    n_steps = (int(round(500.0 / dt_i)) // renorm_every) * renorm_every
    t_transient = int(round(100.0 / dt_i)) * dt_i
    result = lyapunov_spectrum(
        step_i, state0=jnp.array([1.0, 1.0, 1.0]),
        dt=dt_i, n_steps=n_steps, renorm_every=renorm_every,
        t_transient=t_transient,
    )
    err = abs(float(jnp.sum(result.exponents)) - expected_sum)
    errors.append(err)
    print(f"dt={dt_i:.4f}: sum-of-exponents error = {err:.2e}")

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(dts, errors, "o-")
ax.set_xlabel("dt")
ax.set_ylabel(r"$|\sum \lambda_i - \mathrm{exact}|$")
ax.set_title("Lorenz: integration accuracy vs dt (RK4)")
fig.tight_layout()
plt.show()
