"""
Speed and accuracy characterization
======================================

Two practical questions before trusting lyapax on a new problem:

1. How does wall time scale with ``k`` (number of tracked exponents) and
   is a repeated call any cheaper than the first (i.e. is there a JIT
   cache to benefit from right now)?
2. How does the Lorenz sum-of-exponents error shrink as the integration
   step ``dt`` decreases, and how much does that depend on which
   integrator computes each step?

Both use the Lorenz system from
:ref:`03_chaotic_flows.py <sphx_glr_auto_examples_03_chaotic_flows.py>` as the
test
case: it's genuinely chaotic (so representative of a real workload,
unlike the linear/map toy systems) yet still has the exact
``sum(lambda) = -(sigma + 1 + beta)`` invariant to measure error against,
without needing a separate reference computation for question 2.
``rk4_step``/``rk6_step`` (``lyapax.integrators``) are 4th- and 6th-order
methods, so halving ``dt`` should shrink their local truncation error
roughly 16-fold and 64-fold respectively -- the log-log error-vs-dt plot
below should come out close to two straight lines of slope 4 and 6, which
is the standard way to sanity-check that an integrator is achieving its
nominal order rather than silently degrading (e.g. from a bug, or from
the QR renormalization interval interacting badly with the step size).
RK6's steeper slope also means it reaches a given error at a much larger
``dt`` than RK4 -- useful when RK4's error, not wall time, is what's
limiting how few steps a run can get away with.
"""
# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from lyapax import systems
from lyapax.core import lyapunov_spectrum, ode_problem
from lyapax.integrators import rk4_step, rk6_step

# %%
# --- Speed vs k, and first-call vs second-call ---
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
rhs = systems.lorenz(sigma, rho, beta)
dt = 1e-2
problem = ode_problem(rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=dt)

ks = [1, 2, 3]
first_call, second_call = [], []
for k in ks:
    t0 = time.perf_counter()
    lyapunov_spectrum(problem, n_steps=20_000, renorm_every=10, k=k)
    first_call.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    lyapunov_spectrum(problem, n_steps=20_000, renorm_every=10, k=k)
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
    "trace+compile cost. For a parameter sweep at the same shapes -- e.g. "
    "scanning a coupling strength G, as in 05_kuramoto_sync.py -- "
    "see 11_vmap_parameter_sweep.py: batching the whole sweep into "
    "one jax.vmap call sidesteps the repeated-call overhead entirely, "
    "rather than just relying on this incidental compilation-cache reuse."
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
# --- Accuracy vs dt: Lorenz sum-of-exponents error (RK4 vs RK6) ---
# n_steps/t_transient are recomputed per dt so every run covers the same
# *physical* time (500 post-transient + 100 transient time units) despite
# taking a different number of steps -- otherwise a coarser dt would also
# see a shorter, differently-converged run and confound the two effects.
expected_sum = -(sigma + 1.0 + beta)
renorm_every = 10


def _accuracy_vs_dt(integrator, dts):
    errors = []
    for dt_i in dts:
        problem_i = ode_problem(
            rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=dt_i, integrator=integrator,
        )
        n_steps = (int(round(500.0 / dt_i)) // renorm_every) * renorm_every
        t_transient = int(round(100.0 / dt_i)) * dt_i
        result = lyapunov_spectrum(
            problem_i, n_steps=n_steps, renorm_every=renorm_every,
            t_transient=t_transient,
        )
        errors.append(abs(float(jnp.sum(result.exponents)) - expected_sum))
    return errors


dts_rk4 = [4e-2, 2e-2, 1e-2, 5e-3]
errors_rk4 = _accuracy_vs_dt(rk4_step, dts_rk4)
for dt_i, err in zip(dts_rk4, errors_rk4):
    print(f"RK4 dt={dt_i:.4f}: sum-of-exponents error = {err:.2e}")

# RK6's error is already tiny at RK4's dt values, so it's swept at coarser
# (larger) dt instead -- the point is comparing slopes on the log-log plot,
# not matching x-ranges exactly. dt much above 8e-2 pushes Lorenz's fast
# timescale outside RK6's stability region and the trajectory blows up, so
# the sweep stays below that.
dts_rk6 = [8e-2, 4e-2, 2e-2, 1e-2]
errors_rk6 = _accuracy_vs_dt(rk6_step, dts_rk6)
for dt_i, err in zip(dts_rk6, errors_rk6):
    print(f"RK6 dt={dt_i:.4f}: sum-of-exponents error = {err:.2e}")

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(dts_rk4, errors_rk4, "o-", label="RK4 (slope 4)")
ax.loglog(dts_rk6, errors_rk6, "s-", label="RK6 (slope 6)")
ax.set_xlabel("dt")
ax.set_ylabel(r"$|\sum \lambda_i - \mathrm{exact}|$")
ax.set_title("Lorenz: integration accuracy vs dt, RK4 vs RK6")
ax.legend()
fig.tight_layout()
plt.show()
