"""
Resuming a DDE run: convergence_drift + resume, ring-buffer edition
=====================================================================

:ref:`16_convergence_drift.py <sphx_glr_auto_examples_16_convergence_drift.py>`
showed the "run a chunk, look, continue if not converged" workflow for a
plain ODE. The same workflow applies to a fixed-delay DDE via
``lyapunov_spectrum_dde(..., resume=...)`` -- with one difference under the
hood: an ODE's Markovian state is just ``state``, so its checkpoint
(``lyapax.core.LyapunovCheckpoint``) only needs to carry ``state`` and the
tangent basis ``Y``. A DDE's Markovian state is ``(state, buf)`` together --
``buf`` is the ring buffer holding the recent history the delayed term
reads from -- so ``lyapax.dde.DDECheckpoint`` additionally carries ``buf``,
its own tangent basis ``Y_buf``, and the ring-buffer step counter ``t``.
Dropping any of those would mean the next chunk starts from an incomplete
history and silently gets a biased exponent, not a crash.

``convergence_drift`` and the resume loop themselves are identical to the
ODE case -- ``lyapunov_spectrum_dde`` returns the same ``LyapunovResult``
shape, just with a ``DDECheckpoint`` instead of a ``LyapunovCheckpoint``
under ``result.checkpoint``.

**The test system.** Mackey-Glass, the standard chaotic DDE benchmark --
its dominant exponent is small (order 1e-2 to 1e-3) and noisy chunk to
chunk, so unlike a clean linear DDE it genuinely takes several resumed
chunks to settle within tolerance, making it a more honest showcase of the
run-inspect-resume loop than a system that converges in one chunk.

**Reading the plot.** Each vertical dashed line is a resume point; the
curve is seamless across them (same as the ODE demo), settling toward a
small positive value as more chunks accumulate.
"""
# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from lyapax import systems
from lyapax.core import convergence_drift
from lyapax.dde import dde_problem, lyapunov_spectrum_dde

# %%
beta, gamma, n, tau, dt = 0.2, 0.1, 10.0, 17.0, 1.0
rhs = systems.mackey_glass(beta=beta, gamma=gamma, n=n)
problem = dde_problem(rhs, state0=jnp.array([1.2]), tau=tau, dt=dt)
chunk, tol = 1_000, 5e-3

result = lyapunov_spectrum_dde(problem, n_steps=chunk, k=2, renorm_every=10, t_transient=3_000.0)
history_chunks, time_chunks = [np.array(result.history)], [np.array(result.times)]
drift = convergence_drift(result, tol=tol)
print(f"chunk 1: lambda1={float(result.exponents[0]):.5f}  converged={bool(drift.converged[0])}")

while not bool(drift.converged[0]):
    result = lyapunov_spectrum_dde(
        problem, n_steps=chunk, k=2, renorm_every=10, resume=result.checkpoint)
    history_chunks.append(np.array(result.history))
    time_chunks.append(np.array(result.times))
    drift = convergence_drift(result, tol=tol)
    print(f"chunk {len(history_chunks)}: lambda1={float(result.exponents[0]):.5f}  "
          f"converged={bool(drift.converged[0])}")

# %%
history, times = np.concatenate(history_chunks), np.concatenate(time_chunks)

fig, ax = plt.subplots(figsize=(7, 4))
for i in range(2):
    ax.plot(times, history[:, i], label=rf"$\lambda_{i + 1}$")
for c in time_chunks[:-1]:
    ax.axvline(c[-1], color="gray", linestyle="--", linewidth=0.8)
ax.axhline(0.0, color="gray", lw=0.5)
ax.set_xlabel("time")
ax.set_ylabel("running LE estimate")
ax.set_title(f"Mackey-Glass DDE resumed across {len(history_chunks)} chunks of n_steps={chunk}")
ax.legend()
fig.tight_layout()
plt.show()
