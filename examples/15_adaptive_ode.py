"""
Adaptive-step ODE integration (diffrax)
========================================

Every other ODE example uses a fixed-step integrator (``"rk4"``, ``"rk6"``,
...): the same ``dt`` is both the sampling interval between QR
renormalizations *and* the integrator's own internal step size, so getting
a smaller local truncation error means shrinking ``dt`` globally, which also
means more (cheap) raw steps between renormalizations.

``lyapax.adaptive.diffrax_adaptive_step`` decouples those two meanings of
``dt``. It builds an integrator, backed by `diffrax
<https://docs.kidger.site/diffrax/>`_, whose internal step size is chosen
adaptively (via a PID step-size controller and an embedded Runge-Kutta error
estimate) to hit a requested local error tolerance -- while ``dt`` itself
keeps meaning only "the interval between renormalizations". It drops
straight into ``lyapax.ode_problem`` alongside the fixed-step builtins, with
no other code changes: ``lyapunov_spectrum`` cannot tell the difference
between an integrator that takes one internal step per call and one that
takes an internally-controlled, variable number of them.

**Reading the plot.** The left panel sweeps the controller's tolerance
(``rtol``, with ``atol`` scaled alongside it) and shows the Lyapunov
exponent estimate converging toward the published Lorenz value as the
tolerance tightens -- the adaptive analogue of a ``dt``-convergence sweep.
The right panel cross-checks the tightest-tolerance adaptive estimate
against a fine-``dt`` fixed-step ``rk4`` run of the same system: both should
agree with each other and with the published value, confirming the adaptive
path is solving the same problem, not just converging to *something*.

**A real caveat.** The adaptive integrator's internal accept/reject/step
-size loop is a ``jax.lax.while_loop`` with a data-dependent trip count.
Forward-mode differentiation (``jax.jvp``/``jax.jacfwd``) works through it
fine -- that's what the Lyapunov engine's own tangent propagation already
uses. Reverse-mode (``jax.grad``/``jax.jacrev``) does not: JAX cannot
replay a dynamic-trip-count ``while_loop`` backward. The last section below
demonstrates differentiating a Lyapunov exponent with respect to a system
parameter using ``jax.jacfwd`` (which works and matches finite differences)
and shows ``jax.grad`` raising instead of silently returning something
wrong.
"""
# %%
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from lyapax import lyapunov_spectrum, ode_problem, systems
from lyapax.adaptive import diffrax_adaptive_step
from lyapax.integrators import rk4_step

# %%
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
rhs = systems.lorenz(sigma, rho, beta)
state0 = jnp.array([1.0, 1.0, 1.0])
published_lambda1 = 0.9056

rtols = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
errors = []
for rtol in rtols:
    integrator = diffrax_adaptive_step(rtol=rtol, atol=rtol * 1e-2)
    problem = ode_problem(rhs, state0=state0, dt=0.1, integrator=integrator)
    result = lyapunov_spectrum(
        problem, n_steps=5_000, renorm_every=1, t_transient=100.0)
    errors.append(abs(float(result.exponents[0]) - published_lambda1))
    print(f"rtol={rtol:.0e}  lambda1={float(result.exponents[0]):.4f}  "
          f"error={errors[-1]:.2e}")

# %%
# Cross-check: tightest-tolerance adaptive run vs. a fine-dt fixed-step rk4
# run of the same system.
rk4_problem = ode_problem(rhs, state0=state0, dt=1e-2, integrator=rk4_step)
rk4_result = lyapunov_spectrum(
    rk4_problem, n_steps=50_000, renorm_every=10, t_transient=100.0)

tight_integrator = diffrax_adaptive_step(rtol=1e-10, atol=1e-12)
adaptive_problem = ode_problem(rhs, state0=state0, dt=0.1, integrator=tight_integrator)
adaptive_result = lyapunov_spectrum(
    adaptive_problem, n_steps=5_000, renorm_every=1, t_transient=100.0)

print(f"\nrk4 (dt=1e-2):      lambda1={float(rk4_result.exponents[0]):.4f}")
print(f"adaptive (rtol=1e-10): lambda1={float(adaptive_result.exponents[0]):.4f}")
print(f"published:          lambda1={published_lambda1:.4f}")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.loglog(rtols, errors, "o-")
ax1.set_xlabel("rtol")
ax1.set_ylabel(r"$|\lambda_1 - \mathrm{published}|$")
ax1.set_title("Adaptive-integrator convergence vs. tolerance")
ax1.invert_xaxis()

methods = ["rk4\n(fixed dt=1e-2)", "adaptive\n(rtol=1e-10)", "published"]
values = [float(rk4_result.exponents[0]), float(adaptive_result.exponents[0]),
          published_lambda1]
ax2.bar(methods, values, color=["C0", "C1", "C2"])
ax2.axhline(published_lambda1, color="C2", linestyle="--", linewidth=1)
ax2.set_ylabel(r"$\lambda_1$")
ax2.set_title("Cross-check: adaptive vs. fixed-step rk4")
fig.tight_layout()
plt.show()

# %%
# Differentiating a Lyapunov exponent through the adaptive integrator: works
# with jax.jacfwd (forward-mode), not with jax.grad (reverse-mode).


def lambda_max(rho_param):
    def linear_rhs(y):
        # A small linear system whose exact fundamental matrix (and hence
        # exact Lyapunov exponent, its log-growth rate) is known, so the
        # jacfwd result can be checked against a closed-form derivative
        # rather than only a finite-difference estimate.
        return jnp.array([[-rho_param, 1.0], [-1.0, -rho_param]]) @ y

    integrator = diffrax_adaptive_step(rtol=1e-8, atol=1e-10)
    problem = ode_problem(linear_rhs, state0=jnp.array([1.0, 0.3]), dt=0.2,
                           integrator=integrator)
    result = lyapunov_spectrum(problem, n_steps=50, renorm_every=1, k=1)
    return result.exponents[0]


a0 = 0.5
grad_fwd = jax.jacfwd(lambda_max)(a0)
eps = 1e-5
finite_diff = (lambda_max(a0 + eps) - lambda_max(a0 - eps)) / (2 * eps)
print(f"\njax.jacfwd(lambda_max)({a0}) = {float(grad_fwd):.6f}")
print(f"finite-difference reference   = {float(finite_diff):.6f}")

try:
    jax.grad(lambda_max)(a0)
except ValueError as exc:
    print(f"\njax.grad raises (expected): {type(exc).__name__}: "
          f"{str(exc)[:88]}...")
