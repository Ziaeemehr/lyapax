"""
DDE history reads: grid-snapped vs Hermite-interpolated
============================================================

Every delayed-coupling example so far has read history off an exact ring-
buffer grid point: the physical delay ``tau`` is rounded to the nearest
whole number of ``dt`` steps before the run even starts
(``resolve_tau_steps``), and every later lookup pulls that exact stored
sample. That's simple and fully differentiable, but it means the delay
actually being simulated is ``tau_eff = tau_steps * dt``, not the ``tau``
that was asked for -- and, because rounding to the nearest integer is not
a smooth function of ``dt``, the *size* of that mismatch does not shrink
smoothly as ``dt`` gets finer. It can even get worse at a smaller ``dt``
if the rounding happens to land less favorably there than at a coarser
one.

``lyapax.dde_problem(..., interpolate=True)`` (and its networked
counterpart, ``network_dde_problem``) removes the rounding: instead of
reading one stored ring-buffer sample, it reconstructs the delayed value
via a cubic Hermite interpolant built from the *value and derivative*
stored at the two grid points bracketing the exact delayed time. ``tau``
is then used exactly -- no snapping, no warning -- and refining ``dt``
gives a smooth, monotonically shrinking error instead of a jumpy one.

**The test system.** The scalar linear DDE ``x'(t) = -a*x(t-tau)`` has a
transcendental characteristic equation whose dominant root is a Lambert-W
expression, giving an exact reference Lyapunov exponent to compare
against at every ``dt`` -- the same system used in
``plot_01_linear_ode.py``'s non-delayed counterpart, but here it lets this
demo measure *how the error behaves as dt shrinks*, not just its size at
one ``dt``.

**Reading the plot.** The grid-snapped curve should look noticeably less
smooth than the interpolated one -- not simply "bigger error", but visibly
non-monotonic in places, since its error is dominated by which way
``tau/dt`` happens to round at each ``dt``, not by a shrinking numerical
truncation error. The interpolated curve should look like a clean,
(roughly) straight line on log-log axes instead.

**An honest caveat.** The interpolated curve's slope comes out close to 1
(error roughly halves when ``dt`` halves), not the much steeper slope a
textbook Hermite interpolant (built from exact derivatives) would give.
That cap does not come from the interpolation formula itself -- tested in
isolation, against a known smooth function, it reconstructs values at its
expected ~4th-order accuracy. A related fix landed for zero-delay coupled
networks: they were capped the same way by coupling being read once per
step and held fixed across it, rather than re-evaluated as the state
evolved within it; recomputing coupling at each integrator stage's own
state removed that cap entirely there. The analogous change for this
delayed case -- reconstructing the delayed history lookup at each stage's
own intra-step *time*, instead of once per step -- was tried and did
*not* reduce this O(dt) cap, for reasons not yet understood (see
notes/stepping_accuracy_review.md for what was ruled out). Interpolation
still delivers what it promises -- an arbitrary, non-grid-aligned ``tau``
used exactly, with smooth, predictable convergence -- just capped at this
still-unexplained limit rather than at a higher order.
"""
# %%
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import lambertw

jax.config.update("jax_enable_x64", True)

from lyapax import systems
from lyapax.dde import dde_problem, lyapunov_spectrum_dde

# %%
a = 0.5
tau = 0.317  # deliberately not a clean multiple of any dt swept below
rhs = systems.linear_scalar_dde(a=a)
expected = float((lambertw(-a * tau, k=0) / tau).real)
print(f"exact dominant exponent (Lambert W): {expected:.6f}")

dts = [4e-2, 2e-2, 1e-2, 5e-3, 2.5e-3]
n_steps, renorm_every, t_transient = 40_000, 5, 20.0


def _run(interpolate):
    errors, tau_used = [], []
    for dt in dts:
        problem = dde_problem(
            rhs, state0=jnp.array([1.0]), tau=tau, dt=dt,
            integrator="heun", interpolate=interpolate,
        )
        result = lyapunov_spectrum_dde(
            problem, n_steps=n_steps, k=1, renorm_every=renorm_every,
            t_transient=t_transient,
        )
        errors.append(abs(float(result.exponents[0]) - expected))
        tau_used.append(problem.tau_steps * dt)
    return errors, tau_used


errors_snap, tau_eff_snap = _run(interpolate=False)
errors_interp, tau_eff_interp = _run(interpolate=True)

print(f"\n{'dt':>8}  {'tau_eff (snap)':>14}  {'err (snap)':>11}  "
      f"{'tau_eff (interp)':>16}  {'err (interp)':>13}")
for dt, te_s, e_s, te_i, e_i in zip(
        dts, tau_eff_snap, errors_snap, tau_eff_interp, errors_interp):
    print(f"{dt:8.4f}  {te_s:14.5f}  {e_s:11.2e}  {te_i:16.5f}  {e_i:13.2e}")

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(dts, errors_snap, "o-", label="grid-snapped (default)")
ax.loglog(dts, errors_interp, "s-", label="Hermite-interpolated")
ax.set_xlabel("dt")
ax.set_ylabel(r"$|\lambda_1 - \mathrm{exact}|$")
ax.set_title(r"Scalar linear DDE, $\tau=0.317$ (not a multiple of any dt swept)")
ax.legend()
fig.tight_layout()
plt.show()
