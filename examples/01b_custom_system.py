"""
Writing your own system: no registry needed
===============================================

``rhs`` is just a plain ``state -> jnp.ndarray`` callable -- not a fixed
set of named systems dispatched through hardcoded if/elif branches
internally. That means using your own equations never requires touching
the library's source or registering anything: any function with the
right signature works with ``ode_problem``, exactly like the built-in
``lorenz``/``rossler``/``linear_system`` in ``lyapax.systems``.

**The system.** To prove the two are interchangeable, this demo
reimplements ``lyapax.systems.lorenz`` by hand -- no import from
``lyapax.systems`` at all -- and checks the result against the same
exact structural invariant used in
:ref:`03_chaotic_flows.py <sphx_glr_auto_examples_03_chaotic_flows.py>`:
``trace(J) = -(sigma + 1 + beta)`` is constant, so the sum of all three
exponents is known exactly, regardless of how well the attractor itself
is resolved.
"""
# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from lyapax.core import lyapunov_spectrum, ode_problem


# %%
def my_lorenz(state):
    """A user-written system -- deliberately not imported from
    lyapax.systems, to prove the extension point needs no library
    changes and no registration step."""
    x, y, z = state[0], state[1], state[2]
    return jnp.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])


# %%
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
problem = ode_problem(my_lorenz, state0=jnp.array([1.0, 1.0, 1.0]), dt=1e-2)

result = lyapunov_spectrum(
    problem, n_steps=50_000, renorm_every=10, t_transient=100.0,
)

expected_sum = -(sigma + 1.0 + beta)
estimate = np.array(result.exponents)
print(f"lambda        = {estimate}")
print(f"sum(lambda)   = {float(jnp.sum(result.exponents)):.4f}"
      f"   exact -(sigma+1+beta) = {expected_sum:.4f}")
print(f"lambda1       = {float(result.exponents[0]):.4f}   published ~0.9056")
