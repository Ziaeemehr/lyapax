"""
Custom coupling functions: no registry needed
=================================================

lyapax coupling is a plain callable
``coupling_fn(cvar_state, weights, params) -> coupling`` -- not a closed
enum dispatched through hardcoded if/elif branches (contrast with vbi's
``CouplingSpec.kind``, see the M3 note in ``notes/milestones.md``). This
demo writes one from scratch -- no import from ``lyapax.coupling`` at all
-- and plugs it directly into ``make_network_step_fn``, reproducing the
exact linear-network result from ``plot_04_linear_network.py``.

**The system.** Same idea as ``plot_04_linear_network.py`` -- scalar
linear node dynamics ``x_i' = gamma * x_i + c_i`` coupled linearly, so
the whole network reduces to ``x' = (gamma*I + G*W) x`` -- but scaled
down to 2 nodes joined by a single edge (``weights = [[0,1],[1,0]]``).
That adjacency matrix has eigenvalues ``{1, -1}``, so ``A``'s eigenvalues
are exactly ``gamma +/- G = {-0.8, -2.2}``. The only thing this example
changes versus ``plot_04`` is the coupling term itself:
``my_linear_coupling`` below reimplements ``lyapax.coupling.linear_coupling``
by hand (same ``G * W @ x`` formula) purely to demonstrate that
``make_network_step_fn`` accepts *any* callable of the right signature --
the result should (and does) match ``plot_04``'s exact-eigenvalue check
bit-for-bit in spirit, just for this smaller network.
"""
# %%
import os 
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from lyapax.core import lyapunov_spectrum
from lyapax.network import make_network_step_fn
from lyapax.simulator import ModelSpec, StateVar, Parameter, build_jax_dfun


# %%
def my_linear_coupling(cvar_state, weights, params):
    """A user-written coupling -- deliberately not imported from
    lyapax.coupling, to prove the extension point needs no library
    changes and no registration step."""
    G = params["G"]
    return G * jnp.einsum("ts,cs->ct", weights, cvar_state)


# %%
weights = np.array([[0., 1.], [1., 0.]])
gamma, G = -1.5, 0.7
A = gamma * np.eye(2) + G * weights
expected = np.sort(np.linalg.eigvalsh(A))[::-1]

model = ModelSpec(
    name="linear_node",
    state_variables=(StateVar("x", default_init=0.0),),
    parameters=(Parameter("gamma", gamma),),
    cvar=("x",),
    dfun_str={"x": "gamma * x + c"},
)
dfun = build_jax_dfun(model)
params = {"gamma": gamma, "G": G}
dt = 1e-3
step = make_network_step_fn(
    dfun, jnp.array(weights), model.cvar_indices, params, dt,
    coupling_fn=my_linear_coupling,  # <- user function, not a lyapax builder
)

result = lyapunov_spectrum(
    step, state0=jnp.array([0.2, -0.3]),
    dt=dt, n_steps=20_000, renorm_every=10, t_transient=5.0,
)

estimate = np.array(result.exponents)
print(f"exact eigenvalues:  {expected}")
print(f"lyapax estimate:    {estimate}")
print(f"max abs error:      {np.max(np.abs(estimate - expected)):.2e}")
