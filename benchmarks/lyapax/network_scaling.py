"""Tier 5: dense (all-to-all) Kuramoto network at increasing size, k=5
partial spectrum -- the same growing-network system as
examples/10_matrix_free_scaling.py / examples/14_gpu_acceleration.py, run
here across tools instead of just lyapax CPU vs GPU.

Point of this tier: lyapax's cost is dominated by k (tangent directions
tracked via jax.jvp), not d (network size) -- see
lyapax.core.lyapunov_spectrum's docstring. ChaosTools.jl's tangent
propagation instead forms the full d x d Jacobian via ForwardDiff every
step regardless of k (confirmed directly: d=50 -> 13.5s, d=200 -> 137s,
d=1000 exceeded 14 minutes and was abandoned -- see
benchmarks/chaostools/network_scaling.jl's comment), so this tier is
deliberately run at d=50/200/1000/2000 for lyapax but only d=50/200 for
ChaosTools.jl and d=50 for jitcode (see those scripts' own comments for
why larger sizes aren't attempted there).
"""
from _common import time_and_run, emit  # noqa: I001 -- must set JAX_PLATFORMS before jax import

import jax.numpy as jnp

from lyapax.core import lyapunov_spectrum
from lyapax.coupling import kuramoto_coupling
from lyapax.network import Network, network_problem
from lyapax.simulator import ModelSpec, Parameter, StateVar, build_jax_dfun


def run(n_nodes: int):
    weights = jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)
    model = ModelSpec(
        name="kuramoto", state_variables=(StateVar("theta", default_init=0.0),),
        parameters=(Parameter("omega", 0.0),), cvar=("theta",),
        dfun_str={"theta": "omega + c"},
    )
    dfun = build_jax_dfun(model)
    params = {"omega": jnp.linspace(-1.0, 1.0, n_nodes), "G": 1.0}
    dt = 1e-2
    network = Network(weights=weights, cvar_indices=model.cvar_indices)
    state0 = jnp.linspace(0.0, 2 * jnp.pi, n_nodes, endpoint=False)
    problem = network_problem(
        dfun, network, kuramoto_coupling(alpha=0.0),
        params=params, state0=state0, dt=dt, integrator="rk4",
    )
    return lyapunov_spectrum(problem, n_steps=200, k=5, renorm_every=10, t_transient=0.0)


if __name__ == "__main__":
    for n_nodes in (50, 200, 1000, 2000):
        first_s, warm_s, result = time_and_run(run, n_nodes)
        emit("lyapax", f"kuramoto_scaling_d{n_nodes}", result.exponents, first_s, warm_s)
