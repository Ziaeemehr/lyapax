"""M6 GPU smoke test: confirms lyapax actually runs, and gets the same
answers, on GPU -- not just that `jax.devices()` lists one.

Skipped by default: `tests/conftest.py` forces `JAX_PLATFORMS=cpu` via
`os.environ.setdefault`, which only takes effect if `JAX_PLATFORMS` isn't
already set. Opt in by setting it yourself before pytest starts, e.g.:

    JAX_PLATFORMS=cuda pytest tests/test_gpu.py -v

M0 recorded this dev machine's GPU failing on any real op with
`INTERNAL: RET_CHECK ... dnn_support != nullptr` (cudnn/driver mismatch)
even though `jax.devices()` listed a `CudaDevice` -- exactly the kind of
"looks present but silently broken" failure a device-count check alone
would miss, hence running the full Lyapunov pipeline here rather than
just asserting a GPU device exists.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lyapax import systems
from lyapax.core import lyapunov_spectrum
from lyapax.coupling import kuramoto_coupling
from lyapax.integrators import rk4_step
from lyapax.network import make_network_step_fn
from lyapax.simulator import ModelSpec, Parameter, StateVar, build_jax_dfun

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="GPU not selected -- run with JAX_PLATFORMS=cuda to opt in",
)


def test_lorenz_lambda1_matches_published_value_on_gpu():
    # Same system, same tolerances as
    # test_lyapunov_core.test_lorenz_lambda1_matches_published_value --
    # the point is reproducing that CPU result on GPU, not a new check.
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rhs = systems.lorenz(sigma, rho, beta)
    dt = 1e-2
    step = rk4_step(rhs, dt)

    result = lyapunov_spectrum(
        step, state0=jnp.array([1.0, 1.0, 1.0]),
        dt=dt, n_steps=50_000, renorm_every=10, t_transient=100.0,
    )

    assert result.exponents.devices() == {jax.devices("gpu")[0]}
    assert abs(float(result.exponents[0]) - 0.9056) < 0.08
    assert abs(float(result.exponents[1])) < 0.03


def _kuramoto_model(omega: float) -> ModelSpec:
    return ModelSpec(
        name="kuramoto",
        state_variables=(StateVar("theta", default_init=0.0),),
        parameters=(Parameter("omega", omega),),
        cvar=("theta",),
        dfun_str={"theta": "omega + c"},
    )


def test_large_network_matrix_free_path_runs_on_gpu():
    # Same shape as test_network.test_large_network_benchmark_scale (M6) --
    # confirms the jax.jvp/vmap matrix-free path, the actual M6 payload,
    # also runs correctly on GPU, and reports the wall-clock time so a
    # human can eyeball the CPU/GPU comparison (not a pass/fail assertion,
    # since GPU throughput is hardware- and warmup-dependent).
    n_nodes = 200
    weights = jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)
    model = _kuramoto_model(omega=1.0)
    dfun = build_jax_dfun(model)
    params = {"omega": jnp.linspace(-1.0, 1.0, n_nodes), "G": 1.0}
    dt = 1e-2
    step = make_network_step_fn(
        dfun, weights, model.cvar_indices, params, dt,
        coupling_fn=kuramoto_coupling(alpha=0.0),
    )
    state0 = jnp.linspace(0.0, 2 * jnp.pi, n_nodes, endpoint=False)

    result = lyapunov_spectrum(
        step, state0=state0, dt=dt, n_steps=2_000, k=5, renorm_every=10, t_transient=5.0,
    )
    jax.block_until_ready(result.exponents)

    t0 = time.perf_counter()
    result = lyapunov_spectrum(
        step, state0=state0, dt=dt, n_steps=2_000, k=5, renorm_every=10, t_transient=5.0,
    )
    jax.block_until_ready(result.exponents)
    elapsed = time.perf_counter() - t0
    print(f"\n200-node network, k=5, 2000 steps, GPU warm: {elapsed:.3f}s")

    assert result.exponents.devices() == {jax.devices("gpu")[0]}
    assert result.exponents.shape == (5,)
    assert np.all(np.isfinite(np.array(result.exponents)))
