"""
GPU acceleration: when does it actually pay off?
================================================================

``lyapax`` runs on GPU with no code changes -- JAX picks its backend
(CPU, GPU, ...) from an environment variable at startup, and none of
``lyapax``'s own functions know or care which one is active. But "runs on
GPU" and "faster on GPU" are two different claims. A GPU wins on
*throughput*: it does many floating-point operations in parallel, at the
cost of a fixed per-call overhead (kernel launch, host<->device data
transfer) that a CPU doesn't pay. For a small problem -- a state vector
with only a handful of entries, say -- that fixed overhead dwarfs the
actual arithmetic, and the CPU wins outright. Once the per-step arithmetic
is large enough to amortize the overhead, the GPU pulls ahead, often by a
large margin.

**The comparison.** The same growing all-to-all Kuramoto network as
:ref:`10_matrix_free_scaling.py <sphx_glr_auto_examples_10_matrix_free_scaling.py>`
(``d`` = ``n_nodes``, one phase per
node, dense ``d x d`` coupling weights), timed end-to-end through
``lyapax.core.lyapunov_spectrum`` at a fixed ``k=5`` and increasing ``d``,
once on CPU and once on GPU. Growing ``d`` grows the coupling matrix's
element count as ``d**2``, so the per-step arithmetic (and the case for a
GPU) grows quickly even though ``k`` stays fixed.

**Why this script re-launches itself as a subprocess.** JAX commits to one
backend for the lifetime of a process -- it reads ``JAX_PLATFORMS`` the
first time any JAX op runs and can't switch afterwards. To get a genuine
CPU number *and* a genuine GPU number in one script, each timed run below
happens in its own fresh subprocess with ``JAX_PLATFORMS`` set to the
backend being measured for that run -- the same approach
``benchmarks/collect_results.py`` uses so its CPU and GPU benchmark rows
don't silently collide into one.

The documentation page uses the stored reference figure below, generated
once on a GPU-capable machine, rather than re-running this benchmark on
the documentation builder.

.. image:: ../_static/gpu_acceleration_reference.png
   :alt: Reference CPU/GPU timing sweep for demo 14
   :align: center

If this machine has no working GPU (or JAX can't see one), local script
runs do not fabricate GPU timings. The live figure marks the GPU backend
as unavailable instead.
"""
# %%
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_path = "../docs/_static/gpu_acceleration_reference.png"

try:
    THIS_FILE = Path(__file__).resolve()
    SPHINX_GALLERY_RUN = False
except NameError:
    # sphinx-gallery exec()s examples without __file__, but chdirs into the
    # example's directory first, so the filename alone still resolves.
    THIS_FILE = (Path.cwd() / "14_gpu_acceleration.py").resolve()
    SPHINX_GALLERY_RUN = True

REFERENCE_FIGURE = THIS_FILE.parents[1] / "docs" / "_static" / "gpu_acceleration_reference.png"


def _run_lyapunov_spectrum(backend: str, n_nodes: int, n_steps: int) -> None:
    """Worker body: build the network, time one lyapunov_spectrum call,
    print {"backend", "elapsed_s"} as JSON. Runs in its own subprocess
    (see _time_backend below) so JAX_PLATFORMS is fixed before JAX loads.
    """
    os.environ["JAX_PLATFORMS"] = "cuda" if backend == "gpu" else "cpu"
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    from lyapax.core import lyapunov_spectrum
    from lyapax.coupling import kuramoto_coupling
    from lyapax.network import Network, network_problem
    from lyapax.simulator import ModelSpec, Parameter, StateVar, build_jax_dfun

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
        params=params, state0=state0, dt=dt,
    )

    # Warmup: pay JIT tracing/compilation, and for GPU the first
    # host->device transfer, once, outside the timed call.
    warm = lyapunov_spectrum(
        problem, n_steps=10, k=5, renorm_every=5, t_transient=0.0,
    )
    jax.block_until_ready(warm.exponents)

    t0 = time.perf_counter()
    result = lyapunov_spectrum(
        problem, n_steps=n_steps, k=5, renorm_every=10, t_transient=0.0,
    )
    jax.block_until_ready(result.exponents)
    elapsed = time.perf_counter() - t0

    print(json.dumps({"backend": jax.default_backend(), "elapsed_s": elapsed}))


if globals().get("__name__") == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
    _run_lyapunov_spectrum(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    sys.exit(0)


# %%
def _time_backend(backend: str, n_nodes: int, n_steps: int) -> dict | None:
    """Run one (backend, n_nodes) timing in a fresh subprocess (see the
    docstring above for why) and return its {"backend", "elapsed_s"}
    payload, or None if that backend isn't actually usable here.
    """
    proc = subprocess.run(
        [sys.executable, str(THIS_FILE), "--worker", backend, str(n_nodes), str(n_steps)],
        capture_output=True, text=True, timeout=180,
    )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return None


# %%
if SPHINX_GALLERY_RUN:
    print(f"Using stored reference figure: {REFERENCE_FIGURE}")
else:
    node_sizes = [20, 50, 100, 200, 500, 1000, 2000]
    n_raw_steps = 200

    gpu_probe = _time_backend("gpu", n_nodes=10, n_steps=10)
    has_gpu = gpu_probe is not None and gpu_probe["backend"] == "gpu"
    if not has_gpu:
        print("No working GPU backend found (JAX_PLATFORMS=cuda probe failed) -- "
              "GPU timings will be marked unavailable in the plot. Re-run on a "
              "machine with a working CUDA/JAX GPU setup to see the CPU/GPU "
              "crossover.")

    cpu_times, gpu_times = [], []
    for n_nodes in node_sizes:
        cpu = _time_backend("cpu", n_nodes, n_raw_steps)
        cpu_times.append(cpu["elapsed_s"])
        msg = f"d={n_nodes:5d}  cpu={cpu_times[-1]:7.3f}s"
        if has_gpu:
            gpu = _time_backend("gpu", n_nodes, n_raw_steps)
            gpu_times.append(gpu["elapsed_s"])
            msg += f"  gpu={gpu_times[-1]:7.3f}s  speedup={cpu_times[-1] / gpu_times[-1]:5.2f}x"
        print(msg)

    # %%
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(node_sizes, cpu_times, "o-", label="CPU")
    if has_gpu:
        ax.plot(node_sizes, gpu_times, "s-", label="GPU")
    else:
        ax.plot([], [], "s--", color="C1", label="GPU unavailable on this build")
        ax.text(
            0.04, 0.96,
            "GPU timings not measured\n(no CUDA backend available)",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
    ax.set_xlabel("network size d (n_nodes)")
    ax.set_ylabel(f"wall time for {n_raw_steps} raw steps, k=5 (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("lyapunov_spectrum wall time vs network size: CPU vs GPU")
    ax.legend()
    fig.tight_layout()
    plt.show()

# %%
# The dense d x d coupling matrix means the per-step arithmetic grows like
# d**2 as the network gets bigger, even though only k=5 Lyapunov exponents
# are tracked throughout. On a machine with a working GPU, expect the two
# curves to start close together (small d: both pay roughly the same fixed
# overhead) and then diverge sharply as d grows -- the CPU curve bending
# upward faster than the GPU curve, which is the point where moving a
# lyapax computation to GPU is worth doing. Where exactly that crossover
# falls depends on the specific CPU and GPU in the machine running this
# script, which is why this example measures it rather than asserting a
# fixed threshold.
