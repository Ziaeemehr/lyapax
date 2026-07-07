<p align="center">
  <img src="docs/_static/lyapax_logo.svg" alt="LYAPAX logo" width="520">
</p>

# lyapax

[![CI](https://github.com/Ziaeemehr/lyapax/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ziaeemehr/lyapax/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation Status](https://readthedocs.org/projects/lyapax/badge/?version=latest)](https://lyapax.readthedocs.io/en/latest/?badge=latest)

JAX-native Lyapunov exponent computation for ODEs and DDEs, via the
Benettin/QR method with `jax.jvp`/`jax.vmap` tangent propagation.

## Install

```bash
pip install lyapax
```

For development (running the test suite or examples), install from a clone instead:

```bash
pip install -e ".[dev]"      # core + pytest/scipy for the test suite
pip install -e ".[examples]" # + matplotlib, to run examples/
pip install -e ".[docs]"     # + sphinx/sphinx-gallery, to build docs/
```

Requires `jax>=0.10`, Python `>=3.11`.

**Enable float64 before anything else.** Lyapunov exponents are averages of
log-growth rates accumulated over many steps; JAX's default float32
silently degrades long-horizon estimates. Do this first, before creating
any `jax.numpy` arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

`lyapunov_spectrum`/`lyapunov_spectrum_dde` warn if called with a float32
`state0` and x64 is off.

## Minimal example

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lyapax import lyapunov_spectrum, ode_problem
from lyapax import systems

rhs = systems.lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
problem = ode_problem(rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=1e-2)

result = lyapunov_spectrum(
    problem, n_steps=50_000, renorm_every=10, t_transient=100.0,
)
print(result.exponents)  # ~ [0.906, 0.0, -14.57]
```

`result.history` gives the running per-column estimate at each
renormalization point, for checking convergence.

## Method and scope

`lyapunov_spectrum` propagates a `(d, k)` matrix of tangent vectors
alongside the trajectory via `jax.jvp` (one forward-mode pass per tracked
column, batched with `jax.vmap`), QR-decomposes it every `renorm_every`
steps, accumulates `log|diag(R)|`, and divides by elapsed time
(Benettin's method). `k` defaults to the full spectrum (`k = d`); cost
scales with `k`, not `d`, so partial spectra of high-dimensional systems
are cheap.

`lyapunov_spectrum_dde` generalizes this to fixed-delay DDEs by
differentiating through an augmented `(state, ring_buffer)` carry.

**What this does *not* do:**

- **No adaptive/stiff ODE solvers.** `step_fn` is whatever fixed-step map
  you hand it (`ode_problem(..., integrator="rk4")` by default, direct
  `rk4_step`/`euler_step` step functions, or Euler/Heun/RK6 via
  `lyapax.simulator`). Exponents are for the numerical time-`dt` map, not
  an exact flow — check `dt`-convergence for anything you report.
- **DDE delays are integer-step only.** A physical delay `tau` is rounded
  to the nearest multiple of `dt` (`lyapax.dde.resolve_tau_steps`); there
  is no sub-step interpolation. Use `lyapax.dde.tau_eff` to see the delay
  actually used, and shrink `dt` to converge it toward `tau`.
- **`history` columns are ordered once**, by the final row. Near-
  degenerate exponents can swap order over the run — see
  `LyapunovResult.history`'s docstring.

## Examples

Runnable, sphinx-gallery-formatted demos in `examples/` (`pip install
-e ".[examples]"`, then `python examples/01_linear_ode.py`):

| File | Task |
|---|---|
| `00_time_series_sanity_check.py` | Inspecting the raw simulated time series before computing exponents |
| `01_linear_ode.py` | Exact-eigenvalue sanity check for the QR engine |
| `02_chaotic_maps.py` | 1D/2D maps with closed-form exponents |
| `03_chaotic_flows.py` | Lorenz/Rössler, the standard chaotic-ODE benchmarks |
| `04_linear_network.py` | Coupled network vs. exact Jacobian eigenvalues |
| `05_kuramoto_sync.py` | Kuramoto network synchronization transition |
| `06_custom_coupling.py` | Writing a custom coupling function |
| `07_speed_and_accuracy.py` | Cost/accuracy tradeoffs across settings |
| `08_delayed_coupling.py` | Two-node linear DDE network vs. delay |
| `09_kuramoto_delayed_network.py` | Effect of transmission delay on a Kuramoto network |
| `10_matrix_free_scaling.py` | Dense `jacfwd` vs. `jvp`/`vmap`, and why it matters for partial spectra |
| `11_vmap_parameter_sweep.py` | Batched parameter sweeps via `jax.vmap` |
| `12_public_api_overview.py` | Problem-object API for ODEs, networks, and DDEs |
| `13_dde_history_interpolation.py` | Grid-snapped vs. Hermite-interpolated DDE history reads |
| `14_gpu_acceleration.py` | When GPU execution pays off for larger Lyapunov workloads |

## Building the docs

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

This re-runs every numbered `examples/*.py` file to render it into a
sphinx-gallery gallery (code, output, and plots), alongside the API
reference; open `docs/_build/html/index.html` when it's done.

## Further reading

- `notes/milestones.md` — design history and open risks.
- `notes/validation_systems.md` — the correctness tests this package is
  held to (exact values, structural invariants, literature figures).
- `notes/benchmark_report.md` — comparison against `jitcode`/`jitcdde` and
  `ChaosTools.jl`.
