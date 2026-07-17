<p align="center">
  <img src="docs/_static/lyapax_logo.png" alt="LYAPAX logo" width="260">
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
pip install -e ".[adaptive]" # + diffrax, for lyapax.adaptive
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

sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

def rhs(state):
    x, y, z = state
    return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

problem = ode_problem(rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=1e-2)

result = lyapunov_spectrum(
    problem, n_steps=50_000, renorm_every=10, t_transient=100.0,
)
print(result.exponents)  # ~ [0.906, 0.0, -14.57]
```

`result.history` gives the running per-column estimate at each
renormalization point, for checking convergence.

`rhs` is just a plain `state -> jnp.ndarray` function - write your own for
any system. `lyapax.systems` has this Lorenz system and other standard
test systems (Rössler, Kuramoto, ...) prebuilt, if you'd rather not retype
them.

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

- **No stiff (implicit) ODE solvers.** `step_fn` is whatever integrator
  you hand it - a fixed-step map (`ode_problem(..., integrator="rk4")` by
  default, direct `rk4_step`/`euler_step` step functions, or Euler/Heun/RK6
  via `lyapax.simulator`), or an adaptive-step *explicit* one via the
  optional `adaptive` extra (`lyapax.adaptive.diffrax_adaptive_step`,
  backed by diffrax; `ode_problem` only - not `network_problem` or DDEs).
  Exponents are for the numerical time-`dt` map (or accepted-step sequence,
  for the adaptive integrator), not an exact flow - check `dt`/tolerance
  convergence for anything you report. The adaptive integrator is for
  tolerance-driven accuracy control, not speed - measured 2-4x slower than
  `rk4`/`rk6` at matched accuracy, not faster; see
  `docs/background/capabilities.md`.
- **DDE delays must be known and fixed**, resolved one of two ways. By
  default (grid-snapped), a physical delay `tau` is rounded to the
  nearest multiple of `dt` (`lyapax.dde.resolve_tau_steps`); use
  `lyapax.dde.tau_eff` to see the delay actually used, and shrink `dt` to
  converge it toward `tau`. Passing `interpolate=True` instead
  reconstructs the delayed history with a cubic-Hermite interpolant,
  removing that rounding on the uniform-delay path at some extra cost - see
  `13_dde_history_interpolation.py` and `docs/background/lyapax_implementation.md`.
- **`history` columns are ordered once**, by the final row. Near-
  degenerate exponents can swap order over the run - see
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
| `15_adaptive_ode.py` | Adaptive-step ODE integration via the optional `diffrax` backend |
| `16_convergence_drift.py` | Chunked run-inspect-resume loop with `convergence_drift` |
| `17_dde_resume.py` | Resuming a checkpointed DDE run |
| `18_differentiate_lyapunov_exponent.py` | Differentiating an exponent w.r.t. a system parameter, and where that breaks down |
| `19_kaplan_yorke_dimension.py` | Kaplan-Yorke dimension from a Lyapunov spectrum |

## Building the docs

```bash
pip install -e ".[docs,adaptive]"
sphinx-build -b html docs docs/_build/html
```

This re-runs every numbered `examples/*.py` file to render it into a
sphinx-gallery gallery (code, output, and plots), including
`15_adaptive_ode.py` (hence the `adaptive` extra above), alongside the
API reference; open `docs/_build/html/index.html` when it's done.

## Further reading

- `docs/background/validation.md` - the correctness tests this package is
  held to (exact values, structural invariants, literature figures).
- `docs/background/benchmarks.md` - comparison against `jitcode`/`jitcdde` and
  `ChaosTools.jl`.
- `docs/background/capabilities.md` - a candid summary of what lyapax does
  and does not do, including measured performance characteristics.
