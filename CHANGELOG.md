# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-08-04

### Added

- `save_states`/`decimate` on `lyapunov_spectrum` and
  `lyapunov_spectrum_dde`: optionally emit the state time series the
  Lyapunov accumulation ran on (`result.trajectory`/`result.save_times`),
  from the same scan that computes the exponents -- no extra dynamics
  cost, and the states line up exactly with the run the exponents came
  from. Off by default (`save_states=False`); existing calls and
  `LyapunovResult` positional unpacking are unaffected.
- `lyapax.sweep.sweep_lyapunov_spectrum` gains the same
  `save_states`/`decimate`, plus `batch_size` (run a vmapped sweep in
  memory-bounded chunks instead of one call over the whole grid) and
  `trajectory_to_host` (park each chunk's trajectory on host via
  `jax.device_get` before the next chunk starts).
- `lyapax.sweep.sweep_trajectory`: the replay-only counterpart (via
  `lyapax.utils.simulate_trajectory`, batched with `jax.vmap`) -- a
  batched time series without the tangent-propagation/QR cost of
  computing exponents, landing on the same post-transient window and
  decimation stride as `sweep_lyapunov_spectrum(..., save_states=True)`.
- `lyapax.utils.simulate_trajectory` gains `decimate` and `n_transient`
  (both default to the previous behavior).
- `examples/20_sweep_time_series_features.py`: derives the Kuramoto order
  parameter from a swept trajectory alongside the swept Lyapunov spectrum.
- `examples/21_kuramoto_le_and_time_series.py`: a minimal, fast
  `sweep_lyapunov_spectrum(..., save_states=True)` demo plotting one
  sweep point's saved phase time series next to its Lyapunov exponent
  convergence.
- `sweep_lyapunov_spectrum` warns (does not raise) when an unchunked
  (`batch_size=None`) `save_states=True` call is estimated to materialize
  more than ~1 GiB of trajectory on-device at once, naming `batch_size`
  and `decimate` as the remedies.

## [0.2.0] - 2026-07-15

### Fixed

- README no longer claims DDE delays are integer-step only; it now
  documents both the default grid-snapped mode and the
  `interpolate=True` Hermite-interpolated mode.
- `lyapax.adaptive.diffrax_adaptive_step` no longer silently returns a
  truncated state when its internal `max_steps` is exhausted; it now
  raises (via `equinox.error_if`, working under `jit`/`vmap` too) instead
  of letting `lyapunov_spectrum` silently mis-associate an exponent with
  the wrong elapsed time.
- `lyapunov_spectrum`/`lyapunov_spectrum_dde` now reject `resume=` calls
  whose `dt` doesn't match the checkpointed run's `dt`, instead of only
  validating shapes/tracked dimensions.
- CI's test job now installs the `adaptive` extra, so the adaptive-ODE
  test modules actually run as tests in the matrix instead of only being
  exercised indirectly via the docs build's Sphinx-Gallery execution.
- `docs/api.rst` now includes `lyapax.adaptive`; README's example table,
  development-install instructions, and docs-build command now cover
  demos 15-19 and the `adaptive` extra, matching CI.
- The `ValueError` raised when an adaptive integrator is passed to a
  network/DDE problem no longer says "not supported for DDEs" (it fires
  for any `network_problem`, delayed or not) - corrected to describe the
  actual restriction (adaptive integration works only through a single,
  uncoupled `ode_problem`).
- Docstrings, error messages, README, `pyproject.toml` comments, test
  module docstrings, and the `benchmarks/` scripts no longer reference the
  repository's internal `notes/` directory (not part of the published
  package or docs); those pointers now go to the published `docs/background/`
  pages, proper Sphinx cross-references, or were simply dropped where they
  cited an internal-only design-history note with no public equivalent.

### Added

- A "capabilities and limitations" doc section
  (`background/capabilities`) characterizing `lyapax.adaptive`'s actual
  wall-clock performance: measured 2-4x slower than fixed-step `rk4`/`rk6`
  at matched accuracy (not matched `dt`/`rtol`) on every system tested,
  including a relaxation oscillator and large (`d=1500-3000`) systems - it's
  a tolerance-driven-accuracy and forward-mode-differentiability tool, not
  a speed optimization.
- `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`,
  GitHub issue templates, and a pull request template.
- Optional adaptive-step ODE integration via `lyapax.adaptive`
  (`diffrax_adaptive_step`, the `adaptive` extra), supporting forward-mode
  differentiation (`jax.jacfwd`/`jax.jvp`) but not reverse mode.
- `interpolate=True` cubic-Hermite DDE history reconstruction, alongside
  the existing grid-snapped mode, with second-order (Heun) and
  approximately fourth-order (RK4) accuracy on a scalar analytic case.
- `result.checkpoint` / `resume=` support for continuing a
  `lyapunov_spectrum`/`lyapunov_spectrum_dde` run without restarting, and
  `lyapax.core.convergence_drift` for judging whether a fixed-`n_steps`
  run has settled.
- Differentiability guidance and tests for `lyapunov_spectrum` w.r.t.
  system parameters (`jax.grad`/`jax.jacfwd`), including the documented
  limitation that gradients through a genuinely chaotic trajectory
  inherit its exponential sensitivity and are not reliable long-time
  estimates.
- `lyapax.core.kaplan_yorke_dimension` for computing the Kaplan-Yorke
  (Lyapunov) dimension from a spectrum, with a partial-spectrum guard.

## [0.1.0] - 2026-07-03

Initial release.

### Added

- Lyapunov spectrum computation for ODEs and iterated maps via the
  Benettin/QR method, with matrix-free tangent propagation using
  `jax.jvp` and `jax.vmap`.
- Partial-spectrum support (`k < d`) with cost scaling in `k`, not `d`.
- Coupled-network support (`network_problem`) with built-in linear,
  sigmoidal, and Kuramoto coupling, plus user-defined coupling
  callables.
- Fixed-delay DDE support (`dde_problem`, `network_dde_problem`,
  `lyapunov_spectrum_dde`) via an augmented state/ring-buffer carry,
  with grid-snapped and Hermite-interpolated (`interpolate=True`)
  history modes.
- Batched parameter/initial-condition sweeps via `jax.vmap`
  (`sweep_lyapunov_spectrum`).
- Fixed-step Euler/Heun/RK4/RK6 integrators.
- `ModelSpec`/`build_jax_dfun` for building JAX right-hand sides from
  symbolic expressions.
- 15 runnable, Sphinx-Gallery-formatted examples under `examples/`.
- Sphinx documentation, including background pages on the algorithm,
  capabilities/limitations, and validation.
- Validation against exact eigenvalues, structural invariants (e.g.
  Hénon map sum-of-exponents), literature values, and independent tools
  (`jitcode`, `jitcdde`, `ChaosTools.jl`).
- GitHub Actions CI (pytest + ruff on Python 3.11/3.12) and a PyPI
  trusted-publishing release workflow.

[Unreleased]: https://github.com/Ziaeemehr/lyapax/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Ziaeemehr/lyapax/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Ziaeemehr/lyapax/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ziaeemehr/lyapax/releases/tag/v0.1.0
