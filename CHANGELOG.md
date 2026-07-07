# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- README no longer claims DDE delays are integer-step only; it now
  documents both the default grid-snapped mode and the
  `interpolate=True` Hermite-interpolated mode.

### Added

- `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`,
  GitHub issue templates, and a pull request template.

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

[Unreleased]: https://github.com/Ziaeemehr/lyapax/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ziaeemehr/lyapax/releases/tag/v0.1.0
