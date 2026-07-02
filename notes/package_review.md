# Review of `lyapax`

Scope: this review covers the package code, tests, examples, notes, README, and packaging metadata in this repository. The `benchmarks/` directory is intentionally excluded.

Validation run: `python3 -m pytest` completed with `32 passed, 2 skipped` in 36.25 s. The skipped tests are GPU opt-in tests.

## 1. Scientific correctness

The main implemented algorithm is the Benettin/QR method for Lyapunov spectra. `lyapax.core.lyapunov_spectrum` propagates tangent vectors with JAX forward-mode JVPs, periodically applies QR, accumulates `log(abs(diag(R)))`, and divides by elapsed time. This is mathematically consistent with standard Benettin-style variational/tangent dynamics and QR reorthonormalization. The package does not implement the Wolf time-series reconstruction algorithm; it computes exponents from known differentiable dynamical systems.

For ODEs and maps, the public interface accepts a one-step differentiable map `state -> new_state`. For continuous flows, examples/tests wrap RHS functions in fixed-step RK4. The computed exponents are therefore exponents of the numerical time-`dt` map divided by `dt`, not of an adaptive or exact flow. That is scientifically acceptable when `dt` convergence is checked.

For fixed-delay DDEs, `lyapax.dde.lyapunov_spectrum_dde` treats `(state, ring_buffer)` as a finite-dimensional Markov state and differentiates through the ring-buffer step. This matches the discretized-map idea used for DDE Lyapunov spectra, but only for integer-step delays on a fixed grid. It is not an infinite-dimensional continuous-history DDE method with interpolation or a function-space inner product.

Criticism: DDE support is scientifically limited to integer-step delays and finite-dimensional ring-buffer discretizations. This matters because the spectrum depends on the history discretization and delay rounding. Impact: High for DDE users with physical delays not aligned to `dt`; Low for deliberately grid-aligned fixed-delay experiments. Improvement: expose the effective delay `tau_eff = tau_steps * dt`, document that spectra are for the discretized map, and add user-facing convergence examples at decreasing `dt`.

Criticism: the package sorts exponents only by the final finite-time estimate. This is standard for reporting, but near-degenerate spectra can swap order over time. Impact: Medium for interpreting convergence histories and covariant subspace alignment. Improvement: document that `history` columns are reordered by final order, and consider returning unsorted cumulative estimates or the final permutation.

## 2. Numerical implementation

The simple ODE integrator is classical fixed-step RK4 in `integrators.py`. Network and DDE paths use Euler or Heun inside `simulator.step.make_step_fn`, defaulting to Heun. Tangent vectors are propagated through the same one-step numerical map via `jax.jvp`, which is the correct discrete tangent map for the chosen integrator.

QR is performed every `renorm_every` raw steps. The implementation correctly takes `abs(diag(R))` before logging. It also propagates and renormalizes tangent vectors during the transient, which is a strong choice: it lets tangent bases align with Oseledets subspaces before accumulation and avoids transient tangent overflow.

Criticism: `renorm_every` lacks runtime diagnostics for overflow, underflow, or zero diagonal entries in `R`. This matters because `log(0)` or very small diagonal values silently create `-inf` or unstable estimates, especially for strongly contracting directions and DDE buffer dimensions. Impact: High for long DDE/full-spectrum runs; Medium for top-k chaotic ODE use. Improvement: optionally check `jnp.isfinite(log_growth)` and expose a `check_finite` or warning mode; document practical selection of `renorm_every`.

Criticism: stiff systems are not supported beyond fixed-step explicit RK4/Heun/Euler. This matters because Lyapunov exponents of stiff systems require stable primal integration and stable tangent propagation. Impact: High for stiff ODE/DDE models; Low for the current Lorenz/Rossler/map validation set. Improvement: document non-stiff fixed-step scope and consider a Diffrax ODE step adapter for non-DDE stiff/adaptive workflows.

Criticism: only one DDE test checks two `dt` values; most ODE chaotic-flow tests use a single `dt`. This matters because finite-time Lyapunov estimates can look plausible while still reflecting time-discretization error. Impact: Medium. Improvement: add regression tests or examples that compare Lorenz/Rossler and key DDE cases across at least two `dt` values.

## 3. JAX implementation

The implementation uses JAX idiomatically in the core numerical loops: `jax.jvp`, `jax.vmap`, and `jax.lax.scan` remove Python loops from the raw-step and renormalization loops. This is a strong design for XLA compilation and for partial spectra where `k << d`.

The code does not decorate public functions with `jax.jit`; instead it uses JAX primitives internally. This keeps the API simple and lets users choose when to JIT, but it also means default calls may include tracing/dispatch overhead. Parameter sweeps are implemented with `jax.vmap` in `sweep.py`.

Criticism: no `pmap`/`pjit` support or documented multi-device batching exists. This matters for large parameter sweeps or large networks. Impact: Low for current package maturity; Medium for high-throughput scientific workloads. Improvement: document that sweeps are single-device `vmap` today and add examples for `jit(vmap(...))`; consider `shard_map`/`pmap` later.

Criticism: memory for full DDE spectra scales with `(d_state + d_buf) * k`, and full-spectrum DDE defaults to `k=d_total`. This matters because large horizons or networks can make the tangent matrix and QR expensive. Impact: High for delayed networks. Improvement: make examples default to small `k`, warn in docs/API when `k=None` for large DDE augmented dimensions, and include complexity notes.

Criticism: `_read_delayed_coupling` has a Python loop over coupling variables when building a list inside traced code. The loop count is static and likely small, so this is not a correctness problem. Impact: Low to Medium if many coupling variables are used. Improvement: replace with a vectorized gather over the coupling-variable axis if multi-cvar systems become important.

## 4. API design

The core API is compact: `lyapunov_spectrum(step_fn, state0, dt, n_steps, k, renorm_every, t_transient, seed)` returns a `LyapunovResult` with `exponents`, `history`, and `times`. This is usable and extensible because users can provide arbitrary differentiable JAX step functions.

The coupling API is also extensible: a coupling is a plain callable `(cvar_state, weights, params) -> coupling`, and tests verify user-defined coupling functions. The network adapter cleanly bridges structured `(n_sv, n_nodes)` state to flat Lyapunov inputs.

Criticism: `src/lyapax/__init__.py` is stale and says the tangent/QR engine is not implemented. This matters because package-level documentation contradicts the actual package state. Impact: Medium documentation/API trust issue. Improvement: update the module docstring and export the main public functions/classes.

Criticism: public API exports are minimal; users must know submodule paths. This matters for usability and discoverability. Impact: Medium. Improvement: export `lyapunov_spectrum`, `lyapunov_spectrum_dde`, `LyapunovResult`, `rk4_step`, and common coupling builders from `lyapax.__init__` or document the intended import style.

Criticism: type annotations use broad `Callable` and `dict` in several scientific extension points. This matters because shape and parameter-pytree expectations are central to correct JAX use. Impact: Low to Medium. Improvement: add shape-oriented docstrings consistently and consider Protocols or type aliases for `StepFn`, `CarryStepFn`, `CouplingFn`, and parameter pytrees.

## 5. Software engineering

The project has a clean `src/` layout, focused modules, and tests organized by feature. The code is readable and the core numerical functions are small enough to audit. Reuse of `_run_renorm_scan` avoids duplicated QR accumulation logic between ODE and DDE paths.

Error handling exists for invalid `k`, invalid `n_steps`, and incompatible `renorm_every`. `Connectivity` validates shapes, nonnegative tract lengths, and positive speed.

Criticism: there is no visible CI workflow under `.github/workflows`. This matters because numerical packages need continuous validation across Python/JAX versions. Impact: High for open-source maintainability. Improvement: add GitHub Actions for lint/type/test on CPU with x64 enabled, plus optional GPU workflow if infrastructure is available.

Criticism: dynamic RHS generation uses `exec()` in `build_jax_dfun`. This is inherited/design-driven, but it matters for security and error diagnostics if users load untrusted model specs. Impact: Medium. Improvement: explicitly document that `dfun_str` is trusted code, validate allowed names more strictly, or replace string execution with an expression parser for public releases.

Criticism: no linting, formatting, or type-check configuration is present. This matters for contributor consistency. Impact: Low to Medium. Improvement: add `ruff` and optionally `mypy`/`pyright` in dev dependencies and CI.

## 6. Documentation

The notes are technically substantive. `notes/validation_systems.md` provides a serious validation plan with analytic references and structural invariants. `notes/milestones.md` captures design rationale and known risks.

The README is currently only a stub. It gives almost no installation, usage, API, or mathematical guidance. Examples exist, but documentation is not wired up.

Criticism: README documentation is insufficient for users. This matters because the package exposes nontrivial numerical assumptions: fixed-step maps, x64, finite-time convergence, DDE discretization, and `renorm_every`. Impact: High for adoption and correct use. Improvement: add installation commands, a minimal Lorenz example, a map example, a DDE example, expected outputs, and warnings about convergence/precision.

Criticism: mathematical documentation is mostly in internal notes and source comments, not public docs. This matters because users need to know what algorithm is implemented and what is not implemented. Impact: Medium. Improvement: create a docs page or README section titled "Method" explaining Benettin/QR, JVP tangent propagation, finite-time convergence, and fixed-delay DDE discretization.

## 7. Performance

For ODE/map systems, raw-step tangent propagation costs approximately `O(k)` JVPs per step plus QR every `renorm_every` steps. QR costs approximately `O(d k^2)` for partial spectra and `O(d^3)` when `k=d`. This is a good design for leading exponents in high-dimensional systems.

For DDEs, replace `d` with `d_total = d_state + d_buf`. This can be much larger than the physical state dimension, so top-k computation is essential.

Compared with dense NumPy/SciPy Jacobian methods, this JAX implementation should be more attractive for differentiable JAX models and partial spectra. Compared with DifferentialEquations.jl/ChaosTools.jl, it is less feature-rich: no adaptive ODE solvers, no mature DDE interpolation stack, and no broad ecosystem of validated algorithms. Compared with hand-coded variational equations, JAX JVP reduces derivative-maintenance risk.

Criticism: no public benchmark results are considered in this review, and performance claims in tests are mostly generous smoke ceilings. This matters because XLA compile time, warmup, device transfer, and shape specialization affect real workloads. Impact: Medium. Improvement: add non-benchmark documentation with complexity formulas and separate reproducible benchmark reports outside the core tests.

Criticism: public functions are not JIT-wrapped and no guidance is given on warmup/blocking. This matters because first-call timings can be misleading in JAX. Impact: Medium for performance users. Improvement: document `jax.jit(lambda ...: lyapunov_spectrum(...))` patterns where static arguments are handled correctly, and show `jax.block_until_ready` for timing.

## 8. Validation

Validation is a major strength. Tests cover:

- linear ODE spectra against eigenvalues,
- logistic and tent maps against `ln(2)`,
- Hénon map sum against `ln|b|`,
- Lorenz sum against constant divergence and largest exponent against a published value,
- Rössler qualitative chaotic spectrum,
- partial spectrum consistency,
- JVP tangent propagation against dense `jax.jacfwd`,
- linear networks against full Jacobian eigenvalues,
- custom coupling,
- DDE dominant roots using Lambert W,
- DDE `dt` convergence for one scalar case,
- Mackey-Glass qualitative chaos,
- delayed linear networks against Lambert W formulas,
- sweep equivalence against Python loops,
- opt-in GPU smoke tests.

Criticism: Rössler divergence identity described in `notes/validation_systems.md` is not implemented as a test; only qualitative ranges are checked. This matters because structural invariants are stronger than literature-range checks. Impact: Medium. Improvement: add a Rössler test computing the trajectory mean of `x` and checking `sum(lambda) = a - c + mean(x)`.

Criticism: GPU tests are skipped by default and not CI-backed. This matters because GPU compatibility is claimed only when explicitly selected. Impact: Medium. Improvement: keep CPU default, but add a documented optional GPU CI/job or periodic manual validation record.

## 9. Scientific reproducibility

The tests force CPU and enable x64 in `tests/conftest.py`, which is good for repeatability. Random tangent bases are seeded through `seed`, making initial tangent bases reproducible for fixed shape/dtype/backend.

Dependencies are minimally specified in `pyproject.toml`: `jax>=0.10`, `numpy>=1.23`, dev extras `pytest>=7`, `scipy>=1.10`. There is no lockfile.

Criticism: x64 is required by the tests and notes, but package imports do not enable or enforce it. This matters because ordinary users may run in JAX float32 defaults and get degraded long-horizon exponents. Impact: High. Improvement: document required `jax_enable_x64=True`, add a runtime warning when `state0.dtype` is float32, or provide an explicit setup helper.

Criticism: dependency versions are broad and unpinned. This matters because JAX behavior and linear algebra numerics can change across versions/backends. Impact: Medium. Improvement: keep broad install requirements but add a tested-version matrix or lockfile for reproducible development.

## 10. Strengths

- Scientifically appropriate Benettin/QR implementation for differentiable maps and fixed-step ODE numerical maps.
- Tangent propagation uses `jvp`/`vmap`, avoiding dense Jacobians for partial spectra.
- DDE implementation correctly differentiates the augmented `(state, buffer)` map rather than ignoring delayed-state sensitivities.
- Validation suite is unusually strong for an early scientific package.
- Running convergence history is returned, which supports finite-time diagnostics.
- Coupling API is open and tested with user-defined functions.
- Tests pass in the current environment.

## 11. Weaknesses

- README and public documentation are far behind the implementation.
- x64 requirement is enforced only in tests, not in normal package use.
- Fixed-step explicit integration limits applicability to stiff systems.
- DDE support is finite-dimensional, integer-delay, and fixed-step only.
- No CI workflow is present.
- Package-level `__init__.py` is stale and does not expose the useful API.

## 12. Major concerns

1. Bug/documentation issue: stale package docstring in `__init__.py`.
   Why it matters: it tells users the implemented Lyapunov engine does not exist.
   Impact: Medium.
   Improvement: update the docstring and public exports.

2. Numerical issue: x64 is not enforced or warned about outside tests.
   Why it matters: long Lyapunov averages are sensitive to precision.
   Impact: High.
   Improvement: add documentation plus runtime dtype warnings or explicit x64 setup guidance.

3. Numerical limitation: DDE delay rounding is central but not surfaced in the public API.
   Why it matters: users may interpret rounded-delay spectra as spectra for the exact physical delay.
   Impact: High.
   Improvement: return or expose `tau_steps`, `tau_eff`, and convergence guidance.

4. Software-engineering issue: no CI.
   Why it matters: regressions in numerical code can be subtle and backend/version dependent.
   Impact: High.
   Improvement: add CPU/x64 GitHub Actions running the full test suite.

## 13. Minor suggestions

- Style issue: several source comments refer to milestone history; useful for development, but verbose for a public package. Impact: Low. Improvement: move long design history into docs and keep code comments focused on current invariants.
- Documentation issue: README should link examples by task, not only say they exist. Impact: Medium. Improvement: add a short examples index.
- Enhancement opportunity: add a convergence helper that summarizes last-window drift in `history`. Impact: Low to Medium. Improvement: implement a small utility that reports relative/absolute finite-time change.
- API issue: `renorm_every <= 0` is not explicitly checked. Impact: Low because modulo/scan errors will occur, but messages will be poor. Improvement: add a direct validation error.

## 14. Possible future improvements

- Add Diffrax adapter support for ODEs, including adaptive and stiff solvers where appropriate.
- Add public docs with derivations, algorithm references, and limitations.
- Add optional finite checks and convergence diagnostics.
- Add vectorized per-coupling-variable delayed gather if multi-cvar delayed systems become common.
- Add pmap/pjit examples for large parameter sweeps.
- Add richer DDE interpolation support if exact non-integer delays become a project goal.
- Add package exports, API docs, and type Protocols for extension callables.

## 15. Overall assessment

`lyapax` is scientifically credible for its current intended scope: Lyapunov exponents of differentiable JAX maps, fixed-step ODE discretizations, networks, and fixed-grid DDE discretizations. The core Benettin/QR machinery is implemented correctly, tangent propagation is modern and efficient for partial spectra, and the validation suite is much stronger than the public-facing documentation.

The main risks are not in the central QR/JVP algorithm. They are in user-facing reproducibility and scope clarity: x64 must be made explicit, fixed-step and integer-delay limitations must be documented prominently, and CI should be added. With those addressed, this would be a solid early-stage open-source scientific package.
