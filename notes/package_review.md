# Review of `lyapax`

Scope: this review covers the package code, tests, examples, public documentation,
notes, CI, and packaging metadata in this repository, most recently amended on
2026-07-15 to address the adaptive `max_steps`, checkpoint-resume, adaptive CI
coverage, and documentation-lag findings from the prior pass (commit `9c72dad`).
The `benchmarks/` directory is considered only where its published results affect
user-facing claims; benchmark implementation details are otherwise excluded.

Validation run: `python -m pytest -v` (installed with the `dev,adaptive` extras,
matching CI's updated test job) collected 84 tests and completed with `82 passed,
2 skipped` in ~90 s. The two skips are GPU tests, opt-in via `JAX_PLATFORMS=cuda`;
the Diffrax-dependent adaptive test modules now run rather than skip, since CI's
`test` job installs the `adaptive` extra. CI covers Python 3.11 and 3.12.

## 1. Scientific correctness

The central algorithm remains the Benettin/QR method. `lyapax.core.lyapunov_spectrum`
propagates tangent columns with JAX forward-mode JVPs, periodically applies QR,
accumulates `log(abs(diag(R)))`, and divides by elapsed time. It computes exponents
from a known differentiable dynamical system; it is not a Wolf-style estimator from
an observed scalar time series.

For maps and fixed-step ODEs, the reported values are exponents of the numerical
time-`dt` map divided by `dt`. The newer Diffrax adapter preserves this outer-map
contract while using adaptive internal steps over each interval of length `dt`.
Neither route makes the result an exact-flow exponent: fixed-step users must check
`dt` convergence and adaptive users must check tolerance convergence.

For DDEs, `lyapunov_spectrum_dde` differentiates the augmented `(state, ring_buffer)`
map, including delayed-history sensitivities. Two history modes now exist. The
default grid-snapped mode rounds a physical delay to the grid and remains first-order
in delay representation. `interpolate=True` uses cubic-Hermite history reconstruction
and stage-specific delayed reads; tests demonstrate second-order Heun behavior and an
approximately fourth-order RK4/Hermite accuracy ceiling on a scalar analytic case.
This is a substantial improvement over the original integer-step-only scope, but it
is still a finite-dimensional discretization of a fixed-delay DDE, not a general
function-space DDE solver.

Kaplan-Yorke dimension is now provided as post-processing of a sorted spectrum. Its
`d_total` guard correctly refuses to understate the dimension when a tracked partial
spectrum has not yet reached the cumulative-sum crossing.

Criticism: interpolated history is limited to the uniform-delay path, requires
`tau >= dt`, and does not support state-dependent or distributed delays. Per-edge
heterogeneous delays remain grid-based and tied to built-in linear coupling. Impact:
High for general DDE users; Low for the documented fixed, uniform-delay use case.
Improvement: keep these restrictions prominent in API docs and add interpolation to
the per-edge path only if a concrete scientific use case justifies its complexity.

Criticism: `history` columns are sorted once by the final finite-time estimate. Near-
degenerate exponents can exchange order during a run, so a column is not a persistent
mode identity. Impact: Medium for interpreting convergence histories and tangent
subspaces. Improvement: continue documenting this explicitly and consider exposing
the unsorted history or final permutation.

## 2. Numerical implementation

Fixed-step ODE integration includes Euler, Heun, RK4, and RK6. Coupled-network stages
now recompute coupling at each Runge-Kutta stage. Interpolated DDE stages likewise
read delayed history at the stage time, and the ring-buffer write convention has a
direct regression test. Tangent vectors are propagated through exactly the same
numerical map as the primal state via `jax.jvp`.

QR is applied every `renorm_every` raw outer steps. Tangent vectors are also
renormalized during transients, allowing alignment toward Oseledets subspaces without
unbounded growth. DDE transients cover at least one full ring-buffer cycle. The core
now validates `renorm_every >= 1`, warns when float32 is used with x64 disabled, and
offers eager-only `check_finite=True` diagnostics for non-finite histories.

Adaptive ODE integration is available through the optional
`lyapax.adaptive.diffrax_adaptive_step`. It uses a Diffrax solver with
`scan_kind="lax"`, a PID controller, and a bounded dynamic `lax.while_loop` inside
each outer `dt` interval. Tests compare it with published Lorenz behavior, fixed-step
RK4, tolerance changes, an exact linear tangent action, and finite-difference
parameter derivatives.

Resolved: reaching adaptive `max_steps` now raises via `equinox.error_if` (works
eagerly and under `jax.jit`/`jax.vmap`) instead of silently returning a truncated
state, closing the exponent/elapsed-time mismatch this previously risked. Covered
by `test_max_steps_exhaustion_raises` (`tests/test_adaptive_ode.py`), which forces
the condition deterministically with `max_steps=0`.

Criticism: there is still no implicit/stiff solver path, and adaptive integration is
ODE-only. Impact: High for stiff or multi-timescale systems; Low for the current
Lorenz/Rossler/map validation set. Improvement: either document explicit-only scope
as a firm boundary or add a tested implicit Diffrax adapter whose JVP behavior is
verified independently.

Criticism: broad finite checks occur after the complete run rather than at the QR
block that first generated invalid growth. Impact: Medium for diagnosing long runs.
Improvement: return block-level status or add checkify-based diagnostics if compatible
with the intended JIT workflow.

## 3. JAX implementation and differentiability

The numerical loops use `jax.jvp`, `jax.vmap`, `jax.lax.scan`, and JAX QR operations.
This avoids dense Jacobian construction and makes leading-`k` spectra practical when
`k << d`. Parameter sweeps use `vmap`; GPU execution follows normal JAX backend
selection. Public functions are not themselves decorated with `jax.jit`, leaving
compilation policy to callers and examples.

The fixed-step engine can be differentiated through with `jax.grad` or `jax.jacfwd`.
Tests establish the correct analytic derivative for a non-chaotic linear system.
They also correctly characterize the crucial limitation: differentiating the finite-
trajectory estimator through a chaotic trajectory inherits exponential trajectory
sensitivity, so the returned gradient grows with horizon and is not a useful
long-time Lyapunov sensitivity. Shadowing-based sensitivity is not implemented.

The adaptive adapter supports forward-mode differentiation (`jax.jacfwd`/`jax.jvp`)
but not reverse mode because its data-dependent `lax.while_loop` cannot be replayed
by JAX reverse-mode AD. Requiring `scan_kind="lax"` also couples this feature to a
specific Diffrax solver construction detail, which is tested but should be watched
across dependency updates.

Resolved: `kaplan_yorke_dimension`'s docstring now states explicitly that it is
eager, host-side post-processing (Python-`float` branching), not `jit`/`vmap`/
`grad`-compatible, rather than leaving that implicit in its implementation.

Criticism: no multi-device `pmap`/`shard_map` path or documented distributed sweep
exists. Impact: Low at current maturity; Medium for large parameter studies.
Improvement: first document single-device batching and compilation/timing practice;
add distributed execution only with a representative workload and tests.

## 4. API design

The package now has coherent problem objects: `ODEProblem`, `DDEProblem`, and
`Network`, built through `ode_problem`, `dde_problem`, `network_problem`, and
`network_dde_problem`. These store `state0`, `dt`, and the prepared step function so
users do not repeat dynamics configuration at the spectrum call. Lower-level direct
step-function forms remain available.

`LyapunovResult` now carries `exponents`, `history`, `times`, and a checkpoint.
`LyapunovCheckpoint` and `DDECheckpoint` support continuation without restarting;
the DDE checkpoint additionally preserves the ring buffer and its time index.
`convergence_drift` summarizes tail movement and pairs naturally with chunked resume
loops. The main classes, constructors, diagnostics, integrators, and Kaplan-Yorke
helper are exported from `lyapax.__init__`; the earlier stale package docstring and
minimal-export criticism is resolved.

Partially resolved: `LyapunovCheckpoint`/`DDECheckpoint` now carry `dt` and
`lyapunov_spectrum`/`lyapunov_spectrum_dde` raise on `resume=` if it does not match
the resuming call's `dt` (`dt` was chosen because it is the one stable field cheap
to check without hashing an opaque JAX closure -- for a DDE it also implicitly
guards the grid-snapped delay/horizon, which is derived from `dt`). Covered by
`test_resume_rejects_dt_mismatch` in both `tests/test_lyapunov_core.py` and
`tests/test_dde.py`.

Criticism (remaining): the step function, parameters, integrator, and
renormalization cadence are still not identified or validated on resume -- only
`dt` and the tracked dimensions are. A same-shaped, same-`dt` checkpoint can
still be resumed against a different `rhs`/parameters without an error. Impact:
Medium-High for scientific reproducibility when checkpoints are persisted or
routed programmatically (down from High now that the most likely accidental
mismatch -- a different `dt` -- is caught). Improvement: an explicit, caller-set
`tag`/version field would close the rest of this gap without needing to identify
an arbitrary Python closure; add only if persisted/routed checkpoints become an
actual workflow, per `notes`'s general bias against speculative API surface.

Resolved: `convergence_drift`'s docstring now recommends mixing absolute and
relative tolerance (pointing at `ConvergenceDrift.relative`'s near-zero-exponent
caveat), requiring several consecutive converged chunks rather than trusting a
single pass, and notes that CPU/GPU QR differences can shift which chunk a tight
`tol` first passes on. `examples/16_convergence_drift.py` now also prints the
active JAX backend for the same reason.

Criticism: callable extension points still use broad `Callable`/`dict` annotations.
Impact: Low to Medium because shape and pytree contracts are scientifically relevant.
Improvement: add type aliases or Protocols and consistent shape-oriented docstrings.

## 5. Software engineering

The project has a clean `src/` layout, focused modules, feature-oriented tests, Ruff
configuration, contribution/support files, and GitHub issue/PR templates. GitHub
Actions runs tests on Python 3.11/3.12, Ruff, package build/metadata checks, and a
warning-as-error Sphinx build. A separate trusted-publishing workflow builds and
publishes releases to PyPI. These resolve the original review's missing-CI and
missing-tooling concerns.

The core and DDE engines share QR accumulation through `_run_renorm_scan`. Validation
now covers malformed `n_steps`, `k`, `renorm_every`, conflicting problem/direct
arguments, resume incompatibilities, and adaptive/DDE misuse.

Criticism: `build_jax_dfun` still compiles model strings with `exec()`. This is
acceptable for trusted local specifications but unsafe for untrusted input. Impact:
Medium. Improvement: keep the trust boundary prominent; use a restricted expression
parser before accepting remotely supplied specifications.

Criticism: adaptive support relies on low-level Diffrax solver/controller behavior
rather than its high-level solve API. Impact: Medium maintenance risk across Diffrax
versions. Improvement: pin a tested compatibility range more narrowly or add a CI
matrix entry for the oldest and newest supported Diffrax versions.

## 6. Documentation and examples

The README is no longer a stub. It includes installation, x64 guidance, a minimal
Lorenz example, method/scope notes, DDE delay modes, CI/docs badges, and an examples
table. Sphinx documentation includes background material on Lyapunov exponents, the
implementation, validation, capabilities, performance, motivation, and benchmarks.
Sphinx-Gallery executes runnable examples as part of CI.

There are now twenty numbered demos (`00`-`19`), including the unified public API,
DDE interpolation, GPU crossover, adaptive ODEs, convergence diagnostics, ODE and DDE
resume, parameter differentiation, and Kaplan-Yorke dimension.

Resolved: README's examples table now lists demos 15-19, its development-install
block adds `pip install -e ".[adaptive]"`, and `CHANGELOG.md`'s `[Unreleased]`
section now records adaptive integration, convergence/resume, differentiability
findings, and the Kaplan-Yorke dimension. Remaining: the table is still
hand-maintained (not generated or checked against `examples/`), so it can drift
again -- Impact Low, not worth a generator for twenty entries at this stage.

Resolved: `docs/api.rst` now has an Adaptive section (`lyapax.adaptive`, with a
note on the `adaptive` extra); confirmed by a clean `sphinx-build -b html -W`.

Resolved: README's docs-build command now installs `[docs,adaptive]`, matching CI.

## 7. Performance

For ordinary maps/ODEs, each raw step costs approximately `O(k)` JVPs, batched with
`vmap`, plus QR every `renorm_every` steps. QR costs approximately `O(d k^2)` for a
partial spectrum and `O(d^3)` for a full spectrum. For DDEs, replace `d` with the
augmented `d_total = d_state + d_buffer`; long delays therefore make explicit small
`k` essential.

The repository now includes benchmark documentation and demos comparing matrix-free
propagation with dense Jacobians, fixed integrator accuracy, parameter sweeps, and
CPU/GPU crossover. This is a stronger basis than the original review had, but JAX
compile time, warmup, asynchronous execution, device transfer, and shape specialization
still require careful interpretation.

Criticism: adaptive integration nests an internal accept/reject loop inside each JVP
and outer Lyapunov scan, so cost depends strongly on tolerances and system behavior.
Current public benchmarks do not characterize this overhead. Impact: Medium.
Improvement: report accepted/rejected internal step counts and benchmark adaptive
versus fixed integration at matched exponent error, not merely matched nominal time.

## 8. Validation

Validation remains a major strength. The suite now covers:

- linear ODE spectra against eigenvalues;
- logistic/tent maps against `ln(2)` and the Hénon sum against `ln|b|`;
- Lorenz divergence/literature checks and Rössler qualitative behavior;
- partial-spectrum consistency and JVP action against dense `jacfwd`;
- linear networks, custom coupling, stage-accurate integration, and network scale;
- scalar and network DDEs against Lambert-W references;
- DDE ring-buffer timing and interpolated-history convergence order;
- Mackey-Glass qualitative chaos and resumed DDE equivalence;
- sweep equivalence against Python loops and opt-in GPU smoke tests;
- adaptive Diffrax tangent action, tolerance behavior, RK4 agreement, and AD limits;
- ODE/DDE checkpoint continuation and convergence-drift validation;
- fixed-step parameter gradients in non-chaotic and chaotic regimes;
- Kaplan-Yorke edge cases, partial-spectrum guards, and the Lorenz reference value; and
- adaptive `max_steps` exhaustion and ODE/DDE resume-`dt`-mismatch rejection.

Resolved: `ci.yml`'s `test` job now installs `[dev,adaptive]` (was `[dev]`), so the
dedicated adaptive pytest modules run as tests on every push/PR across Python
3.11/3.12, not only implicitly via the docs job's Sphinx-Gallery execution.

Criticism: GPU tests remain opt-in and are not run by hosted CI. Impact: Medium for
backend claims and resume/convergence reproducibility. Improvement: maintain a
periodic GPU validation record or an optional self-hosted job, including CPU/GPU
tolerance comparisons rather than bitwise expectations.

Correction (prior review pass was stale): the Rössler divergence identity is
represented by a structural test -- `test_rossler_sum_matches_divergence_identity`
in `tests/test_lyapunov_core.py` verifies `sum(lambda) = a - c + mean(x)` over the
same post-transient window the exponent run itself uses. No action needed; this
item should not have appeared as an open criticism.

## 9. Scientific reproducibility

Tests force CPU and enable x64, and random tangent bases are seeded. Runtime entry
points warn when float32 is used while x64 is disabled. README guidance places the
x64 configuration before JAX array creation. Resume checkpoints preserve numerical
state, tangent bases, cumulative growth, elapsed time, and DDE history state.

Dependencies remain broad (`jax>=0.10`, `numpy>=1.23`; Diffrax is optional) and there
is no lockfile. This is reasonable for package installation but insufficient by itself
for exact computational reproduction across JAX/XLA/backend versions. CPU and GPU QR
and reductions are not expected to be bitwise identical, and chaotic trajectories
amplify small differences.

Criticism: the project records locally verified JAX/Diffrax versions in comments but
does not publish a tested dependency matrix or reproducible environment file. Impact:
Medium. Improvement: record versions with benchmark/scientific outputs and test lower
bounds plus a current dependency set in CI.

Criticism: checkpoints are in-memory NamedTuple-like pytrees without an explicit,
versioned serialization contract. Impact: Medium for long simulations resumed across
processes or package upgrades. Improvement: define a stable checkpoint schema with
package version and compatibility metadata before encouraging persistent checkpoints.

## 10. Strengths

- Scientifically appropriate Benettin/QR implementation for differentiable JAX maps,
  fixed/adaptive explicit ODE maps, networks, and fixed-delay DDE discretizations.
- Matrix-free `jvp`/`vmap` tangent propagation with practical partial spectra.
- Higher-order uniform-delay DDE history interpolation with direct timing/order tests.
- Coherent problem-object API plus lower-level step-function escape hatches.
- Inspectable convergence history, diagnostics, and resumable ODE/DDE computation.
- Candid, tested differentiability limits rather than broad AD claims.
- Kaplan-Yorke post-processing with a partial-spectrum correctness guard.
- Strong validation, public documentation, CI, packaging, and contribution scaffolding.

## 11. Weaknesses

- No stiff/implicit ODE solver and no adaptive DDE integration.
- Uniform-delay interpolation does not extend to per-edge heterogeneous delays.
- Checkpoint resume validates `dt` but not the step function, parameters, integrator,
  renormalization cadence, or package version.
- Chaotic long-horizon parameter sensitivities need shadowing methods, which are absent.
- Callable extension points still use broad `Callable`/`dict` annotations.
- No published tested dependency matrix (only single verified versions in
  `pyproject.toml` comments) and no periodic/hosted GPU CI.

## 12. Major concerns

1. Reproducibility: resume compatibility checks now cover `dt` but not
   step-function/parameter/integrator identity.
   Why it matters: a same-shaped, same-`dt` checkpoint can still silently continue
   under different dynamics or parameters while looking valid. Impact: Medium-High
   for scientific reproducibility when checkpoints are persisted or routed
   programmatically. Improvement: an explicit caller-set tag/version field, added
   only if persisted/routed checkpoints become an actual workflow.

2. Scope limitation: naive gradients are not long-time chaotic sensitivities.
   Why it matters: a finite gradient can look authoritative even while diverging with
   horizon. Impact: High for optimization/control claims; Low for validated linear and
   short non-chaotic examples. Improvement: retain prominent warnings and implement a
   shadowing method only if chaotic sensitivity becomes a core goal.

Resolved since the prior pass: adaptive `max_steps` exhaustion now raises instead of
silently truncating (`lyapax.adaptive`, via `equinox.error_if`); adaptive tests now
run in the main CI test matrix (`ci.yml`'s `test` job installs `[dev,adaptive]`); and
checkpoint resume now validates `dt`. See the inline "Resolved"/"Partially resolved"
notes in sections 2, 4, and 8 above for what each fix does and does not cover.

## 13. Minor suggestions

- Replace broad callable annotations with reusable shape/pytree contracts.
- Keep README's examples table in sync with `examples/` as new demos are added (still
  hand-maintained, not generated or checked).
- Record a tested dependency matrix (not just a single verified version) alongside
  benchmark/scientific outputs.

## 14. Possible future improvements

- Add an explicit checkpoint tag/version field once persisted or programmatically
  routed checkpoints become an actual workflow (dynamics/parameter identity is the
  remaining resume-safety gap; `dt` is already checked).
- Add periodic or self-hosted GPU CI coverage.
- Add implicit ODE support only after verifying differentiable tangent propagation.
- Extend interpolated history to heterogeneous delays only with a defined edge-aware API.
- Add shadowing-based sensitivities for genuinely chaotic parameter gradients.
- Add multi-device sweep examples when a representative workload warrants them.
- Make Kaplan-Yorke post-processing transform-compatible if batched use becomes common.

Resolved: adaptive-integration overhead is now characterized at matched exponent
error (not matched nominal `dt`/`rtol`) across small (Lorenz), large (`d=1500-3000`),
GPU, and relaxation-oscillator (Van der Pol) cases -- adaptive integration is 2-4x
slower than fixed-step `rk4`/`rk6` in every case tested, narrowing toward parity only
at large `d`. Documented in `docs/background/capabilities.md`'s "Adaptive integration
is not a speed optimization" section, referenced from `lyapax.adaptive`'s module
docstring. Separately, while investigating this the `ValueError` raised for passing
an adaptive integrator to a network/DDE problem was found to say "not supported for
DDEs" while actually firing for any `network_problem` (delayed or not) -- corrected
to describe the real restriction (adaptive integration works only through a single,
uncoupled `ode_problem`), with a new regression test locking in the non-delayed-network
case specifically.

Also resolved (unprompted cleanup, not itself a review finding): docstrings across
`src/lyapax/` no longer reference the repository's internal `notes/` directory, which
is not part of the published package or docs and was therefore an unusable pointer for
anyone reading the installed package or hosted docs -- those references now point to
proper Sphinx cross-references (`:ref:`/`:doc:`) instead.

## 15. Overall assessment

`lyapax` has moved from a credible early core into a coherent scientific package. It
now combines matrix-free Benettin/QR propagation with problem objects, corrected and
higher-order DDE history handling, adaptive explicit ODE integration, convergence
diagnostics, resumable runs, parameter-differentiation guidance, Kaplan-Yorke
dimension, public docs, and CI. Several of the original review's largest usability and
engineering concerns are resolved, including this pass's three previously "High
impact" findings: adaptive `max_steps` truncation now raises rather than silently
corrupting elapsed-time accounting, adaptive tests run in the main CI matrix rather
than only implicitly through the docs build, and checkpoint resume validates `dt`.

The main remaining risk is narrower than before: checkpoint resume still cannot tell
whether a same-shaped, same-`dt` checkpoint is being resumed against the same
dynamics and parameters, since that would require identifying an opaque JAX closure
rather than comparing a stored value. That gap, plus backend-sensitive convergence
criteria, should be closed (or explicitly accepted as a documented limitation) before
resume-driven workflows are presented as fully reproducible across environments and
processes. Within the documented deterministic, explicit, fixed-delay scope, the
scientific implementation and validation are strong.
