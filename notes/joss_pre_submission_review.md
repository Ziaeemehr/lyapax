# JOSS pre-submission review for LYAPAX

Date: 2026-07-07

Reviewer stance: pre-submission review against the current JOSS review criteria
at <https://joss.readthedocs.io/en/latest/review_criteria.html>, plus common
scientific Python and JAX packaging expectations.

## Executive recommendation

**Recommendation: C. Needs moderate work before submission.**

Technically, LYAPAX is already a credible and useful scientific package. It has
a focused scope, a clear JAX-native contribution, meaningful tests, runnable
examples, Sphinx documentation, a JOSS paper draft, validation notes, and
benchmarks against `jitcode`, `jitcdde`, and `ChaosTools.jl`. The core
algorithmic direction is appropriate: Benettin/QR with matrix-free tangent
propagation using `jax.jvp`, `jax.vmap`, and `jax.lax.scan`.

The main blockers are not that the software is a thin wrapper or scientifically
uninteresting. The main blockers are JOSS process/readiness issues:

- Local git history spans only **2026-07-02 through 2026-07-07** and shows one
  contributor. Current JOSS criteria explicitly flag projects with most history
  concentrated shortly before submission as a possible pre-review screening
  failure.
- The repository is missing community/reuse files expected for a polished JOSS
  package: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, PR
  template, `CHANGELOG.md`, and `CITATION.cff`.
- The JOSS paper draft is promising but does not yet use all of the current
  required section labels, especially `State of the field`, `Software design`,
  `Research impact statement`, and `AI usage disclosure`.
- Documentation has at least one important inconsistency: the README says DDE
  delays are integer-step only, while the docs describe a newer
  Hermite-interpolated mode with `interpolate=True`.
- The package build could not be verified locally because `python -m build`
  failed with `No module named build`. This is easy to fix, but wheel/sdist
  generation should be tested before submission.

If the public GitHub repository actually has older public history, releases,
issues, or contributors that are not present in this local clone, the JOSS
process risk is lower. Based only on this checkout, that is the largest concern.

## Evidence checked

- Repository indexed in codebase-memory as `home-ziaee-git-lyapunov`.
- Test suite: `python3 -m pytest -q` -> **48 passed, 2 skipped** in 50.54 s.
- Documentation build: `sphinx-build -b html docs docs/_build/html` -> **built
  successfully with 1 warning**. Warning: generated
  `docs/auto_examples/14_gpu_acceleration (copy).rst` is not included in a
  toctree.
- Package build: `python3 -m build` -> **not run successfully** because the
  `build` package is not installed in the current environment.
- Editable install exists locally: `lyapax 0.1.0` installed from this checkout.
- Git tag: `v0.1.0`.
- Local commit history: 61 commits by one author, earliest local commit
  2026-07-02, latest 2026-07-07.

## 1. JOSS readiness

### Research usefulness

LYAPAX addresses a real research need: computing Lyapunov spectra for ODEs,
maps, networks, and fixed-delay systems without manually deriving variational
equations. This is useful in nonlinear dynamics, computational neuroscience,
physics, and any JAX-based modeling workflow that needs chaos diagnostics or
stability characterization.

### Novelty and contribution

The strongest contribution is not the Benettin/QR algorithm itself, which is
standard, but the implementation model:

- JAX-native differentiable step functions.
- Matrix-free tangent propagation with `jax.jvp` and batched tangent columns via
  `jax.vmap`.
- Partial spectra whose cost scales with the requested `k`.
- Composable support for user-written JAX models and network couplings.
- Batched parameter sweeps with `vmap`.
- Transparent CPU/GPU execution through JAX.
- Fixed-delay DDE support through an augmented state/ring-buffer approach.

This is more than a thin wrapper. It is a reusable implementation of a
well-known numerical method in a modern autodiff/JIT ecosystem with a clear
computational niche.

### Scope and target audience

The scope is suitable for JOSS: it is not a one-off analysis script, and it is
not trying to become a full dynamical-systems framework. The target audience is
well defined: researchers who already use Python/JAX and need Lyapunov spectra
for custom differentiable dynamical systems, especially networks and DDEs.

### Scientific relevance

The scientific relevance is strong. Lyapunov exponents are a standard
diagnostic for chaos and stability. The repository includes validation against
analytic systems, structural invariants, literature values, and independent
software. This is exactly the kind of scientific positioning JOSS reviewers
expect.

### Current suitability

**Not quite ready as submitted today.** The software is technically promising,
but the current local repository state would likely trigger JOSS reviewer or
editor concerns around development history, open-source practice, community
guidelines, citation metadata, and paper section completeness.

## 2. Documentation

### Strengths

- `README.md` gives a concise summary, installation commands, float64 warning,
  minimal Lorenz example, method overview, examples table, docs build command,
  and links to validation/benchmark notes.
- Sphinx documentation exists under `docs/`.
- API reference is generated via autodoc in `docs/api.rst`.
- The background docs cover Lyapunov exponents, implementation decisions,
  validation, capabilities, and limitations.
- There are 15 numbered examples under `examples/`, and Sphinx-Gallery builds
  them into a gallery.
- The documentation is unusually candid about limitations: fixed-step solvers,
  precision, delay handling, no stochastic/PDE support, and `exec()` safety in
  model specs.

### Missing or weak documentation

- **README inconsistency:** README says DDE delays are integer-step only and
  there is no sub-step interpolation, while `docs/background/capabilities.md`
  and `docs/background/lyapax_implementation.md` describe Hermite interpolation
  via `interpolate=True`. This should be fixed before submission.
- **Community support documentation is missing:** no clear `CONTRIBUTING.md`,
  issue reporting guide, support policy, or development setup guide beyond the
  install extras.
- **Citation information is missing:** no `CITATION.cff` at repository root.
  JOSS itself produces a DOI after acceptance, but pre-submission citation
  metadata is still expected for a polished package.
- **API docs need examples at the function level:** autodoc is present and many
  docstrings are detailed, but reviewers may expect example inputs/outputs for
  the major public functions (`lyapunov_spectrum`, `lyapunov_spectrum_dde`,
  `network_problem`, `dde_problem`, `sweep_lyapunov_spectrum`).
- **Installation should distinguish PyPI vs. source install status:** README
  says `pip install lyapax`. Confirm that `lyapax` is actually published and
  points to this package, or change the primary install instruction to
  `pip install git+https://...` until PyPI is live.
- **Docs warning should be fixed:** Sphinx builds with a warning about
  `14_gpu_acceleration (copy).rst`, which suggests a duplicate generated file
  or stale build artifact.

## 3. Software functionality

### Installation and dependencies

The package uses a standard `pyproject.toml` with setuptools, Python `>=3.11`,
and dependencies on `jax>=0.10` and `numpy>=1.23`. Extras are defined for
development, examples, docs, and benchmarks.

What works:

- Editable install is present locally.
- Tests run successfully.
- Docs build successfully.
- Publish workflow builds sdist/wheel and uses PyPI trusted publishing.

What needs attention:

- `python -m build` could not be checked locally because the `build` package is
  not installed. Add `build` to a release/dev workflow or document
  `python -m pip install build`.
- `pyproject.toml` lacks common metadata: project URLs, classifiers, keywords,
  and license file declaration. These are not mandatory for JOSS but improve
  PyPI readiness.
- The lower bound `jax>=0.10` should be checked carefully. If LYAPAX depends on
  behavior introduced in recent JAX versions, set a more realistic lower bound.

### Usability and API clarity

The public API is reasonably clear:

- `ode_problem(...)` and `lyapunov_spectrum(...)` for ODEs/maps.
- `network_problem(...)` for networks.
- `dde_problem(...)`, `network_dde_problem(...)`, and
  `lyapunov_spectrum_dde(...)` for DDEs.
- `sweep_lyapunov_spectrum(...)` for batched sweeps.

The result object includes exponents and history, which is valuable for
convergence diagnostics.

Potential usability issues:

- `lyapunov_spectrum(problem, n_steps)` is convenient but overloading `state0`
  as positional `n_steps` may surprise users. It is documented, but keyword-only
  usage should be encouraged in examples.
- DDE full-spectrum default can be very expensive because the augmented
  dimension includes the ring buffer. The docstring warns about this, but the
  front-page docs should also emphasize passing `k` for DDE/network systems.
- Some error messages are good (`dt` mismatch, `k`, `n_steps`,
  `renorm_every`), but there should be tests for more invalid shapes and
  non-JAX/non-differentiable functions.

## 4. Testing

### Strengths

The test suite is a strong point. It includes:

- Exact linear ODE eigenvalue checks.
- Complex conjugate eigenvalue checks.
- Logistic and tent map exact exponent checks.
- Hénon determinant/sum-of-exponents invariant.
- Lorenz and Rössler checks against structural/literature expectations.
- Partial-spectrum checks.
- DDE tests against Lambert W references and qualitative Mackey-Glass behavior.
- DDE interpolation and ring-buffer time convention tests.
- Network and delayed-network tests.
- Parameter sweep tests.
- GPU tests, skipped when no GPU is available.
- CI on Python 3.11 and 3.12 with pytest and ruff.

### Missing or recommended tests

- Add a docs build job in CI, ideally treating warnings as errors after fixing
  current warnings.
- Add a package build/check job: `python -m build` and `twine check dist/*`.
- Add CI for Python 3.13 if supported.
- Add tests for public API shape errors and dtype/precision warnings.
- Add reproducibility tests for fixed seeds where feasible.
- Add more explicit convergence tests for DDE interpolation vs. grid-snapped
  mode, including regression thresholds that would catch first-order fallback.
- Add tests that run examples as scripts in a clean environment or rely on
  Sphinx-Gallery in CI to execute them.
- Add benchmark regression tests only if they are lightweight; otherwise keep
  full benchmarks manual but document exact reproduction steps.

## 5. Scientific correctness

### Strengths

The numerical approach is scientifically appropriate:

- Benettin/QR is the standard method for Lyapunov spectra.
- Tangent propagation through the same numerical step map avoids inconsistent
  state/tangent discretizations.
- Periodic QR renormalization is correct and necessary for stability.
- Validation is anchored to exact values, structural invariants, literature
  ranges, and independent tools.
- The docs correctly warn that fixed-step ODE exponents are exponents of the
  numerical time-`dt` map and require `dt` convergence checks.
- The float64 warning is important and appropriate for this class of algorithm.

### Potential weaknesses

- No adaptive/stiff ODE solver support. This is acceptable if clearly scoped,
  but users working on stiff systems may get misleading results unless docs
  repeat the limitation prominently.
- DDE numerical correctness depends heavily on delay history handling. The docs
  now describe Hermite interpolation, while README still describes the older
  grid-snapped-only limitation. Reviewers will want one consistent story.
- `renorm_every` selection is left to the user. The documentation explains the
  risk, but more practical heuristics/examples would help.
- Degenerate or near-degenerate spectra can reorder in `history`. This is
  documented, but downstream users may misinterpret convergence plots.
- GPU results should be presented carefully: JAX backend support is real, but
  speedups are problem-size dependent and depend on compilation/warmup.

## 6. Code quality

### Strengths

- Package organization is clear: `core`, `dde`, `network`, `coupling`,
  `integrators`, `systems`, `sweep`, and `simulator`.
- Public functions have meaningful docstrings.
- The core algorithm is compact and uses JAX control flow rather than Python
  loops in the hot path.
- The project avoids excessive abstraction. Coupling as a plain callable is a
  good design choice.
- Tests are close to the scientific claims.

### Concerns and improvements

- Some core functions are long (`lyapunov_spectrum` ~150 lines,
  `lyapunov_spectrum_dde` ~190 lines). They are understandable, but extracting
  input normalization/validation helpers could improve maintainability.
- Type hints are present but could be tightened. Consider `jaxtyping` or
  clearer shape documentation if the project wants stronger scientific API
  contracts.
- `ModelSpec` string right-hand sides use `exec()` and are explicitly unsafe
  for untrusted input. This is documented, but the API should make that warning
  unavoidable in the relevant docstring.
- Add `ruff format` or `black` to CI if formatting consistency matters.
- Consider adding `mypy`/`pyright` only if the codebase can absorb the
  maintenance cost; scientific JAX code can make static typing noisy.

## 7. JAX best practices

### Strengths

- Uses `jax.lax.scan` for repeated stepping, which is the right pattern for JIT
  compilation and avoids Python loops in traced computations.
- Uses `jax.jvp` for matrix-free tangent propagation.
- Uses `jax.vmap` over tangent columns and parameter sweeps.
- Keeps state immutable and functional.
- Uses JAX arrays and pure step functions.
- Warns about float64 requirements.

### Improvements

- Document which arguments are static when users wrap calls in `jax.jit`.
  `n_steps`, `renorm_every`, `k`, and shapes will be compilation-relevant.
- Consider exposing jitted helper functions or examples showing recommended
  JIT usage for repeated calls.
- Track and document compilation cost separately from runtime in benchmarks.
  The benchmark notes already discuss this; surface the distinction in the
  paper and docs.
- For very large DDE augmented states, memory cost can become substantial.
  Provide a dedicated "large systems" guide recommending partial spectra,
  smaller `k`, and convergence strategies.
- Make sure all examples avoid accidental recompilation inside Python loops.

## 8. Packaging

### Strengths

- Standard `pyproject.toml`.
- Dynamic version from `lyapax.__version__`.
- Optional extras for dev/examples/docs/benchmarks.
- PyPI publish workflow exists and uses trusted publishing.
- MIT license file is present.
- Version tag `v0.1.0` exists.

### Improvements

- Add project URLs:
  - `Homepage`
  - `Documentation`
  - `Repository`
  - `Issues`
  - `Changelog`
- Add classifiers and keywords.
- Add `license-files = ["LICENSE"]` or equivalent modern metadata.
- Add `build` to dev/release tooling and verify `python -m build`.
- Add `twine check`.
- Confirm that `pip install lyapax` resolves to this project on PyPI before
  making it the primary install path.

## 9. GitHub repository quality

### Present

- GitHub Actions CI for tests and lint.
- PyPI publish workflow.
- Read the Docs configuration.
- LICENSE.
- README.
- JOSS paper draft and bibliography.
- Tag `v0.1.0`.

### Missing or expected for polished JOSS submission

- `CONTRIBUTING.md`.
- `CODE_OF_CONDUCT.md`.
- `CITATION.cff`.
- `CHANGELOG.md`.
- Issue templates.
- Pull request template.
- Release notes attached to GitHub releases.
- Zenodo integration or clear plan to archive the reviewed release.
- Public evidence of open development, issues, PRs, or external users.

The current local history also shows one contributor and a very short
development window. Under current JOSS criteria, this is the most serious
repository-quality concern.

## 10. Comparison with existing software

### ChaosTools.jl / DynamicalSystems.jl

ChaosTools.jl is more mature and broader. It lives in the Julia
DynamicalSystems ecosystem and offers extensive functionality beyond Lyapunov
spectra. LYAPAX's advantage is Python/JAX integration, differentiable model
code, `vmap`-friendly sweeps, and GPU execution.

### jitcode / jitcdde

`jitcode` and `jitcdde` are established Python tools that compile symbolic
systems to C. They are strong references for ODE/DDE Lyapunov computation.
LYAPAX differs by accepting ordinary JAX functions instead of symbolic
expressions and by targeting autodiff/JIT/GPU workflows.

### SciPy-based implementations

SciPy examples are often custom scripts around finite differences, explicit
Jacobians, or variational equations. LYAPAX is more reusable and better suited
to differentiable high-dimensional JAX models, but lacks SciPy's adaptive/stiff
solver ecosystem.

### Diffrax ecosystem

Diffrax provides excellent differentiable ODE/SDE/CDE solvers in JAX, including
adaptive methods, but it is not primarily a Lyapunov-spectrum package. A future
LYAPAX integration with Diffrax could reduce the fixed-step limitation. For
now, LYAPAX fills a more specialized spectrum-computation role.

### Other Python Lyapunov packages

Most small Python Lyapunov packages/scripts are narrower, less validated, or
not JAX-native. LYAPAX's differentiability, vectorization, partial-spectrum
design, and DDE/network emphasis give it a credible novelty claim.

### Novelty assessment

The novelty is sufficient for JOSS if the paper clearly frames the contribution
as a JAX-native, matrix-free, differentiable, vectorized implementation rather
than a new mathematical algorithm. The paper should cite and compare with the
prior tools above and explain why contributing to those tools would not satisfy
the JAX/GPU/autodiff use case.

## 11. Paper readiness

The current `joss/paper.md` is a good draft. It includes a summary, statement
of need, method/implementation, validation, acknowledgements, and references.

Before submission, update it to satisfy current JOSS required content:

- Add a clearly labeled **State of the field** section comparing
  ChaosTools.jl/DynamicalSystems.jl, `jitcode`, `jitcdde`, SciPy-style
  implementations, and Diffrax-adjacent workflows.
- Add a clearly labeled **Software design** section. The existing method text
  can be split/reworked into this section, but it should explicitly discuss
  design tradeoffs: fixed-step maps vs adaptive solvers, JVP/vmap vs dense
  Jacobian, partial spectra, DDE ring-buffer dimension, and plain callables vs
  registries.
- Add a clearly labeled **Research impact statement** with concrete evidence:
  benchmarks, examples, intended research workflows, any publications or
  external users, presentations, or downstream projects.
- Add a clearly labeled **AI usage disclosure**. If no AI tools were used, say
  so. If AI was used, describe how and how correctness was checked.
- Include limitations and future work: adaptive/stiff solvers, Diffrax
  integration, state-dependent/distributed delays, stochastic exponents, DDE
  scalability.
- Ensure benchmark claims are reproducible and not overstated.
- Add a software archive reference once a Zenodo/GitHub release DOI exists.

## 12. Repository weaknesses by severity

### Critical

**Short/single-author local development history**

- Why it matters: Current JOSS criteria explicitly treat very recent,
  concentrated development history and lack of open development as potential
  pre-review screening failures.
- Where it occurs: local git history, 2026-07-02 to 2026-07-07; `git shortlog`
  shows one contributor.
- How to fix: If older public history exists, make sure reviewers can see it.
  Otherwise wait before submission, develop publicly, use issues/PRs, tag
  releases, and document external use or research context.
- Expected effort: Calendar time, not just coding effort. Potentially weeks to
  months if no prior public history exists.

**Missing required current JOSS paper sections**

- Why it matters: The current JOSS criteria require substantive sections for
  statement of need, state of the field, software design, research impact, and
  AI usage disclosure.
- Where it occurs: `joss/paper.md`.
- How to fix: Add or rename sections as described above.
- Expected effort: 0.5-1 day for a strong rewrite if benchmark evidence is
  already available.

**Missing community guidelines**

- Why it matters: JOSS documentation criteria require clear ways to contribute,
  report problems, and seek support.
- Where it occurs: repository root and `.github/`.
- How to fix: Add `CONTRIBUTING.md`, support/issue guidance, issue templates,
  and PR template.
- Expected effort: 1-3 hours.

### Major

**Documentation inconsistency about DDE interpolation**

- Why it matters: Conflicting claims undermine reviewer confidence in the
  implementation and scope.
- Where it occurs: `README.md` vs `docs/background/capabilities.md` and
  `docs/background/lyapax_implementation.md`.
- How to fix: Update README to describe both grid-snapped and
  Hermite-interpolated modes accurately.
- Expected effort: 30-60 minutes.

**No `CITATION.cff`**

- Why it matters: Citation metadata is expected for research software and makes
  the package easier to cite before/after JOSS acceptance.
- Where it occurs: repository root.
- How to fix: Add `CITATION.cff` matching package authors, title, version,
  repository URL, license, and preferred citation.
- Expected effort: 30 minutes.

**Build verification not part of local dev evidence**

- Why it matters: External researchers need installable wheels/sdists.
- Where it occurs: local `python -m build` failed because `build` is missing;
  CI publish workflow builds but regular CI does not verify artifacts.
- How to fix: Add a CI job installing `build` and running `python -m build` and
  `twine check`.
- Expected effort: 1 hour.

**Docs build warning**

- Why it matters: JOSS reviewers may build docs and expect clean output.
- Where it occurs: Sphinx warning for `14_gpu_acceleration (copy).rst`.
- How to fix: Remove stale generated duplicate or adjust exclude patterns.
- Expected effort: 15-30 minutes.

**Incomplete release hygiene**

- Why it matters: JOSS expects a reviewed, archived release.
- Where it occurs: release process/metadata.
- How to fix: Add changelog, release notes, Zenodo archive, and release DOI in
  paper references.
- Expected effort: 1-3 hours plus archive setup.

### Minor

**Missing PyPI metadata polish**

- Why it matters: Improves discoverability and trust.
- Where it occurs: `pyproject.toml`.
- How to fix: Add URLs, classifiers, keywords, license-files.
- Expected effort: 30 minutes.

**API docs could use more concrete examples**

- Why it matters: Autodoc alone is sometimes too reference-like for new users.
- Where it occurs: docstrings/API reference.
- How to fix: Add short example blocks to major public functions.
- Expected effort: 2-4 hours.

**JAX static-argument guidance is limited**

- Why it matters: Users may accidentally trigger recompilation or misuse `jit`.
- Where it occurs: docs.
- How to fix: Add a short JAX performance guide.
- Expected effort: 2-3 hours.

**Safety warning for `exec()` should be more visible**

- Why it matters: Untrusted model specs are unsafe.
- Where it occurs: simulator/model-spec docs and capabilities page.
- How to fix: Put the warning directly in the `ModelSpec` API docs and README
  limitations if that API is public.
- Expected effort: 30-60 minutes.

## 13. Release readiness scores

| Category | Score | Explanation |
|---|---:|---|
| Documentation | 7/10 | Strong README, Sphinx docs, examples, background, validation. Needs consistency fixes, community docs, and richer API examples. |
| Testing | 8/10 | Good scientific tests and CI. Needs docs/build CI, artifact checks, more edge-case tests. |
| Scientific reliability | 7/10 | Good algorithm and validation. Fixed-step and DDE limitations are acceptable but must stay very clear. |
| API design | 7/10 | Clear public API and problem objects. Some positional overloading and DDE defaults may surprise users. |
| Maintainability | 7/10 | Organized and readable. Some long core functions and limited static typing. |
| Performance | 8/10 | Strong JAX design with `scan`, `jvp`, `vmap`, partial spectra, GPU path. Needs clearer compile/runtime benchmark reporting. |
| Reproducibility | 7/10 | Tests, examples, benchmarks, and docs are reproducible locally. Needs artifact builds, release archive, and cleaner benchmark instructions. |
| Overall JOSS readiness | 6/10 | Technically credible, but JOSS process/open-development and repository polish issues remain. |

## 14. Prioritized pre-submission checklist

1. Resolve the JOSS development-history risk: document prior public history if
   it exists, or delay submission until the repository shows sustained public
   development with issues, PRs, releases, and external/community signals.
2. Rewrite `joss/paper.md` to include the current required sections:
   `Statement of need`, `State of the field`, `Software design`, `Research
   impact statement`, and `AI usage disclosure`.
3. Add `CONTRIBUTING.md`, issue templates, PR template, and support/reporting
   guidance.
4. Add `CITATION.cff`.
5. Add `CHANGELOG.md` and release notes for `v0.1.0` or the intended JOSS
   release.
6. Set up Zenodo/GitHub archival DOI for the reviewed release and cite it in
   the paper.
7. Fix README's DDE delay/interpolation inconsistency.
8. Fix the Sphinx duplicate `14_gpu_acceleration (copy)` warning and add docs
   build to CI.
9. Add build verification to CI: `python -m build` and `twine check`.
10. Add project URLs, classifiers, keywords, and license file metadata to
    `pyproject.toml`.
11. Confirm `pip install lyapax` installs this package from PyPI, or adjust
    install instructions until publication is complete.
12. Add function-level examples to the core API docs.
13. Add a short JAX performance/JIT guide covering static arguments,
    compilation cost, warmup, and avoiding accidental recompilation.
14. Add more edge-case tests for invalid shapes, precision warnings,
    non-finite results, and DDE interpolation convergence.
15. In the paper and docs, explicitly position LYAPAX relative to
    ChaosTools.jl/DynamicalSystems.jl, `jitcode`, `jitcdde`, SciPy scripts, and
    Diffrax.

## Final assessment

LYAPAX has enough scientific substance and technical design quality to become a
good JOSS submission. It is not merely a wrapper: it implements a standard
scientific algorithm in a way that is idiomatic and useful for JAX users, with
meaningful validation and performance motivation.

However, based on this checkout, it should not be submitted immediately. The
technical package is close; the JOSS submission package is not. The highest
impact work is to address open-development evidence, current paper section
requirements, community/citation files, documentation consistency, and release
artifact verification.
