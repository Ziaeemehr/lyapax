# Benchmark & Accuracy Report — lyapax vs. published packages

**Status: draft skeleton, paused.** Structure and known values filled in;
live cross-tool runs (jitcode/jitcdde/ChaosTools.jl) are set up but not yet
executed/recorded — picking back up here once the M6/M7 milestone work
(`notes/milestones.md`) is finished. See "Open TODOs" at the bottom for
exactly where to resume.

## Purpose

Before publishing `lyapax`, back its correctness and performance claims
with a report that:

1. Cross-validates lyapax's Lyapunov-exponent estimates against
   **independent, established tools** doing the same computation on the
   same benchmark systems — not just against lyapax's own test suite.
2. Reports **wall-clock performance** on the same systems, so any speed
   claims (e.g. from the JAX/`jax.vmap`/`jax.jvp` design) are backed by a
   real side-by-side number, not just an internal before/after comparison.
3. Is honest about where the comparison is and isn't apples-to-apples
   (different tools use different algorithms/integration schemes/default
   tolerances — see "Fairness notes" below).

## Comparison targets

| Tool | Language | Algorithm | Why compare against it |
|---|---|---|---|
| **jitcode** ([PyPI](https://pypi.org/project/jitcode/)) | Python (symbolic → compiled C) | Benettin/variational-equation method for ODEs | The established Python reference for ODE Lyapunov spectra; symbolic+compiled, a very different execution model from lyapax's JAX tracing. |
| **jitcdde** ([PyPI](https://pypi.org/project/jitcdde/)) | Python (symbolic → compiled C) | Farmer (1982) discretized-map method for DDEs, Hermite-interpolated history | The direct precedent lyapax's own DDE engine follows conceptually (see `notes/milestones.md`, M4) — the natural package to validate the DDE engine against. |
| **ChaosTools.jl** (part of [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/chaostools/stable/)) | Julia | Benettin/QR (`lyapunov`, `lyapunovspectrum`) | The most widely used dynamical-systems toolbox in Julia; a mature, independently-implemented reference for both continuous and discrete (map) systems, covering the map benchmarks jitcode can't (it's ODE-only). |
| **Published literature values** | — | — | Already the primary reference in `notes/validation_systems.md`; repeated here for a single-document summary rather than re-derived. |

Deliberately **not** compared against anything in `lyapunov-master/` (this
repo's own earlier, explicitly-not-validated C++/Python code) — see the
caution at the top of `notes/milestones.md`.

## Benchmark systems

Reuses lyapax's own validated benchmark suite
(`notes/validation_systems.md`) rather than inventing a new one, so every
comparison row is already backed by an independent analytic/structural
reference in addition to the cross-tool numbers:

| System | Category | Independent reference | Applicable tools |
|---|---|---|---|
| Linear ODE, 3 distinct real eigenvalues (Tier 0.1) | Exact | `eigvals(A)` | jitcode, ChaosTools.jl |
| Logistic map, `r=4` (Tier 0.2) | Exact | `ln 2` | ChaosTools.jl only (jitcode is ODE-only) |
| Hénon map (Tier 0.3) | Exact (sum) | `ln|b|` | ChaosTools.jl only |
| Lorenz, `σ=10, ρ=28, β=8/3` (Tier 1.1/2) | Structural + literature | `sum(λ) = -(σ+1+β)`; `λ1 ≈ 0.9056` | jitcode, ChaosTools.jl |
| Rössler, `a=b=0.2, c=5.7` (Tier 1.2/2) | Structural + literature | `sum(λ) = a-c+⟨x⟩`; `λ1 ≈ 0.07` | jitcode, ChaosTools.jl |
| Linear coupled network, 4-node cycle (Tier 3.1) | Exact | `eigvals(γI + G·W)` | jitcode (as a flat ODE), ChaosTools.jl |
| Mackey-Glass, `β=0.2, γ=0.1, n=10, τ=17` (Tier 4.1) | Literature (qualitative) | `λ1` order `1e-2`–`1e-3`, KY dim `2`–`3` | jitcdde |
| Linear scalar DDE, `ẋ=-a x(t-τ)` (Tier 4.2) | Exact | Lambert W root | jitcdde |
| 2-node delayed linear network (Tier 4.3) | Exact | Lambert W root (2x2 case) | jitcdde |

Not planned for cross-tool comparison (no natural equivalent in the other
tools without a lot of extra plumbing): Kuramoto networks, delayed
Kuramoto networks, sigmoidal coupling, per-edge heterogeneous delay
matrices. These stay validated by lyapax's own test suite only
(`tests/test_network.py`, `tests/test_delayed_networks.py`) — worth noting
in the published report as a scope limitation of the comparison, not of
lyapax itself.

## Methodology

- **Same equations, same parameters.** Every system above is implemented
  independently in each tool's native API (symbolic for jitcode/jitcdde,
  `CoupledODEs`/`DeterministicIteratedMap` for ChaosTools.jl, `ModelSpec`/
  `rhs` for lyapax) from the same written-out equations/parameter values —
  not translated automatically, to avoid propagating a translation bug
  into all tools identically.
- **Matched, not identical, settings.** Integration step size, transient
  length, and total run length are matched as closely as each tool's API
  allows; QR-renormalization interval is matched where the concept exists
  in both. Where a tool's own defaults differ meaningfully (e.g. adaptive
  step control), that's recorded, not silently normalized away — see
  Fairness notes.
- **Wall time measured after warmup** where the tool has a JIT/compile
  step (lyapax, jitcode/jitcdde's C compilation), so first-call
  trace/compile cost doesn't dominate a number meant to characterize
  steady-state throughput. Both a first-call and warmed-up number will be
  reported, mirroring `examples/plot_07_speed_and_accuracy.py`'s existing
  practice for lyapax alone.
- **Same machine, same run.** All tools run on the same dev machine, CPU
  only (see `notes/milestones.md`'s GPU note — not applicable to any tool
  here yet), in the same session, to avoid cross-machine noise.

### Fairness notes (fill in as each comparison is actually run)

- jitcode/jitcdde compile the RHS symbolically to C; lyapax traces to XLA.
  Both pay a one-time compile cost with a very different profile (seconds
  to tens of seconds for C compilation vs. XLA tracing) — report first-call
  cost separately from steady-state cost specifically because of this.
- ChaosTools.jl's `lyapunovspectrum` has its own default integrator
  (`OrdinaryDiffEq`-based, adaptive by default) — an adaptive-step
  reference is a *different* numerical method from lyapax's fixed-step
  Heun/RK4, so agreement validates the science (same attractor, same
  exponents) rather than the numerics being identical schemes. Note this
  explicitly rather than presenting it as like-for-like.
- lyapax defaults to `float64` (`notes/milestones.md`, risk #1); confirm
  each other tool's working precision before comparing digits, not just
  headline values.

## Environment

| Component | Version | Notes |
|---|---|---|
| lyapax | `0.0.1` (editable) | `/home/ziaee/envs/lyapax` |
| JAX | `0.10.2` | CPU only, `jax_enable_x64=True` |
| jitcode | `1.7.3` | installed, not yet exercised |
| jitcdde | `1.8.3` | installed, not yet exercised |
| symengine | `0.14.1` | jitcode/jitcdde's symbolic backend |
| Julia | `1.12.1` | installed (`juliaup`) |
| ChaosTools.jl | `3.5.4` | |
| DynamicalSystemsBase.jl | `3.15.7` | |
| OrdinaryDiffEq.jl | `6.105.0` | |
| StaticArrays.jl | `1.9.18` | |
| Machine | (dev machine, CPU) | same GPU/cudnn limitation noted in `notes/milestones.md` M0 applies here too |

## Results — Accuracy

*Not yet populated.* Each table below will have one row per system, columns
for lyapax / jitcode-or-jitcdde / ChaosTools.jl / published value, and the
max-abs-difference between lyapax and each reference.

### ODE systems (lyapax vs. jitcode vs. ChaosTools.jl)

| System | lyapax | jitcode | ChaosTools.jl | Published | Notes |
|---|---|---|---|---|---|
| Linear ODE (Tier 0.1) | TODO | TODO | TODO | exact | |
| Lorenz `λ1` (Tier 1.1/2) | TODO | TODO | TODO | `0.9056` | |
| Lorenz `sum(λ)` | TODO | TODO | TODO | `-13.667` | |
| Rössler `λ1` (Tier 1.2/2) | TODO | TODO | TODO | `~0.07` | |
| 4-node linear network (Tier 3.1) | TODO | TODO | TODO | exact eigenvalues | |

### Map systems (lyapax vs. ChaosTools.jl only)

| System | lyapax | ChaosTools.jl | Exact | Notes |
|---|---|---|---|---|
| Logistic map `r=4` (Tier 0.2) | TODO | TODO | `ln 2 = 0.693147` | |
| Hénon map, `sum(λ)` (Tier 0.3) | TODO | TODO | `ln 0.3 = -1.203973` | |

### DDE systems (lyapax vs. jitcdde)

| System | lyapax | jitcdde | Reference | Notes |
|---|---|---|---|---|
| Linear scalar DDE (Tier 4.2) | TODO | TODO | Lambert W root | |
| Mackey-Glass `λ1` (Tier 4.1) | TODO | TODO | order `1e-2`-`1e-3` | qualitative only, see Tier 4.1 caveats |
| Mackey-Glass KY dimension | TODO | TODO | `2`-`3` | |
| 2-node delayed linear network (Tier 4.3) | TODO | TODO (network DDE — check jitcdde supports multi-variable) | Lambert W root (2x2) | jitcdde may need a hand-rolled 2-variable system since it's not natively network-shaped |

## Results — Performance

*Not yet populated with cross-tool numbers.* lyapax-internal numbers
already measured this session (see `notes/milestones.md`, M6) are recorded
here for reference, clearly labeled as internal-only:

| Comparison | Result | Source |
|---|---|---|
| lyapax matrix-free (jvp/vmap) vs. dense jacfwd, 200-node Kuramoto network, `k=5` | **~23x** faster (`5.41s` → `0.24s`, 200 raw steps) | `notes/milestones.md` M6, internal only |
| lyapax `jax.vmap` parameter sweep vs. Python loop, 13-point Kuramoto `G` sweep | **~2.9x** faster (`4.05s` → `1.38s`) | `notes/milestones.md` M6, internal only |

Cross-tool wall-time table (TODO — same systems as the accuracy tables
above, first-call and steady-state columns per tool):

| System | lyapax (warm) | jitcode/jitcdde (warm) | ChaosTools.jl (warm) | lyapax (1st call) | jitcode/jitcdde (1st call) | ChaosTools.jl (1st call) |
|---|---|---|---|---|---|---|
| Lorenz | TODO | TODO | TODO | TODO | TODO | TODO |
| Mackey-Glass | TODO | TODO | — | TODO | TODO | — |
| 200-node network (lyapax only — no natural equivalent in the other tools without hand-building the coupled ODE system) | TODO | — | — | TODO | — | — |

## Open TODOs (resume here)

1. ~~Confirm Julia install finished cleanly.~~ Done —
   `Pkg.add(["ChaosTools", "DynamicalSystemsBase", "OrdinaryDiffEq", "StaticArrays"])`
   completed and precompiled, `using ChaosTools` succeeds. ~~Record exact
   package versions.~~ Done — see Environment table above.
2. Create a `benchmarks/` directory (sibling to `examples/`, `tests/`) with
   one script per tool per system group, e.g.:
   - `benchmarks/jitcode/lorenz.py`, `benchmarks/jitcode/linear_ode.py`, `benchmarks/jitcode/network.py`
   - `benchmarks/jitcdde/mackey_glass.py`, `benchmarks/jitcdde/linear_scalar.py`
   - `benchmarks/chaostools/lorenz.jl`, `benchmarks/chaostools/maps.jl`, `benchmarks/chaostools/network.jl`
   - `benchmarks/lyapax/` (thin wrappers around the existing `examples/` scripts, or reuse them directly)
   - a small `benchmarks/collect_results.py` that runs everything and
     writes a results table (json/csv) this report's tables get filled in
     from, so re-running the whole report later is one command rather than
     manual copy-paste.
3. Run each accuracy table row, fill in the numbers + max-abs-diff.
4. Run each performance table row (warm + first-call, matching
   `plot_07_speed_and_accuracy.py`'s methodology).
5. Write a short "Discussion" section once the tables are populated:
   where lyapax agrees/disagrees, and by how much; where lyapax is
   faster/slower and why (JAX tracing/JIT vs. C compilation vs. Julia's
   own JIT); explicitly call out the network/coupling scale demos
   (`plot_10_matrix_free_scaling.py`, `plot_11_vmap_parameter_sweep.py`)
   as capability the other tools don't have an equivalent one-line API
   for, rather than a number to compare.
6. Once populated, this report is the source for whatever gets published
   (README badge/section, paper, docs page) — keep the raw numbers here
   and write a condensed summary elsewhere rather than duplicating tables.
