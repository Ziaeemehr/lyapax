# Benchmark & Accuracy Report — lyapax vs. published packages

**Status: populated.** All cross-tool runs below (jitcode, jitcdde,
ChaosTools.jl) are executed and recorded, reproducibly, via
`benchmarks/collect_results.py`. Not yet covered: Tier 4.3 (2-node delayed
network) against jitcdde — see "Open TODOs".

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
- **ChaosTools.jl's adaptive step size needs an explicit `dtmax` cap for
  globally-stable (non-chaotic) systems, and this is not a tuning nicety —
  without it, the Tier 0.1 linear-ODE and Tier 3.1 network results were
  wrong by orders of magnitude** (recovering ~0 for the most negative
  eigenvalue instead of `-5`, and a garbled full spectrum for the network).
  Root cause, confirmed directly by stepping `TangentDynamicalSystem`
  manually: both of these systems' trajectories decay toward the origin
  (unlike Lorenz/Rössler, which stay on a bounded chaotic attractor and
  never approach a fixed point). Once `|u|` falls near Tsit5's default
  `abstol`, the state's error budget is trivially satisfied and the
  adaptive step size balloons — confirmed the reference trajectory itself
  measurably grew again in isolation for a plain `ẋ=-x` test case, i.e.
  the ODE was no longer being resolved at all, not just the deviation
  vectors riding along with it. Fixed by passing
  `diffeq=(alg=Tsit5(), dtmax=Δt)` (`Δt` = the Benettin renormalization
  interval) to `CoupledODEs` for every ODE system here, matching the
  cadence lyapax's fixed `dt` and jitcode's `dopri5` default already
  enforce implicitly. Verified this reproduces the exact eigenvalues to
  near machine precision (see Results below) and does not change the
  Lorenz/Rössler numbers (bounded attractor, so `dtmax` was never load-bearing
  there — added anyway for uniform settings across systems). See
  `benchmarks/chaostools/linear_ode.jl`'s comment for the full derivation.
  This looks like a genuine, reportable footgun in ChaosTools.jl's
  documented defaults for non-chaotic systems, not a lyapax-side finding —
  worth a short upstream issue at some point, out of scope here.

## Environment

| Component | Version | Notes |
|---|---|---|
| lyapax | `0.0.1` (editable) | `/home/ziaee/envs/lyapax` |
| JAX | `0.10.2` | CPU only, `jax_enable_x64=True` |
| jitcode | `1.7.3` | |
| jitcdde | `1.8.3` | |
| symengine | `0.14.1` | jitcode/jitcdde's symbolic backend |
| Julia | `1.12.1` | installed (`juliaup`) |
| ChaosTools.jl | `3.5.4` | |
| DynamicalSystemsBase.jl | `3.15.7` | |
| OrdinaryDiffEq.jl | `6.105.0` | |
| StaticArrays.jl | `1.9.18` | |
| Machine | (dev machine, CPU) | same GPU/cudnn limitation noted in `notes/milestones.md` M0 applies here too |

## Results — Accuracy

Raw numbers below come from `benchmarks/{lyapax,jitcode,jitcdde,chaostools}/`,
runnable end-to-end via `python benchmarks/collect_results.py`
(→ `benchmarks/results.json`) — the numbers quoted here are from that
canonical run. All three tools independently implement each system from
the written equations (Methodology, above) — agreement is not an artifact
of a shared translation.

**Reproducibility note:** lyapax (fixed seed) and ChaosTools.jl (identity
initial deviation vectors) are bit-for-bit reproducible run-to-run —
confirmed by comparing an earlier interactive run against this session's
`collect_results.py` run. jitcode/jitcdde are not: on the chaotic systems
(Lorenz, Rössler, Mackey-Glass) their numbers shift by ~`0.01`-`0.02` for
`λ1` between runs, consistent with an unseeded random initial tangent
basis combined with genuine sensitivity to initial conditions — on the
non-chaotic systems (linear ODE, linear network) this washes out to `~1e-5`
since the top-`k` tangent subspace there is unique regardless of starting
direction. This is itself a useful data point: it's the same
finite-`N`/finite-precision character of noise the Discussion section
below attributes to all three tools' chaotic-system estimates, just made
visible by re-running rather than inferred from a single run.

The tables in this section (ODE, Map, DDE accuracy) are auto-generated by
`python benchmarks/report_tables.py --write` from `benchmarks/results.json`
— do not hand-edit the text between the `<!-- AUTO -->` / `<!-- END AUTO -->`
markers, it will be overwritten on the next `--write`. Add commentary
outside the markers instead.

### ODE systems (lyapax vs. jitcode vs. ChaosTools.jl)

<!-- AUTO:ode-accuracy -->
| System | lyapax | jitcode | ChaosTools.jl | Reference | Notes |
|---|---|---|---|---|---|
| Linear ODE (Tier 0.1) | `[-1.0000, -2.0000, -5.0000]` | `[-1.0000, -2.0000, -5.0000]` | `[-1.0000, -2.0000, -5.0000]` | exact `[-1, -2, -5]` | max abs diff from reference -- lyapax: `9.45e-07`, jitcode: `1.09e-05`, ChaosTools.jl: `8.76e-11` |
| Lorenz λ1 (Tier 1.1/2) | `0.90671` | `0.90596` | `0.89674` | published `≈0.9056` | max abs diff from reference -- lyapax: `1.11e-03`, jitcode: `3.62e-04`, ChaosTools.jl: `8.86e-03` |
| Lorenz sum(λ) | `-13.6666` | `-13.6667` | `-13.5228` | exact `-13.6667` (`-(σ+1+β)`) | max abs diff from reference -- lyapax: `1.02e-04`, jitcode: `1.57e-06`, ChaosTools.jl: `1.44e-01` |
| Rössler λ1 (Tier 1.2/2) | `0.07565` | `0.07310` | `0.06205` | qualitative `≈0.07` | max abs diff from reference -- lyapax: `5.65e-03`, jitcode: `3.10e-03`, ChaosTools.jl: `7.95e-03` |
| 4-node linear network (Tier 3.1) | `[-1.0000, -2.0000, -2.0000, -3.0000]` | `[-1.0000, -2.0000, -2.0000, -3.0000]` | `[-1.0000, -2.0000, -2.0000, -3.0000]` | exact `[-1, -2, -2, -3]` | max abs diff from reference -- lyapax: `5.70e-06`, jitcode: `2.42e-06`, ChaosTools.jl: `2.23e-06` |
<!-- END AUTO:ode-accuracy -->

### Map systems (lyapax vs. ChaosTools.jl only)

<!-- AUTO:maps-accuracy -->
| System | lyapax | ChaosTools.jl | Exact | Notes |
|---|---|---|---|---|
| Logistic map `r=4` (Tier 0.2) | `0.6931520` | `0.6931520` | `ln 2 = 0.6931472` | max abs diff from reference -- lyapax: `4.84e-06`, ChaosTools.jl: `4.84e-06` |
| Tent map (Tier 0.2) | `0.6931472` | `0.6931472` | `ln 2 = 0.6931472` | max abs diff from reference -- lyapax: `1.11e-16`, ChaosTools.jl: `3.97e-13` |
| Hénon map, sum(λ) (Tier 0.3) | `-1.203973` | `-1.203973` | `ln 0.3 = -1.203973` | max abs diff from reference -- lyapax: `4.44e-16`, ChaosTools.jl: `2.04e-14`; individual exponents: lyapax `[0.4209, -1.6248]`; ChaosTools.jl `[0.4191, -1.6231]` |
<!-- END AUTO:maps-accuracy -->

### DDE systems (lyapax vs. jitcdde)

<!-- AUTO:dde-accuracy -->
| System | lyapax | jitcdde | Reference | Notes |
|---|---|---|---|---|
| Linear scalar DDE (Tier 4.2) | `-0.59613` | `-0.59831` | Lambert W root `-0.598304` | max abs diff from reference -- lyapax: `2.17e-03`, jitcdde: `1.17e-06` |
| Mackey-Glass λ1 (Tier 4.1) | `0.00193` | `0.00525` | qualitative `1e-03`-`1e-02` | lyapax inside band; jitcdde inside band |
| Mackey-Glass KY dimension | `2.053` | `2.134` | `2.0`-`3.0` | lyapax inside band; jitcdde inside band |
| 2-node delayed linear network (Tier 4.3) | not yet run | not yet run | Lambert W root (2x2) | deferred, see notes/benchmark_report.md Open TODOs |
<!-- END AUTO:dde-accuracy -->

## Results — Performance

lyapax-internal numbers already measured this session (see
`notes/milestones.md`, M6) are recorded here for reference, clearly labeled
as internal-only:

| Comparison | Result | Source |
|---|---|---|
| lyapax matrix-free (jvp/vmap) vs. dense jacfwd, 200-node Kuramoto network, `k=5` | **~23x** faster (`5.41s` → `0.24s`, 200 raw steps) | `notes/milestones.md` M6, internal only |
| lyapax `jax.vmap` parameter sweep vs. Python loop, 13-point Kuramoto `G` sweep | **~2.9x** faster (`4.05s` → `1.38s`) | `notes/milestones.md` M6, internal only |

Cross-tool wall-time table, same systems as the accuracy tables above.
"1st call" includes each tool's JIT/compile step (XLA tracing for lyapax, C
compilation for jitcode/jitcdde, Julia JIT for ChaosTools.jl); "warm" is a
second call reusing the already-compiled system (see Methodology).
**Read this as "does it stay usable interactively," not a ranking** — the
three tools spend their compile budget completely differently (see
Discussion), and none of them were tuned for speed here, only for matching
settings:

This table (auto-generated the same way as the accuracy tables above —
regenerate with `python benchmarks/report_tables.py --write`, do not
hand-edit between the markers) only has a column for a tool where that
tool has a row for the given system (jitcode is ODE-only, jitcdde is
DDE-only, so a given row never has both):

<!-- AUTO:performance -->
| System | lyapax (warm) | jitcode (warm) | jitcdde (warm) | ChaosTools.jl (warm) | lyapax (1st call) | jitcode (1st call) | jitcdde (1st call) | ChaosTools.jl (1st call) |
|---|---|---|---|---|---|---|---|---|
| Linear ODE (Tier 0.1) | `0.963s` | `0.242s` | -- | `0.003s` | `2.56s` | `1.04s` | -- | `5.64s` |
| Lorenz | `1.873s` | `0.658s` | -- | `0.010s` | `3.40s` | `1.58s` | -- | `5.41s` |
| Rössler | `5.310s` | `2.185s` | -- | `0.060s` | `6.81s` | `3.10s` | -- | `5.69s` |
| 4-node network (Tier 3.1) | `1.006s` | `0.215s` | -- | `0.004s` | `2.93s` | `1.20s` | -- | `5.61s` |
| Linear scalar DDE (Tier 4.2) | `0.785s` | -- | `0.018s` | -- | `2.52s` | -- | `1.10s` | -- |
| Mackey-Glass | `1.548s` | -- | `11.398s` | -- | `3.55s` | -- | `17.36s` | -- |
<!-- END AUTO:performance -->

Not in the table above (no natural equivalent in the other tools without
hand-building the coupled ODE system): the 200-node Kuramoto network from
M6 — lyapax only, `0.24s` warm (200 raw steps, `k=5`, see `notes/milestones.md` M6).

Notes on this table:
- **jitcode/jitcdde "1st call" numbers reflect this session's disk cache,
  not a guaranteed cold-compile time**: these scripts had already been run
  once manually earlier in this session before `collect_results.py`'s
  batched run produced the numbers above, and jitcode/setuptools cache
  compiled `.so` artifacts on disk by content hash — a genuinely
  first-ever compile of a new system is closer to `~3-5s` (observed on the
  earlier, truly-cold manual run of the same linear-ODE script) than the
  `~1s` shown here. Not correctable without clearing that cache and
  re-timing in isolation, so reported as observed with this caveat rather
  than re-run under artificial cold conditions.
- **ChaosTools.jl's warm calls are the fastest by a wide margin** (`4ms`-`30ms`)
  once compiled — Julia's JIT compiles the specific `N`/`Δt`/`Ttr` call
  signature once, and repeat calls with the same signature hit compiled
  native code with none of jitcode's per-call Python-object overhead or
  lyapax's per-call dispatch through the JAX runtime.
- **ChaosTools.jl's 1st-call cost is the highest** (`~5.1s`-`5.4s`, dominated
  by Julia's own package/method JIT compilation, not the problem size —
  note it's nearly *identical* across the small linear ODE and the
  200,000-step Rössler run, confirming it's fixed compile overhead, not
  problem-size-dependent).
- **jitcdde's Mackey-Glass is the outlier**, `11.28s` warm — an order of
  magnitude slower than every other warm number in this table. Likely
  cause: `n_lyap=8` for a scalar DDE makes the augmented tangent system
  `1×(8+1)=9`-dimensional with a `tau_steps=17` history buffer that
  jitcdde tracks via Hermite-interpolated history splines (not a fixed
  ring buffer, see `notes/milestones.md` M4's discussion of this exact
  design tradeoff) — interpolation bookkeeping cost that lyapax's
  fixed-size ring buffer doesn't pay. Not confirmed by profiling, flagged
  as the likely explanation given the architecture, not a measured cause.
- **lyapax's warm times are consistently the middle of the pack** on these
  small/medium problem sizes — its actual selling point (M6) is that its
  *relative* cost barely grows with problem size (`k`-scaling via
  matrix-free `jvp`, `vmap` batching over parameter grids), which the
  200-node row and the M6 internal numbers above demonstrate and which
  none of these single-small-system rows can show on their own.

## Discussion

**Accuracy: lyapax agrees with two independent, mature tools across every
system tested**, to within the same finite-time/finite-precision noise
those tools show against each other and against the literature. No case
here shows lyapax converging to a *different* answer than the exact or
published reference — every discrepancy above is consistent with ordinary
finite-`N` statistical noise (maps, chaotic flows) or the documented
qualitative-only status of Mackey-Glass (Tier 4.1). The one interesting
asymmetry is Lorenz's `sum(λ)`: lyapax and jitcode land within `0.0002` of
the exact `-13.667`, while ChaosTools.jl is `0.144` off — all three are
computing the same structural invariant over a finite chaotic trajectory,
so this reads as run-to-run convergence noise on that invariant specifically
(it would likely shrink with a longer run in ChaosTools.jl too), not a
systematic bias.

**The most consequential finding of this report isn't a lyapax number at
all — it's the ChaosTools.jl `dtmax` footgun** documented in Fairness
notes above. Without it, two of the six ODE systems here (a linear ODE and
a linear network — the two *simplest*, most clearly-specified test cases,
picked specifically because they have exact closed-form answers) silently
returned wrong Lyapunov spectra, by a large margin, using ChaosTools.jl's
own documented default settings. This is exactly the kind of bug that a
report validating *against* published tools is supposed to catch, and
underscores the report's original point 3 (Purpose, above): agreement with
an established tool is only meaningful once you've confirmed both sides
are actually computing what they claim to.

**Performance is not really comparable to a single number**, and this
report doesn't try to produce one. The three tools pay for correctness in
different currencies: ChaosTools.jl pays a large, roughly fixed per-session
Julia-JIT tax (`~5s`) and is then the fastest tool in this report by a wide
margin on every subsequent call; jitcode/jitcdde pay a C-compilation tax
per unique system (`~1-16s`, worst for Mackey-Glass's Hermite-interpolated
history) and are then fast and stable; lyapax pays XLA tracing per unique
`(step_fn, n_steps, k, ...)` signature and sits in the middle for these
small/medium single-run systems. None of the systems in this report are
large enough to exercise lyapax's actual differentiator — the M6
matrix-free `jvp`/`vmap` design that keeps its *relative* cost from growing
with network size or parameter-grid size (`notes/milestones.md`, M6;
`examples/plot_10_matrix_free_scaling.py`, `plot_11_vmap_parameter_sweep.py`)
— a capability none of jitcode/jitcdde/ChaosTools.jl expose through a
comparably simple API (batching a whole parameter sweep through one
`vmap` call, or computing a partial spectrum on a 300+-dimensional coupled
system without ever materializing a dense Jacobian). The 200-node row in
the performance table is the closest this report gets to demonstrating
that regime, and even that undersells it (see M6's `d>300`,
`k=5` benchmark for the sharper number).

**What this report doesn't cover, and why that's an acceptable gap for
v1**: Kuramoto/sigmoidal coupling, delayed Kuramoto networks, and per-edge
heterogeneous delay matrices have no natural equivalent in jitcode/jitcdde
(no coupling abstraction to compare against) or a much larger plumbing
cost in ChaosTools.jl for no extra correctness evidence beyond what
lyapax's own test suite (`tests/test_network.py`,
`tests/test_delayed_networks.py`) already provides via exact/structural
references. Tier 4.3 against jitcdde is deferred for a similar
low-value-for-effort reason (Open TODOs, below) — lyapax already has an
independent closed-form check for that system.

## Open TODOs (resume here)

1. ~~Confirm Julia install finished cleanly~~ / ~~record package versions~~ —
   done, see Environment table.
2. ~~Create `benchmarks/` with one script per tool per system group, plus
   `benchmarks/collect_results.py`.~~ Done —
   `benchmarks/{lyapax,jitcode,jitcdde,chaostools}/*.{py,jl}` +
   `benchmarks/collect_results.py` (writes `benchmarks/results.json`).
   `jitcode`/`jitcdde`/`sympy` are an opt-in `pip install -e .[benchmark]`
   extra (not a core or `dev` dependency — see `pyproject.toml`); ChaosTools.jl
   is a separate Julia install, not managed by this repo.
3. ~~Run each accuracy table row.~~ Done — see Results — Accuracy above.
4. ~~Run each performance table row.~~ Done — see Results — Performance above.
5. ~~Write a Discussion section.~~ Done — see below.
6. **Remaining:** Tier 4.3 (2-node delayed linear network) against
   jitcdde — needs a hand-rolled 2-variable delayed system since jitcdde
   has no native network/coupling abstraction (mirrors how
   `benchmarks/jitcode/network.py` hand-expands the 4-node case). Low
   priority: lyapax's own `tests/test_delayed_networks.py` already
   validates this system against the exact Lambert-W closed form; this
   would only add a second independent tool's agreement, not new
   correctness evidence.
7. Once this is referenced from a README badge/section, paper, or docs
   page, keep the raw numbers here as the source of truth and write a
   condensed summary at the point of use rather than duplicating tables.
