# Benchmarks

How `lyapax`'s Lyapunov-exponent estimates and wall-clock performance
compare against other tools that compute the same quantity, on the same
benchmark systems `lyapax`'s own test suite is validated against (see
{doc}`validation`).

The tables on this page are generated, not hand-written, and are
refreshed periodically rather than on every commit or doc build — treat
any specific number below as a snapshot, not a live guarantee. To
reproduce or refresh them yourself:

```bash
python benchmarks/collect_results.py   # runs every tool, writes benchmarks/results.json
python benchmarks/report_tables.py --write   # regenerates the tables below from that file
```

`collect_results.py` needs optional dependencies this doc build doesn't
carry (a Julia install with ChaosTools.jl, `jitcode`/`jitcdde`, and,
for the GPU column, an actual GPU) — see
`benchmarks/collect_results.py`'s module docstring for the full
requirements. That's also why these tables are committed as a static
snapshot (`benchmarks/results.json`) instead of being executed as part
of building this documentation.

## Comparison targets

- **[jitcode](https://pypi.org/project/jitcode/)** — the established
  Python reference for ODE Lyapunov spectra (symbolic → compiled C).
- **[jitcdde](https://pypi.org/project/jitcdde/)** — the DDE counterpart,
  and the closest existing precedent for `lyapax`'s own delayed-system
  engine.
- **[ChaosTools.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/chaostools/stable/)**
  — the most widely used dynamical-systems toolbox in Julia; covers both
  continuous and discrete (map) systems.
- **Published/analytic references** — exact eigenvalues, Lambert-W
  roots, or literature values, wherever a benchmark system has one.

None of these share `lyapax`'s exact integration scheme or default
tolerances, so this is a cross-validation of the computed exponents
against independent implementations, not a claim that every tool is
configured identically. Step size, transient length, total run length,
and QR-renormalization interval are matched as closely as each tool's
API allows; where a tool's own defaults differ meaningfully (e.g.
adaptive step control), that's left as-is rather than silently
normalized away.

## Accuracy

Each row compares the estimated Lyapunov exponent(s) (or, for chaotic
systems without a closed-form spectrum, a structural invariant like
`sum(λ)`) against an exact or published reference, across whichever
tools support that system class (map-only systems have no `jitcode`
column since it's ODE-only; DDE systems have no ChaosTools.jl column
since it has no delay-equation support).

### ODE systems

<!-- AUTO:ode-accuracy -->
| System | lyapax | lyapax (RK6) | jitcode | ChaosTools.jl | ChaosTools.jl (RK4) | ChaosTools.jl (Vern6) | Reference | Notes |
|---|---|---|---|---|---|---|---|---|
| Linear ODE (Tier 0.1) | `[-1.0000, -2.0000, -5.0000]` | `[-1.0000, -2.0000, -5.0000]` | `[-1.0000, -2.0000, -5.0000]` | `[-1.0000, -2.0000, -5.0000]` | `[-1.0000, -2.0000, -5.0000]` | `[-1.0000, -2.0000, -5.0000]` | exact `[-1, -2, -5]` | max abs diff from reference -- lyapax: `9.45e-07`, lyapax (RK6): `9.45e-07`, jitcode: `3.23e-06`, ChaosTools.jl: `8.76e-11`, ChaosTools.jl (RK4): `2.80e-11`, ChaosTools.jl (Vern6): `2.03e-12` |
| Lorenz λ1 (Tier 1.1/2) | `0.90172` | `0.90878` | `0.90782` | `0.89674` | `0.90632` | `0.91334` | published `≈0.9056` | max abs diff from reference -- lyapax: `3.88e-03`, lyapax (RK6): `3.18e-03`, jitcode: `2.22e-03`, ChaosTools.jl: `8.86e-03`, ChaosTools.jl (RK4): `7.17e-04`, ChaosTools.jl (Vern6): `7.74e-03` |
| Lorenz sum(λ) | `-13.6666` | `-13.6667` | `-13.6667` | `-13.5228` | `-13.6666` | `-13.6667` | exact `-13.6667` (`-(σ+1+β)`) | max abs diff from reference -- lyapax: `1.02e-04`, lyapax (RK6): `1.79e-09`, jitcode: `1.55e-06`, ChaosTools.jl: `1.44e-01`, ChaosTools.jl (RK4): `1.03e-04`, ChaosTools.jl (Vern6): `1.82e-09` |
| Rössler λ1 (Tier 1.2/2) | `0.07080` | `0.08017` | `0.07244` | `0.06205` | `0.07545` | `0.07264` | qualitative `≈0.07` | max abs diff from reference -- lyapax: `7.98e-04`, lyapax (RK6): `1.02e-02`, jitcode: `2.44e-03`, ChaosTools.jl: `7.95e-03`, ChaosTools.jl (RK4): `5.45e-03`, ChaosTools.jl (Vern6): `2.64e-03` |
| 4-node linear network (Tier 3.1) | `[-1.0000, -2.0000, -2.0000, -3.0000]` | `[-1.0000, -2.0000, -2.0000, -3.0000]` | `[-1.0000, -2.0000, -2.0000, -3.0000]` | `[-1.0000, -2.0000, -2.0000, -3.0000]` | `[-1.0000, -2.0000, -2.0000, -3.0000]` | `[-1.0000, -2.0000, -2.0000, -3.0000]` | exact `[-1, -2, -2, -3]` | max abs diff from reference -- lyapax: `3.53e-06`, lyapax (RK6): `3.53e-06`, jitcode: `3.83e-06`, ChaosTools.jl: `2.23e-06`, ChaosTools.jl (RK4): `2.18e-06`, ChaosTools.jl (Vern6): `2.18e-06` |
<!-- END AUTO:ode-accuracy -->

### Map systems

<!-- AUTO:maps-accuracy -->
| System | lyapax | ChaosTools.jl | Exact | Notes |
|---|---|---|---|---|
| Logistic map `r=4` (Tier 0.2) | `0.6931520` | `0.6931520` | `ln 2 = 0.6931472` | max abs diff from reference -- lyapax: `4.84e-06`, ChaosTools.jl: `4.84e-06` |
| Tent map (Tier 0.2) | `0.6931472` | `0.6931472` | `ln 2 = 0.6931472` | max abs diff from reference -- lyapax: `1.11e-16`, ChaosTools.jl: `3.97e-13` |
| Hénon map, sum(λ) (Tier 0.3) | `-1.203973` | `-1.203973` | `ln 0.3 = -1.203973` | max abs diff from reference -- lyapax: `2.22e-16`, ChaosTools.jl: `2.04e-14`; individual exponents: lyapax `[0.4193, -1.6233]`; ChaosTools.jl `[0.4191, -1.6231]` |
<!-- END AUTO:maps-accuracy -->

### DDE systems

<!-- AUTO:dde-accuracy -->
| System | lyapax | lyapax (RK6) | jitcdde | Reference | Notes |
|---|---|---|---|---|---|
| Linear scalar DDE (Tier 4.2) | `-0.60050` | `-0.60050` | `-0.59831` | Lambert W root `-0.598304` | max abs diff from reference -- lyapax: `2.19e-03`, lyapax (RK6): `2.19e-03`, jitcdde: `8.01e-07` |
| Mackey-Glass λ1 (Tier 4.1) | `0.00796` | `0.00727` | `0.00531` | qualitative `1e-03`-`1e-02` | lyapax inside band; lyapax (RK6) inside band; jitcdde inside band |
| Mackey-Glass KY dimension | `2.206` | `2.191` | `2.137` | `2.0`-`3.0` | lyapax inside band; lyapax (RK6) inside band; jitcdde inside band |
| 2-node delayed linear network (Tier 4.3) | not yet run | not yet run | not yet run | Lambert W root (2x2) | deferred -- not yet run against jitcdde |
<!-- END AUTO:dde-accuracy -->

## Performance

Steady-state per-call wall-clock time (each tool's JIT/compile step has
already run by the time these numbers are measured). The GPU columns
re-run the same `lyapax` scripts with a CUDA backend rather than CPU —
see `benchmarks/collect_results.py` for how that pass is skipped (with a
warning, not a failure) when no working GPU is found.

The default `ChaosTools.jl` column uses its own idiomatic choice
(`Tsit5`, adaptive step control) — a different algorithm from `lyapax`'s
fixed-step `rk4`/`rk6`, so that timing isn't a same-method comparison
(see the "Comparison targets" fairness note above). The **`ChaosTools.jl
(RK4)`** and **`ChaosTools.jl (Vern6)`** columns close that gap: they run
`OrdinaryDiffEq.jl`'s `RK4()` and `Vern6()` (the exact tableau `lyapax`'s
`rk6_step` implements — see `lyapax.integrators.rk6_combine`'s
docstring) with `adaptive=false` at `lyapax`'s own fixed `dt`, so the two
sides are running the literal same algorithm and step size. The gap
that remains at that point is a genuine implementation/runtime
difference (Julia/LLVM vs. JAX/XLA at these small state sizes), not an
algorithm mismatch.

<!-- AUTO:performance -->
| System | lyapax | lyapax (RK6) | lyapax (GPU) | lyapax (RK6, GPU) | jitcode | jitcdde | ChaosTools.jl | ChaosTools.jl (RK4) | ChaosTools.jl (Vern6) |
|---|---|---|---|---|---|---|---|---|---|
| Linear ODE (Tier 0.1) | `0.264s` | `0.449s` | `0.942s` | `1.518s` | `0.193s` | -- | `0.006s` | `0.007s` | `0.010s` |
| Lorenz | `0.393s` | `0.659s` | `1.790s` | `3.470s` | `0.640s` | -- | `0.010s` | `0.021s` | `0.037s` |
| Rössler | `0.461s` | `0.750s` | `4.782s` | `9.454s` | `2.159s` | -- | `0.031s` | `0.076s` | `0.123s` |
| 4-node network (Tier 3.1) | `0.429s` | `0.761s` | `1.189s` | `2.304s` | `0.203s` | -- | `0.004s` | `0.012s` | `0.018s` |
| Linear scalar DDE (Tier 4.2) | `0.330s` | `0.462s` | `0.692s` | `0.911s` | -- | `0.021s` | -- | -- | -- |
| Mackey-Glass | `0.652s` | `0.895s` | `1.450s` | `2.100s` | -- | `10.564s` | -- | -- | -- |
<!-- END AUTO:performance -->

The plot below is generated from the same `benchmarks/results.json`
snapshot as the tables above; regenerate it with
`python benchmarks/report_plots.py`. It's every tool's steady-state
wall-clock time per system, log-scaled since the range spans several
orders of magnitude -- `jitcode`/`jitcdde` are the tools closest to
`lyapax`'s own execution model (a one-time trace/compile, then a tight
numerical loop for every call after); `ChaosTools.jl` has near-zero
per-call overhead on these small toy systems and is the fastest tool
for most of them regardless.

![Warm wall-clock time per system, all tools](/_static/benchmarks_performance.png)
