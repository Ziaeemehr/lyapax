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
| System | lyapax | lyapax (RK6) | jitcode | ChaosTools.jl | Reference | Notes |
|---|---|---|---|---|---|---|
| Linear ODE (Tier 0.1) | `[-1.0000, -2.0000, -5.0000]` | `[-1.0000, -2.0000, -5.0000]` | `[-1.0000, -2.0000, -5.0000]` | `[-1.0000, -2.0000, -5.0000]` | exact `[-1, -2, -5]` | max abs diff from reference -- lyapax: `9.45e-07`, lyapax (RK6): `9.45e-07`, jitcode: `1.16e-05`, ChaosTools.jl: `8.76e-11` |
| Lorenz λ1 (Tier 1.1/2) | `0.90172` | `0.90878` | `0.90105` | `0.89674` | published `≈0.9056` | max abs diff from reference -- lyapax: `3.88e-03`, lyapax (RK6): `3.18e-03`, jitcode: `4.55e-03`, ChaosTools.jl: `8.86e-03` |
| Lorenz sum(λ) | `-13.6666` | `-13.6667` | `-13.6667` | `-13.5228` | exact `-13.6667` (`-(σ+1+β)`) | max abs diff from reference -- lyapax: `1.02e-04`, lyapax (RK6): `1.79e-09`, jitcode: `1.56e-06`, ChaosTools.jl: `1.44e-01` |
| Rössler λ1 (Tier 1.2/2) | `0.07080` | `0.08017` | `0.07895` | `0.06205` | qualitative `≈0.07` | max abs diff from reference -- lyapax: `7.98e-04`, lyapax (RK6): `1.02e-02`, jitcode: `8.95e-03`, ChaosTools.jl: `7.95e-03` |
| 4-node linear network (Tier 3.1) | `[-1.0000, -2.0000, -2.0000, -3.0000]` | `[-1.0000, -2.0000, -2.0000, -3.0000]` | `[-1.0014, -1.9986, -2.0000, -3.0000]` | `[-1.0000, -2.0000, -2.0000, -3.0000]` | exact `[-1, -2, -2, -3]` | max abs diff from reference -- lyapax: `3.53e-06`, lyapax (RK6): `3.53e-06`, jitcode: `1.36e-03`, ChaosTools.jl: `2.23e-06` |
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
| Linear scalar DDE (Tier 4.2) | `-0.60050` | `-0.60050` | `-0.59830` | Lambert W root `-0.598304` | max abs diff from reference -- lyapax: `2.19e-03`, lyapax (RK6): `2.19e-03`, jitcdde: `6.51e-07` |
| Mackey-Glass λ1 (Tier 4.1) | `0.00796` | `0.00727` | `0.00502` | qualitative `1e-03`-`1e-02` | lyapax inside band; lyapax (RK6) inside band; jitcdde inside band |
| Mackey-Glass KY dimension | `2.206` | `2.191` | `2.129` | `2.0`-`3.0` | lyapax inside band; lyapax (RK6) inside band; jitcdde inside band |
| 2-node delayed linear network (Tier 4.3) | not yet run | not yet run | not yet run | Lambert W root (2x2) | deferred -- not yet run against jitcdde |
<!-- END AUTO:dde-accuracy -->

## Performance

Wall-clock time, reported both as a first call (includes JIT trace/
compile time for `lyapax`/`jitcode`/`jitcdde`) and a warm call
(steady-state, post-compile), so a one-time compilation cost doesn't
distort the number meant to characterize steady-state throughput. The
GPU columns re-run the same `lyapax` scripts with a CUDA backend rather
than CPU — see `benchmarks/collect_results.py` for how that pass is
skipped (with a warning, not a failure) when no working GPU is found.

<!-- AUTO:performance -->
| System | lyapax (warm) | lyapax (RK6) (warm) | lyapax (GPU) (warm) | lyapax (RK6, GPU) (warm) | jitcode (warm) | jitcdde (warm) | ChaosTools.jl (warm) | lyapax (1st call) | lyapax (RK6) (1st call) | lyapax (GPU) (1st call) | lyapax (RK6, GPU) (1st call) | jitcode (1st call) | jitcdde (1st call) | ChaosTools.jl (1st call) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Linear ODE (Tier 0.1) | `0.259s` | `0.457s` | `0.919s` | `1.460s` | `0.190s` | -- | `0.003s` | `1.46s` | `0.48s` | `2.66s` | `1.54s` | `0.88s` | -- | `5.05s` |
| Lorenz | `0.372s` | `0.693s` | `1.737s` | `3.487s` | `0.589s` | -- | `0.010s` | `1.58s` | `0.66s` | `3.41s` | `3.43s` | `1.45s` | -- | `5.12s` |
| Rössler | `0.420s` | `0.681s` | `4.827s` | `9.486s` | `2.002s` | -- | `0.029s` | `1.62s` | `0.70s` | `6.33s` | `9.36s` | `2.79s` | -- | `5.12s` |
| 4-node network (Tier 3.1) | `0.403s` | `0.691s` | `1.175s` | `2.290s` | `0.188s` | -- | `0.004s` | `1.60s` | `0.68s` | `2.94s` | `2.36s` | `1.10s` | -- | `5.51s` |
| Linear scalar DDE (Tier 4.2) | `0.322s` | `0.424s` | `0.733s` | `0.853s` | -- | `0.020s` | -- | `1.58s` | `0.43s` | `2.37s` | `0.90s` | -- | `1.01s` | -- |
| Mackey-Glass | `0.563s` | `0.865s` | `1.442s` | `1.964s` | -- | `10.149s` | -- | `2.06s` | `0.87s` | `3.47s` | `2.01s` | -- | `16.05s` | -- |
<!-- END AUTO:performance -->

The plots below are generated from the same `benchmarks/results.json`
snapshot as the tables above; regenerate both together with
`python benchmarks/report_plots.py`.

### Where lyapax comes out ahead

`lyapax` and the compiled-C tools (`jitcode`/`jitcdde`) both pay a
one-time trace/compile cost, then run a tight numerical loop for every
call after — so the warm-call time is the fairer comparison for
steady-state throughput, and `jitcode`/`jitcdde` are the tools closest
to `lyapax`'s own execution model (unlike `ChaosTools.jl`, which has
near-zero call overhead on these small toy systems and is the fastest
tool for most of them regardless). The chart below expresses each
system as a speedup ratio (competitor warm time ÷ lyapax warm time) on
a log scale: a bar above the `1×` line means lyapax was faster on that
system, below means it wasn't — every system is shown, not just the
favorable ones.

![lyapax speedup vs. jitcode/jitcdde/ChaosTools.jl, per benchmark system](/_static/benchmarks_speedup.png)

### Full performance picture

For absolute context (not just relative to lyapax): every tool's warm
wall-clock time per system, log-scaled since the range spans several
orders of magnitude.

![Warm wall-clock time per system, all tools](/_static/benchmarks_performance.png)
