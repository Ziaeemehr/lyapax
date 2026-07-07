# Capabilities and limitations

A candid summary of what lyapax does and does not do, so you can decide
quickly whether it fits your problem.

## What lyapax can do

- **Full or partial Lyapunov spectra for ODEs.** Track all $d$ exponents
  or only the leading $k \le d$, for a hand-written JAX right-hand side
  or a `ModelSpec`/`build_jax_dfun`-compiled symbolic one; cost scales
  with $k$, not $d$ (see {doc}`lyapax_implementation`).
- **Coupled networks.** Arbitrary topology (a weight matrix), any
  coupling rule — built-in linear / sigmoidal / Kuramoto, or a
  user-written callable — with fixed-step Euler / Heun / RK4 / RK6
  integration.
- **Fixed-delay DDE networks.** Both a single *uniform* delay shared by
  every edge (`network_dde_problem(..., tau=...)`) and a genuine
  per-edge heterogeneous delay matrix via the lower-level simulator
  path — the latter currently only with the built-in linear coupling
  (see the gaps below).
- **Grid-snapped or Hermite-interpolated DDE history** — fast-and-simple
  versus exact-$\tau$, higher-order-accurate; see
  {doc}`lyapax_implementation` for the tradeoff.
- **Discrete chaotic maps** (logistic, Hénon, or any user map) — same
  engine, `dt=1.0` per iterate, no integrator involved.
- **Batched parameter / initial-condition sweeps** via `jax.vmap`
  (`sweep_lyapunov_spectrum`): a whole parameter grid as one XLA call.
- **Transparent GPU execution** — JAX picks the backend; no code
  changes. Whether the GPU is faster depends on problem size.
- **A validation suite anchored to independent sources** — exact
  eigenvalues, structural invariants, published literature values; see
  {doc}`validation`.

## What lyapax cannot do

- **No adaptive or stiff ODE integration.** Every integrator is
  fixed-step (Euler / Heun / RK4 / RK6); there is no adaptive-step or
  implicit solver. The computed exponents are those of the numerical
  time-`dt` map, not the exact continuous flow — checking convergence
  in `dt` is the caller's responsibility (the gallery's
  speed-and-accuracy example shows how).
- **DDE delays must be known and fixed.** No state-dependent or
  distributed delays. Grid-snapped mode further rounds $\tau$ to an
  integer multiple of `dt`; only `interpolate=True` removes that
  restriction, and only on the uniform-delay path.
- **Per-edge delays cannot be combined with a custom coupling rule.**
  Heterogeneous per-edge delay matrices currently work only with the
  built-in linear coupling; a delayed per-edge Kuramoto network, for
  example, would need an edge-aware coupling signature that does not
  exist yet.
- **No stochastic / noise-driven Lyapunov exponents.** Noise injection
  is a deliberate non-goal: it is non-smooth and not meaningfully
  compatible with the deterministic tangent-space spectrum computed
  here.
- **No PDEs or spatiotemporal chaos.** State is always a
  finite-dimensional vector (or an `(n_state_vars, n_nodes)` array for
  networks); there is no spatial discretization support.
- **`history` column ordering is not stable for near-degenerate
  exponents.** Columns are ordered once, by the final row; exponents
  that nearly cross can swap order mid-run. See
  `LyapunovResult.history`'s docstring.
- **`dfun_str` model specs are compiled with `exec()`.** `ModelSpec`
  string right-hand sides are code-generated without sanitization —
  fine for specs you write yourself, **not safe** for specs built from
  untrusted input.
