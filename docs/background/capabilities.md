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
- **Adaptive-step ODE integration** (optional `adaptive` extra,
  `lyapax.adaptive.diffrax_adaptive_step`) — drops into `ode_problem`
  alongside the fixed-step builtins with no other code changes; `ode_problem`
  only (not `network_problem`, not DDEs), see the gaps below.
- **Batched parameter / initial-condition sweeps** via `jax.vmap`
  (`sweep_lyapunov_spectrum`): a whole parameter grid as one XLA call.
- **Kaplan-Yorke (Lyapunov) dimension** (`lyapax.core.kaplan_yorke_dimension`)
  — a pure post-processing readout of an existing `LyapunovResult.exponents`,
  no extra simulation. Raises rather than silently understating the answer
  if given a partial spectrum (`d_total=`) whose cumulative sum never
  crosses zero within the tracked exponents.
- **Differentiating an exponent w.r.t. a system parameter**
  (`jax.grad`/`jax.jacfwd(lyapunov_spectrum(...).exponents[i])`) — reliable
  for non-chaotic or short-horizon systems (e.g. gradient-based tuning of
  a parameter toward a target exponent); see the caveat below for chaotic
  trajectories.
- **Transparent GPU execution** — JAX picks the backend; no code
  changes. Whether the GPU is faster depends on problem size.
- **A validation suite anchored to independent sources** — exact
  eigenvalues, structural invariants, published literature values; see
  {doc}`validation`.

## What lyapax cannot do

- **No stiff (implicit) ODE integration.** The built-in integrators are
  fixed-step (Euler / Heun / RK4 / RK6); adaptive-step *explicit*
  Runge-Kutta is available for single, uncoupled ODEs (`ode_problem`) via
  the optional `adaptive` extra (`lyapax.adaptive.diffrax_adaptive_step`,
  backed by [diffrax](https://docs.kidger.site/diffrax/)) — not for
  `network_problem` (rejected outright, even for a non-delayed network)
  or DDEs, and there is still no implicit solver, so genuinely stiff
  systems are out of scope regardless. The computed exponents are those
  of the numerical time-`dt` map (or, for the adaptive integrator, of the
  accepted-step sequence under a given `rtol`/`atol`), not the exact
  continuous flow — checking convergence in `dt` (or tolerance) is the
  caller's responsibility (the gallery's speed-and-accuracy and
  adaptive-ODE examples show how). The adaptive integrator's internal
  step-size control is a dynamic-trip-count `while_loop`, so
  differentiating through it needs `jax.jacfwd` (forward-mode);
  `jax.grad`/`jax.jacrev` (reverse-mode) do not work through it.
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
- **Gradients through a *chaotic* trajectory are numerically useless,
  even though they don't raise an error.** `jax.grad`/`jax.jacfwd` of an
  exponent w.r.t. a parameter differentiate through the whole unrolled
  state trajectory, so on a chaotic system the result inherits that
  trajectory's own exponential sensitivity — it grows roughly like
  `exp(lambda_max * horizon)` and is unreliable well before it overflows
  (a known chaotic-sensitivity-analysis phenomenon, not a lyapax bug —
  see `lyapax.core`'s module docstring). Safe for non-chaotic/short-
  horizon systems; always sanity-check a chaotic-system gradient against
  finite differences before trusting it.

(adaptive-integration-speed)=
## Adaptive integration is not a speed optimization

Measured, not assumed: at matched accuracy (not matched `dt`/`rtol`, which
isn't a fair comparison across integrators with different meanings of
"step size"), `diffrax_adaptive_step` was 2-4x slower than fixed-step
`rk4`/`rk6` on a small chaotic system (Lorenz), up to ~4.5x slower on GPU
for that same small system, and still ~4x slower even on a relaxation
oscillator (Van der Pol) with sharply time-varying stiffness — the
scenario adaptive stepping exists for. The overhead ratio does shrink at
large state dimension (`d=1500-3000`), approaching rough parity, but never
produced a clear win in any case tested.

Root cause: the internal accept/reject `jax.lax.while_loop` pays real
per-step overhead (PID controller update, solver-step bookkeeping) that a
fixed-step raw step doesn't, and avoiding extra fixed-step steps only pays
off in wall-clock terms if each one was expensive to begin with — which
isn't the case for any system small enough to fit the exact/published
references this page and {doc}`validation` rely on.

Use the adaptive integrator for tolerance-driven accuracy control (no
manual `dt`-convergence sweep) or because it's the only forward-mode-
differentiable adaptive option here — not for throughput.
