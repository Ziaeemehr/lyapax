# Milestones — JAX Lyapunov Exponent Package (ODE + DDE)

Working name in this doc: **`lyapax`** (placeholder — rename freely). New, standalone
package rooted in this repo (`/home/ziaee/git/lyapunov`). It does **not** depend on
`vbi` at import time; the parts of `vbi`'s JAX backend that are relevant are
**vendored and trimmed**, not imported, per the decision to keep this repo's
release cycle independent of `vbi_develop_auto`.

> **Caution carried over from this session:** the numeric results sitting in
> `lyapunov-master/` (Wolf/Sandri C++ and Python implementations) are *not*
> to be treated as validated ground truth. Reuse the algorithmic structure
> (variational-equation + QR/Gram-Schmidt renormalization) as a design
> reference only. All correctness claims in this plan are anchored to
> independent sources: analytic results, structural invariants, and
> published literature values — see `notes/validation_systems.md`.

## Why this is more tractable than a from-scratch build

`/home/ziaee/git/inference/vbi_develop_auto/vbi/simulator` already implements almost
exactly the architecture you described from TVB (model dynamics and coupling
defined separately, ODE and DDE sharing one code path):

- `spec/model.py` — `ModelSpec`: state vars + params + a coupling variable
  list (`cvar`) + `dfun_str` (plain-text per-state-variable RHS expressions).
- `spec/coupling.py` — `CouplingSpec`: linear / sigmoidal / kuramoto coupling
  kernels, delay-agnostic.
- `spec/connectivity.py` — `Connectivity`: weights + tract lengths + speed →
  integer-step delay matrix (`delay_steps`) and required ring-buffer depth
  (`horizon`).
- `backend/jax_/codegen.py` — compiles `dfun_str` into a plain JAX function
  via `exec()`; nothing about it blocks `jacfwd`/`jvp`.
- `backend/jax_/simulator.py` — `_make_step_fn(...)` returns a pure
  `step(carry, _) -> (new_carry, new_state)` used inside `lax.scan`. Delay
  coupling reads a ring buffer (`buf`) via a flat-index gather
  (`_read_delayed_coupling`); no-delay coupling reads `state` directly. Both
  paths share the *same* `step` closure — the ODE case is literally the DDE
  case with `horizon == 1`.

This means the hard part — a JAX-native, jit/vmap-friendly integrator with a
coupling abstraction that unifies ODE and DDE — **already exists and works**.
What's missing is only the tangent-space (Lyapunov) layer on top, plus
validation. That reframes the difficulty:

- **ODE Lyapunov spectrum**: low-to-moderate. `step` is already a pure
  function of `(state, params)`; autodiff gives the Jacobian for free (no
  more hand-derived Jacobians like the ~30-line block in
  `lyapunov-master/kuramoto/py/jitcode/main.py`'s Hindmarsh-Rose `Jac`
  construction, which is exactly the kind of error-prone code this replaces).
- **DDE Lyapunov spectrum**: moderate. The key insight: since the ring
  buffer `buf` is part of the `carry`, the pair `(state, buf)` is the
  complete Markovian state of the delayed system. Differentiating the whole
  `step(carry, _)` with `jax.jvp` w.r.t. `(state, buf)` automatically
  propagates sensitivity through the delayed gather — this **is** the
  discretized-map method for DDE Lyapunov exponents (Farmer, 1982) done for
  free by autodiff, with no hand-derived delayed variational equation
  (no separate "J0, J1" terms to get right). This is the same trick used
  for the C++ Hermite-interpolated history buffer in
  `lyapunov-master/DDE_VanderPole/cpp_wolf/src/dde_solver.cpp`, just letting
  JAX do the differentiation instead of deriving it by hand.
- diffrax (JAX's ODE library) has **no DDE support** as of this writing (open
  feature request, [diffrax#406](https://github.com/patrick-kidger/diffrax/issues/406)),
  confirming that a fixed-step, ring-buffer-based custom integrator (as
  vendored from `vbi`) is currently the only practical JAX-native DDE path,
  not a gap specific to this project.

Net estimate: a solid, tested v1 covering fixed-step ODE + fixed-delay DDE,
full or partial spectrum, single or networked nodes — **on the order of a
few focused weeks**, not a large undertaking, because the integrator half is
already solved.

## Cross-cutting technical risks (apply to every milestone)

1. **Precision.** `vbi`'s JAX backend defaults to `float32`
   (`IntegratorSpec.jax_dtype`). Lyapunov exponents are long-horizon
   log-growth-rate averages — float32 error accumulates and will quietly
   corrupt exponents past a fairly short horizon. **`lyapax` must default to
   `jax_enable_x64=True` / float64**, unlike the vendored source. Flag any
   API that silently allows float32.
2. **Tangent vector blow-up / collapse.** Between renormalizations, tangent
   vectors grow like `exp(λ_max · renorm_every · dt)` or shrink toward zero
   for the least-aligned directions. `renorm_every` must be small enough to
   avoid overflow (matches the caution already noted in the jitcode docs the
   user is familiar with). Needs a runtime check/warning, not just docs.
3. **Non-smooth ops in the vendored step function.** `jnp.clip` (state
   bounds) and stochastic noise injection are either non-differentiable at
   the boundary or not meaningfully compatible with a deterministic Lyapunov
   spectrum. v1 restricts to `stochastic=False` and either drops the clip or
   asserts it never activates on the validation trajectories (add a runtime
   check, not a silent pass-through).
4. **Delay/step alignment.** The vendored ring buffer uses **integer-step**
   delays (`round(tract_length / (speed * dt))`) — no sub-step interpolation
   (unlike the Hermite interpolation in the old C++ DDE code). This is
   simpler and fully autodiff-friendly, but means `tau` is only exact up to
   `dt` rounding. Needs a documented accuracy tradeoff and a convergence
   test (LE estimate vs. decreasing `dt` should converge to the same value
   for a fixed physical `tau`).
5. **QR vs. Gram-Schmidt.** Use `jnp.linalg.qr` (as in the Sandri-style code
   in `lyapunov-master/lorenz/py/Sandri/src/pyLyapunov.py`) rather than a
   hand-rolled Gram-Schmidt (as in `lyapunov-master/hindmarsh/wolf/py/src/lib.py`)
   — mathematically equivalent, but QR is one call, vectorizes cleanly under
   `vmap`, and avoids re-deriving the classical/modified Gram-Schmidt
   numerical-stability tradeoffs.
6. **Sign/order convention.** `jnp.linalg.qr`'s `Q` is only defined up to
   column sign; must take `abs(diag(R))` before the log (already done
   correctly in the reference Sandri code) and sort the final spectrum
   descending.

## M0 — Scaffolding & design lock-in — ✅ done

- Repo layout: `src/lyapax/vendored/`, `tests/`; `pyproject.toml` (setuptools,
  src-layout, `pytest` config). Package installs editable
  (`pip install -e .`) into `vbienv` (jax 0.10.0, the dev/test env used for
  `vbi` too).
- `tests/conftest.py` sets `JAX_PLATFORMS=cpu` and
  `jax.config.update("jax_enable_x64", True)` at import time (before jax
  picks a backend), not inside a fixture — fixtures run too late to affect
  backend selection.
- **GPU note:** this dev machine's GPU fails with
  `INTERNAL: RET_CHECK ... dnn_support != nullptr` on any real op (cudnn/driver
  mismatch, confirmed during M0) even though `jax.devices()` lists a
  `CudaDevice`. CPU works cleanly with `float64`. `lyapax` dev/tests default
  to CPU via `JAX_PLATFORMS=cpu`; GPU verification is out of scope until M6,
  and will need a working cudnn install first (infra issue, not a `lyapax`
  bug).
- Vendored (copied, trimmed) from `vbi` into `src/lyapax/vendored/`:
  `model_spec.py` (`ModelSpec`/`StateVar`/`Parameter`), `coupling.py`
  (`CouplingSpec` — linear only — and a trimmed `Connectivity` with
  `delay_steps`/`horizon`), `step.py` (`build_jax_dfun` codegen +
  `make_step_fn`, the ring-buffer `step(carry, _)` factory). Dropped:
  monitors, stimuli, BOLD/Balloon-Windkessel, the sweeper, stochastic noise,
  state clipping, sigmoidal/kuramoto coupling — see
  `src/lyapax/vendored/NOTICE.md` for the full provenance/diff list
  (vbi is Apache-2.0; NOTICE.md carries the required attribution).
- **Definition of done — met:** `pytest tests/` passes 4/4:
  `test_x64_enabled`, `test_running_on_cpu`, and two plumbing checks —
  an uncoupled linear-decay trajectory matches `exp(gamma·t)` analytically
  (validates dfun codegen + Heun integrator), and a 2-node delayed network
  runs end-to-end through the ring-buffer path with finite output (validates
  the DDE-shaped plumbing before any Lyapunov math sits on top of it in M1/M4).

## M1 — Core Benettin/QR engine, single-node ODE — ✅ done

- `lyapax.core.lyapunov_spectrum(step_fn, state0, dt, n_steps, k=None, renorm_every=1, t_transient=0.0, seed=0) -> LyapunovResult(exponents, history, times)`
  in `src/lyapax/core.py`. `step_fn` is a plain `state (d,) -> new_state (d,)`
  function (parameters closed over by the caller) — deliberately independent
  of the vendored carry-based `step(carry, _)` from M0, since single-node
  ODE needs none of the coupling/ring-buffer machinery. M3 will need a
  thin adapter from the vendored step to this shape; M4/M5 will need a
  variant that also carries tangent state for the delay buffer.
- Tangent propagation: `(d, k)` tangent matrix `Y`, random-orthonormal
  init, updated each raw step via `jax.jacfwd(step_fn)(state) @ Y` (dense,
  as planned — matrix-free deferred to M6).
- Every `renorm_every` steps: QR-decompose `Y`, accumulate `log|diag(R)|`,
  replace `Y` with `Q`.
- `k < d` (partial spectrum) already supported — this is M2's headline
  feature but fell out of the M1 design for free, so M2 is now mostly just
  "write more tests for it" (done — see `test_partial_spectrum_matches_full_spectrum_leading_columns`).
- **Non-obvious fix found via testing:** the transient must evolve *and
  periodically renormalize* the tangent matrix `Y` jointly with the state,
  not just run the plain trajectory forward and start tangent-tracking
  fresh afterward. Reason: even for a non-chaotic linear system with no
  attractor to relax onto, a randomly-initialized `Y0` needs time to align
  with the eigendirections (Oseledets subspaces) — skipping this biased the
  3-eigenvalue linear test by ~9% (`test_linear_system_distinct_real_eigenvalues`
  went from failing to passing once transient handling was fixed). This
  wasn't visible in the original naive design taken from
  `lyapunov-master`'s `ttrans` handling (state-only transient), which is
  exactly the kind of not-solid-findings gap flagged for that codebase.
  Implemented as `_advance()` in `core.py`, reused for both the transient
  and each accumulation block so a long transient still gets periodic
  QR-renormalization and can't overflow (risk #2).
- **Definition of done — met:** `tests/test_lyapunov_core.py`, 9/9 passing
  (plus the 4 from M0, 13/13 total, ~10s):
  - Tier 0.1: 3-distinct-real-eigenvalue linear system, exact to `2e-3`;
    complex-conjugate-pair linear system, exact to `5e-3`.
  - Tier 0.2: logistic map (`r=4`) and tent map, both within `0.02` of
    the exact `ln(2)`.
  - Tier 0.3: Hénon map, sum of exponents within `5e-3` of the exact
    `ln(0.3)` identity.
  - Tier 1.1 + Tier 2: Lorenz — sum of exponents within `0.05` of the exact
    `-(σ+1+β)`; `λ1` within `0.08` of the published `0.9056`; `|λ2| < 0.03`.
  - Tier 1.2 + Tier 2: Rössler — `λ1 ∈ (0.02, 0.12)`; `|λ2| < 0.02`.
  - Cross-check: `k=1` (top-1) matches `k=3` (full spectrum)'s leading
    exponent to `0.01` on Lorenz — both are independent finite-time
    estimates of the same quantity, not required to be bit-identical.
- Benchmark systems (`linear_system`, `logistic_map`, `tent_map`,
  `henon_map`, `lorenz`, `rossler`) live in `src/lyapax/systems.py` as
  plain flat-vector JAX functions, deliberately *not* routed through the
  vendored `ModelSpec`/coupling machinery (that indirection only pays for
  itself once coupling is actually involved, in M3+). A general-purpose
  fixed-step RK4 integrator (better accuracy than the vendored Euler/Heun
  for chaotic-flow validation) lives in `src/lyapax/integrators.py`.

## M2 — Partial spectrum + convergence diagnostics — mostly folded into M1

- `k < d` support and the running-estimate `history`/`times` output both
  fell out of M1's design directly (`lyapunov_spectrum(..., k=..)` and
  `LyapunovResult.history`) and are already tested
  (`test_partial_spectrum_matches_full_spectrum_leading_columns`). What's
  left here, if anything: a convenience helper for a relative-change
  stopping criterion / convergence plot, and top-2 (not just top-1)
  regression tests. Low priority — revisit only if M3+ work exposes a real
  need for it.

## M3 — Coupling + multi-node networks (still zero-delay) — ✅ done

- **Scope decision (this session):** include linear, sigmoidal, and
  Kuramoto coupling now rather than deferring sigmoidal/Kuramoto to M7.
  Rationale: all three are smooth/differentiable, so `jax.jacfwd` handles
  the tangent math with no extra derivation per kind (unlike
  `lyapunov-master`'s hand-derived-Jacobian approach) — the marginal cost
  of supporting them is low, and sigmoidal coupling in particular is close
  to a hard requirement for anything resembling a real neural-mass/TVB
  model (`jr_sigmoidal` deliberately excluded — it's Jansen-Rit-specific
  and can wait for an actual JR-based use case).
- **Extensibility decision (this session):** do **not** replicate `vbi`'s
  closed-enum coupling dispatch. In both `vbi`'s numpy and JAX backends,
  `CouplingSpec.kind` is a fixed string literal dispatched through a
  hardcoded `if/elif` duplicated in each backend (`jr_sigmoidal` is even
  special-cased to skip the `G` multiply) — a user cannot add a new
  coupling kind without patching `vbi`'s source in two places. This is an
  asymmetry with `ModelSpec`, which is already open (`dfun_str` is
  arbitrary user-authored math, no fixed "model kind" enum). `lyapax`
  instead makes coupling a **plain callable** with a fixed signature,
  e.g. `coupling_fn(cvar_state_or_delayed, weights, params) -> coupling`,
  that the step function just calls. `linear`/`sigmoidal`/`kuramoto` ship
  as builder functions returning that callable, but a user's own function
  with the same signature works with zero library changes — no registry,
  no enum to extend. (The same fix would apply to `vbi`'s `CouplingSpec`,
  e.g. mirroring the `register_model()` extension point that already
  exists for models — but that's a separate, `vbi`-side decision, not
  made here.)
- Implemented as `src/lyapax/coupling.py` (`linear_coupling`,
  `sigmoidal_coupling`, `kuramoto_coupling` builders — each returns a
  `coupling_fn(cvar_state, weights, params) -> coupling`) +
  `src/lyapax/network.py` (`make_network_step_fn`, which reshapes a flat
  `(n_sv*n_nodes,)` state to/from `(n_sv, n_nodes)` around a dfun +
  coupling + Euler/Heun step). Deliberately **not** built on the vendored
  ring-buffer `step(carry, _)` from M0 — that shape carries delay-buffer
  state needed from M4 on, and `lyapax.core.lyapunov_spectrum` wants a
  plain flat `state -> new_state` map, so M3 reshapes directly instead.
  This means the vendored `make_step_fn`'s zero-delay branch (hardcoded
  linear-only) is now duplicated logic, not reused — flagged for M4 to
  resolve by giving the ring-buffer step a `coupling_fn` parameter too,
  unifying with M3's design rather than keeping two coupling mechanisms.
- **Definition of done — met:** `tests/test_network.py`, 5/5 passing
  (18/18 total across the whole suite, ~13s):
  - Tier 3.1 exact test: a 4-node linear network (4-cycle graph, symmetric
    weights) matches `eigvalsh(gamma*I + G*W)` to `3e-3`.
  - The same exact test repeated with a **hand-written coupling function**
    (no import from `lyapax.coupling` at all) instead of `linear_coupling()`
    — concrete proof of the "no registry needed" extensibility claim,
    directly answering the custom-coupling question from this session.
  - Sigmoidal coupling: finite-output smoke test (no equally clean
    closed-form reference exists yet).
  - Kuramoto coupling: an **exact** check at `G=0` — decoupled phase
    oscillators have `dtheta/dt = omega` (state-independent), so the step
    map's Jacobian is exactly the identity and every Lyapunov exponent
    must be exactly `0` (`atol=1e-8`, no literature value needed at all);
    plus a finite-output smoke test at `G>0`.

## M4 — DDE support (fixed delay, fixed step)

- Reintroduce the ring buffer into the state carried through `lax.scan`.
  Tangent state becomes `(Y_state, Y_buf)`; propagate via `jax.jvp` on the
  *whole* `step(carry, _)` (state **and** buffer), not `jacfwd` on state
  alone — this is the point where the "just differentiate the existing
  simulator" trick pays off instead of deriving `∂f/∂x(t)` and
  `∂f/∂x(t-τ)` by hand.
- QR-renormalize the stacked `(state, buffer-slice-in-play)` tangent
  vectors; only the delay-relevant slice of the buffer needs to be tracked,
  not the full ring.
- **Definition of done:** Tier 4 — Mackey-Glass equation reproduces a
  positive largest exponent in the range reported by Farmer (1982) and
  later replications (see `notes/validation_systems.md` for the tolerance
  band — DDE LE literature values are not as tightly reproducible
  digit-for-digit as ODE ones, so the acceptance bar is sign + order of
  magnitude + qualitative spectrum shape, not 4-digit matching).
- Convergence-vs-`dt` test for a fixed physical `tau`, to characterize the
  integer-step delay rounding error from risk #4 above.

## M5 — Delayed networks (per-edge delay matrix)

- Combine M3 + M4: multiple nodes, per-edge `tract_length`/delay, coupling
  through the ring buffer exactly as in `vbi`'s `_read_delayed_coupling`.
- **Definition of done:** delay → 0 limit recovers M3 results (regression
  test, not a new external reference); a small symmetric 2-node delayed
  network has a semi-analytically tractable characteristic equation usable
  as an approximate independent check (see `notes/validation_systems.md`
  Tier 4b).

## M6 — Performance & usability

- Matrix-free tangent propagation via `jax.jvp` (avoid materializing the
  full Jacobian) for large `d * n_nodes` — needed once network size grows
  past the toy validation cases.
- `vmap` over parameter grids / initial conditions for LE-vs-parameter
  sweeps (bifurcation-diagram-style), mirroring `vbi`'s `JaxSweeper` pattern
  but not reusing its code (per the vendoring decision).
- GPU smoke test (no correctness change expected, just confirm it runs).
- Packaging: `pyproject.toml`, README, one or two example notebooks.

## M7 — Stretch goals (not required for v1)

- Adaptive-step ODE integration (diffrax) + Benettin, as an alternative to
  fixed-step for stiff/multi-timescale ODE systems.
- State-dependent or distributed delays.
- Sigmoidal/kuramoto coupling kernels in the DDE tangent path.

## Explicit non-goals for v1

- Stochastic (noise-driven) Lyapunov exponents / finite-time LE under noise.
- PDEs / spatiotemporal chaos.
- Sub-step delay interpolation (Hermite, as in the old C++ code) — v1 uses
  integer-step delays only; revisit if the convergence-vs-`dt` test in M4
  shows the rounding error is unacceptable for target use cases.

## Open decisions to confirm before/while implementing

- [x] Package layout: vendor-a-copy (not a `vbi` dependency) — decided this
      session; scaffolded in M0.
- [x] Package name: `lyapax` — kept as the working name after M0 scaffolding;
      still trivial to rename (one directory + `pyproject.toml` field) if a
      better name comes up.
- [x] Minimum JAX version: `>=0.10`, matching the version installed and
      tested against in `vbienv` (0.10.0).
- [ ] GPU: currently unusable on the dev machine (cudnn/driver mismatch,
      see M0). Not blocking — v1 targets CPU correctness first (M1-M5); GPU
      is M6 and needs the environment fixed independently of this codebase.
- [ ] Whether M1's Jacobian should default to `jacfwd` (dense, simple) or
      `jvp`-per-column from the start — recommend starting dense (M1-M5) and
      only switching to matrix-free in M6 once there's a concrete network
      size that needs it; premature matrix-free code adds complexity before
      it's needed.
