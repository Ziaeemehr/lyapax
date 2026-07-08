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

- Repo layout: `src/lyapax/simulator/`, `tests/`; `pyproject.toml` (setuptools,
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
- Vendored (copied, trimmed) from `vbi` into `src/lyapax/simulator/`:
  `model_spec.py` (`ModelSpec`/`StateVar`/`Parameter`), `coupling.py`
  (`CouplingSpec` — linear only — and a trimmed `Connectivity` with
  `delay_steps`/`horizon`), `step.py` (`build_jax_dfun` codegen +
  `make_step_fn`, the ring-buffer `step(carry, _)` factory). Dropped:
  monitors, stimuli, BOLD/Balloon-Windkessel, the sweeper, stochastic noise,
  state clipping, sigmoidal/kuramoto coupling — see
  `src/lyapax/simulator/NOTICE.md` for the full provenance/diff list
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
  ODE needs none of the coupling/ring-buffer machinery. M4 needed (and got)
  a variant that also carries tangent state for the delay buffer
  (`lyapax.dde.lyapunov_spectrum_dde`) — **correction, this session:** the
  prediction here that "M3 will need a thin adapter from the vendored step"
  turned out wrong; M3 built its own separate flat-state step function
  (`lyapax.network.make_network_step_fn`) instead of adapting the vendored
  one, which is why M4 later found `network.py` and `simulator/step.py`
  independently reimplementing byte-for-byte identical `_euler`/`_heun`
  helpers — see the M5 section below.
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

## M4 — DDE support (fixed delay, fixed step) — ✅ done

- **Two design passes this session; this section describes the final one.**
  The first pass (superseded, no longer in the codebase) used state
  augmentation — folding a sliding history window into a flat `state` fed
  through the unmodified M1 dense-`jacfwd` engine. It worked but built a
  second, parallel DDE mechanism instead of reusing the already-correct,
  already-tested vendored ring buffer (`lyapax.simulator.step`), and its
  dense tangent propagation (O(horizon²) per step) doesn't scale to real
  coupled delayed networks, which is the actual target use case (raised in
  review: "the whole story was to be able to work on large systems of
  coupled equations"). Replaced with the design below, which reuses the
  vendored ring buffer as the single simulator and adds only the piece
  that was genuinely missing — a tangent-propagation engine that
  differentiates through the whole `(state, buf)` carry, not `state` alone
  (differentiating `state` only, as the first pass effectively did by
  construction, silently drops the `∂f/∂x(t-τ)` sensitivity term — a
  correctness bug, not an approximation).
- **Precedent check:** this matches how the established `jitcdde_lyap`
  package computes DDE Lyapunov exponents (Farmer 1982's method — augment
  the system with explicit tangent dynamics, propagate alongside the
  primal, periodically renormalize). It stores history as Hermite-
  interpolation anchors (needed because it targets general *adaptive*-step
  DDEs); our scope is fixed-step with an integer-step delay, so the
  delayed value always lands exactly on a stored grid point, making a
  finite-size ring buffer the correct, simpler specialization — and the
  tangent space is already finite-dimensional, so a plain `jnp.linalg.qr`
  suffices where jitcdde needs a continuous-function inner product.
- **`src/lyapax/simulator/step.py`** (additive, backward-compatible —
  `tests/test_setup.py`'s two call sites use only keyword args, so the
  original `G_default`/`coup_a`/`coup_b`/`delay_steps`-based behavior is
  byte-for-byte unchanged when the new params are omitted; see
  `simulator/NOTICE.md` for the deviation writeup): `make_step_fn` gained
  `coupling_fn` (a `lyapax.coupling`-style plain callable, replacing the
  hardcoded linear formula) and `tau_steps` (a single global delay). A new
  `_read_uniform_delayed_cvar` (O(1) modular ring-buffer read) handles the
  delayed branch when `coupling_fn` is given; the original per-edge
  `_read_delayed_coupling` is untouched, still used when `coupling_fn` is
  `None`. **Scope limit:** per-edge heterogeneous delays together with a
  custom `coupling_fn` aren't supported yet — `_read_delayed_coupling`
  bakes the coupling formula into the per-edge gather inline (different
  edges read different time offsets, so there's no single per-node
  "delayed cvar_state" to hand a plain `coupling_fn`); that needs an
  edge-aware `coupling_fn` signature, a real design fork left for M5.
- **`src/lyapax/dde.py`** (rewritten): `lyapunov_spectrum_dde(step_fn,
  state0, buf0, params, dt, n_steps, k=None, renorm_every=1,
  t_transient=0.0, seed=0) -> LyapunovResult`, operating on
  `lyapax.simulator.make_step_fn(..., coupling_fn=..., tau_steps=...)`'s
  carry-based step. Tangent state `(Y_state, Y_buf)` mirrors the primal
  carry's shapes plus a trailing `k` axis; per raw step, all `k` tangent
  columns propagate via one `jax.vmap`-batched `jax.jvp` call — O(k)
  forward passes per step, not O(d_total) for the full augmented
  dimension, which is the actual fix for the scaling problem the first
  pass had. Every `renorm_every` steps, flatten+stack `(Y_state, Y_buf)`,
  `jnp.linalg.qr`, accumulate `log|diag(R)|`, split `Q` back. Also
  `resolve_tau_steps`/`constant_history_buf0` helpers. Two subtleties
  caught by an independent design-review pass before implementation (both
  silent-wrong-answer classes of bug, no crash/NaN):
  - The ring-buffer step counter `t` **must live in the scanned carry and
    be re-read every raw step**, not closed over once per
    `renorm_every`-block — verified directly that closing over a stale `t`
    diverges from the correct answer with no error raised.
  - The transient must cover **at least one full ring cycle**
    (`horizon * dt`) — until every buffer slot has been written at least
    once from real dynamics, delayed-direction tangent information is
    incomplete, and a random initial tangent basis is disproportionately
    concentrated in the buffer's trivial shift-register directions.
    `lyapunov_spectrum_dde` enforces this as an internal floor
    (`max(t_transient, horizon*dt)`), so a too-short user-supplied
    `t_transient` degrades gracefully rather than silently biasing the
    result (`test_transient_floor_prevents_bias_from_short_user_transient`).
  - The whole ring buffer's tangent is tracked (not a "delay-relevant
    slice"): for the per-edge-delay case this extends to later, any slot
    can become delay-relevant to some node at some future step, so no
    fixed subset is safe to drop. (For today's uniform-delay case this is
    also simply the correct, simplest choice.)
  - The shared "renorm block → history/exponents" tail (previously
    duplicated near-verbatim from `core.py`) is factored into
    `core._run_renorm_scan`, used by both engines.
- The two Tier-4 benchmarks are expressed as 1-node self-loop
  `ModelSpec`/coupling networks (`weights=[[1.0]]`, `linear_coupling`) —
  "coupled to your own delayed history" — living in `tests/test_dde.py`
  (mirroring `test_network.py`'s locally-defined `ModelSpec` helpers), not
  in `src/lyapax/systems.py`, since they now need the ModelSpec/coupling
  machinery that module deliberately avoids.
- **Definition of done — met:** `tests/test_dde.py`, 7/7 passing (25/25
  total across the whole suite, ~22s):
  - Tier 4.2: linear scalar DDE (`a=0.5, tau=0.3`) — dominant exponent
    within `0.01` of the Lambert W root (`lambda = W(-a*tau)/tau`).
  - `dt`-convergence (same physical `tau`, `dt` in `{2e-2, 1e-2}`) and
    small-delay-recovers-ODE-decay regression, both as in the first pass.
  - `test_transient_floor_prevents_bias_from_short_user_transient` and
    `test_tangent_propagation_matches_dense_jacfwd` (new): the latter
    directly validates the `jvp`/`vmap` tangent-propagation mechanism
    against dense `jax.jacfwd` on a small case, independent of any
    downstream statistical convergence — agreement to `1e-10`, i.e.
    machine precision. Addresses the earlier gap where only final LE
    *values*, never the Jacobian mechanism itself, were validated.
  - Tier 4.1: Mackey-Glass (`beta=0.2, gamma=0.1, n=10, tau=17, dt=1.0`,
    `k=8` — not the full `d_total=18` spectrum, since the most contracting
    directions underflow `log|diag(R)|` before they're needed, the same
    numerical-sensitivity `jitcdde`'s docs flag for deep negative
    exponents) — `lambda1` in `(0.0005, 0.05)` (observed `~0.002-0.0024`),
    `|lambda2| < 0.01` (observed `~1e-4`), remaining spectrum negative,
    Kaplan-Yorke dimension in `(1.5, 3.5)` (observed `~2.06`, matching the
    literature's `2-3` range). Robust across PRNG seed and initial
    condition.
  - `test_delayed_network_benchmark_scale` (new): a 10-node delayed
    Kuramoto network, `d_total > 300` (deliberately beyond what dense
    `jacfwd` handles practically per step) — finite output, `k=5`,
    completes in ~2s locally (30s CI ceiling). The concrete demonstration
    that this redesign actually scales to coupled networks, which the
    first pass could not.
- `scipy` added to the `dev` extra in `pyproject.toml` (for
  `scipy.special.lambertw` in the Tier 4.2 test).
- **Follow-up (this session, API consistency):** raised in review — the ODE
  side lets a non-networked system skip coupling machinery entirely
  (`systems.py`'s plain `rhs(state) -> dstate`, no `ModelSpec` needed
  unless there's an actual network, M3+), but the DDE side as first shipped
  forced *every* system through `ModelSpec`/`coupling_fn`/the vendored
  step, even a single scalar equation like Mackey-Glass, because the ring
  buffer only existed inside that machinery. Fixed with a convenience
  layer mirroring `systems.py` + `integrators.rk4_step`'s relationship on
  the ODE side, without adding a second DDE mechanism (still the same
  vendored ring buffer underneath):
  - `lyapax.dde.make_scalar_delayed_step_fn(rhs_delayed, m, tau_steps, dt)`
    — builds the trivial "coupled to your own delayed history" 1-node
    self-loop wiring internally (raw `dfun`, identity `coupling_fn`, no
    `ModelSpec`) around `lyapax.simulator.make_step_fn`, from a plain
    `rhs_delayed(state_now, state_delayed) -> dstate`.
  - `lyapax.dde.scalar_delayed_history0(state0_now, tau_steps)` — the
    matching `(state0, buf0)` builder from a flat initial condition.
  - `mackey_glass`/`linear_scalar_dde` restored to `src/lyapax/systems.py`
    (Tier 4 section) as the math-only `rhs_delayed` builders that feed it,
    same role `lorenz`/`rossler` play for the ODE case.
  - `tests/test_dde.py`'s scalar-DDE tests rewritten against this new
    front door (shorter, no `ModelSpec` boilerplate);
    `test_delayed_network_benchmark_scale` kept on the direct
    `ModelSpec`/`coupling_fn` path deliberately, for coverage of both
    entry points into the one shared engine.
  - **Bug caught while making this change:** `lyapunov_spectrum_dde`'s
    transient-floor logic (`horizon*dt` minimum, see above) was gated
    behind `if t_transient > 0.0`, so a caller passing exactly
    `t_transient=0.0` skipped the transient block entirely — floor never
    applied, defeating the guarantee. Fixed to run unconditionally (DDE
    transients, unlike the ODE case, are never legitimately skippable).
    `test_transient_floor_prevents_bias_from_short_user_transient` predates
    the fix and happened to still pass (the first accumulation block's own
    internal QR provided partial, accidental correction) — worth noting as
    a reminder that a passing regression test doesn't always mean the
    code path it's named for was actually exercised.

## M5 — Delayed networks (per-edge delay matrix) — ✅ done

- **Scope smaller than originally sketched (verified before writing any M5
  code):** `lyapunov_spectrum_dde` has no opinion on delay structure — it
  differentiates through whatever carry `step_fn` produces, so it already
  runs correctly against a genuine per-edge (heterogeneous, asymmetric)
  `delay_steps` matrix via the untouched legacy `_read_delayed_coupling`
  path (`coupling_fn=None`), with no new engine code — spot-checked
  directly on an asymmetric 2-node linear network
  (`delay_steps=[[0,10],[30,0]]`), finite output, before writing the tests
  below. M5's real remaining work was mostly **validation**, plus one
  cleanup (below) — not new engine architecture. The one still-genuine gap
  (already flagged in M4, unchanged): a *custom* `coupling_fn`
  (sigmoidal/Kuramoto-style) combined with per-edge delays needs the
  edge-aware `coupling_fn` signature fork noted in the M4 simulator/step.py
  bullet — linear per-edge-delayed coupling works today via the legacy path.
- **Cleanup folded in:** `lyapax.network.make_network_step_fn` (M3,
  zero-delay only) used to carry its own `_euler`/`_heun`, byte-for-byte
  identical to `lyapax.simulator.make_step_fn`'s — built that way because
  M3 predates M4's carry-based Lyapunov engine, so at the time there was
  no way to feed a carry-based step into anything. Retired: `network.py`
  now delegates to `simulator.make_step_fn(has_delays=False, ...)` via a
  thin carry-to-flat adapter (for `has_delays=False` the vendored step's
  `buf` never changes and `t` is never read, so closing over a fixed
  placeholder `buf`/`t` and only threading `state` through
  `lyapunov_spectrum`'s `jacfwd` is exactly equivalent to differentiating
  the whole carry — no tangent information lost). `_euler`/`_heun` now
  exist in exactly one place. All 5 pre-existing `test_network.py` tests
  pass unchanged, confirming the refactor is behavior-preserving.
- **Definition of done — met:** `tests/test_delayed_networks.py`, 2/2
  passing (27/27 total across the whole suite):
  - `test_per_edge_delay_near_zero_recovers_m3_eigenvalues`: the Tier 3.1
    4-node linear network, run two ways — M3's zero-delay
    `lyapunov_spectrum`, and M5's per-edge `delay_steps` matrix (uniform
    value `tau_steps=1`, the smallest resolvable delay, but still routed
    through the general per-edge machinery, not the uniform-`tau_steps`
    shortcut) via `lyapunov_spectrum_dde`. Both match the exact
    eigenvalues to `3e-3`, and match each other to `1e-3`.
  - `test_two_node_symmetric_delayed_network_matches_lambert_w` (Tier 4.3):
    a symmetric 2-node delayed linear network
    (`x1'=gamma*x1+G*x2(t-tau)`, `x2'=gamma*x2+G*x1(t-tau)`). The
    symmetric-mode ansatz (`x1=x2`) gives
    `lambda_sym = gamma + W(G*tau*e^{-gamma*tau})/tau`; the antisymmetric
    mode (`x1=-x2`) gives `lambda_antisym = gamma +
    W(-G*tau*e^{-gamma*tau})/tau` (**note:** the sign flips inside the `W`
    argument, not on the whole term — verified numerically before
    committing to this formula; an earlier draft of this note had it
    imprecisely as `gamma +/- W(...)`). Matches to `5e-3` at
    `gamma=-1, G=0.5, tau=0.3`.
- **Follow-up, post-M5 (this session): `src/lyapax/vendored/` renamed to
  `src/lyapax/simulator/`**, along with two new example demos
  (`examples/plot_08_delayed_coupling.py` — per-edge delay sweep against
  the Tier 4.3 Lambert W solution; `examples/plot_09_kuramoto_delayed_network.py`
  — delayed Kuramoto network, extending `plot_05`). Rationale: `vendored`
  named the module after *where the code came from* (copied from `vbi`),
  which reads fine in a design doc but is a confusing name to import and
  call day-to-day; `simulator` names it after *what it does* (compiles
  `ModelSpec`/`Connectivity` into the ring-buffer JAX step function).
  Provenance is unaffected — `simulator/NOTICE.md` still carries the
  vbi attribution — this was a rename, not a re-scoping. All `lyapax.vendored`
  import paths (source, tests, examples) and path references in this doc
  were updated together (verified: all 27 tests still pass, both new
  example scripts run standalone and reproduce their closed-form checks).

## M6 — Performance & usability

- **Matrix-free tangent propagation via `jax.jvp` for the plain ODE/zero-delay-network
  path — ✅ done (this session).** Was already done for the DDE path as part
  of M4's redesign (`lyapax.dde.lyapunov_spectrum_dde`); `lyapax.core.lyapunov_spectrum`
  (M1) still used dense `jax.jacfwd` per raw step, which computes all `d`
  Jacobian columns regardless of how many (`k`) are actually tracked — a
  bottleneck specifically for large non-delayed networks run with `k < d`
  (the whole point of the partial-spectrum feature from M1/M2). Fixed by
  giving `core.py`'s `_advance` the same `jax.jvp`-per-column,
  `jax.vmap`-batched pattern `dde.py` already used and validated: cost is
  now O(k) forward-mode passes per raw step, not O(d).
  - **Behavior-preserving, not a new algorithm:** `jax.jacfwd` is itself
    implemented as `vmap(jvp)` over the `d` standard basis columns, so this
    change only skips computing the `d-k` untracked columns jacfwd would
    otherwise waste time on — the math per tracked column is identical.
    Confirmed both ways: all 27 pre-existing tests pass unchanged (no
    tolerance changes needed anywhere), and a new mechanism check,
    `test_tangent_propagation_matches_dense_jacfwd` (mirroring the
    identically-named DDE test from M4), verifies the vmapped-jvp tangent
    action equals an ad hoc dense-`jacfwd` reference to `1e-10` on a small
    3D linear system — independent of any downstream QR/statistical
    convergence.
  - **Measured speedup:** a 200-node Kuramoto network (`d=200`), tracking
    `k=5` exponents, 200 raw steps: dense `jacfwd` `5.41s` vs. jvp/vmap
    `0.24s` — **~23x**, both JIT-compiled and timed after a warmup call (ad
    hoc benchmark script, not part of the test suite). `k` this small
    relative to `d` is exactly the partial-spectrum regime M1/M2 built for
    but M1's dense engine couldn't exploit until now.
  - New scale-benchmark regression test, `test_large_network_benchmark_scale`
    in `tests/test_network.py` (mirrors `test_dde.py`'s
    `test_delayed_network_benchmark_scale` from M4): the same 200-node
    network, `k=5`, `n_steps=2_000` — finite output, well under the 30s CI
    ceiling.
  - Total suite: 29/29 passing (27 prior + the 2 new tests above).
- **`vmap` over parameter grids / initial conditions for LE-vs-parameter
  sweeps — ✅ done (this session).** Mirrors `vbi`'s `JaxSweeper` pattern
  (batch a simulation over a parameter grid) without reusing its code (per
  the vendoring decision) — and needed almost no new code, because
  `lyapax.simulator.make_step_fn`'s carry already threads `params` through
  as data rather than closing over it (see that function's docstring —
  anticipated from M0/M4 specifically for this). The only piece that still
  closed over `params` at construction time was the thin
  `lyapax.network.make_network_step_fn` adapter; its new sibling
  `make_parametrized_network_step_fn` takes `params` as a call-time
  argument instead (`(state, params) -> new_state`), and `params` alone is
  swept with `jax.vmap`.
  - `src/lyapax/sweep.py` (new): `sweep_lyapunov_spectrum(step_fn, state0,
    params_batch, dt, n_steps, k=None, renorm_every=1, t_transient=0.0,
    seed=0, state0_batch=None) -> LyapunovResult` (every field gets an
    extra leading batch axis). Implementation is `jax.vmap` wrapped around
    a single, unmodified call to `lyapax.core.lyapunov_spectrum` — no new
    tangent-propagation or QR code, since batching is orthogonal to what
    the engine actually computes. `state0_batch` covers the
    "or initial conditions" half of this bullet — vmaps `(params, state0)`
    together when given, instead of a fixed `state0` shared across the grid.
  - **Definition of done — met:** `tests/test_sweep.py`, 3/3 passing (32/32
    total across the whole suite): a `G`-sweep matches a Python-loop
    reference over `lyapunov_spectrum` to `1e-8` (in practice: bit-for-bit
    identical, `0.0` max diff observed — same computation, just batched,
    not an independent numerical method); the exact `G=0`
    zero-spectrum identity holds across every element of a batch, not just
    one Python-loop call; and an initial-condition sweep
    (`state0_batch`) on the Tier 3.1 linear network matches the exact
    eigenvalues for every row.
  - **Measured speedup:** the `plot_05`-style 13-point Kuramoto `G` sweep,
    ~2.9x faster as one vmapped call (`1.38s`) vs. the Python loop
    (`4.05s`) — smaller than the ~23x matrix-free win above (this is
    dispatch-overhead amortization across grid points on CPU, not an
    algorithmic complexity change; expect a bigger win on GPU, where the
    grid points can execute with genuine hardware parallelism instead of
    just avoiding repeated Python-level dispatch).
  - New demo: `examples/plot_11_vmap_parameter_sweep.py` — reproduces
    `plot_05`'s figure via the vmap sweep, printing the max-diff against a
    fresh Python-loop run of the same system for a direct correctness
    check alongside the speed comparison.
- **GPU smoke test — ✅ done (this session).** M0's blocker
  (`INTERNAL: RET_CHECK ... dnn_support != nullptr` on any real op despite
  `jax.devices()` listing a `CudaDevice`) is resolved on this dev machine —
  confirmed by hand first (`jnp.dot` on a small array actually executes on
  `cuda:0`, not just device-enumerates), then via the new
  `tests/test_gpu.py`. **Skipped by default**, opt in with
  `JAX_PLATFORMS=cuda pytest tests/test_gpu.py`: `tests/conftest.py` sets
  `JAX_PLATFORMS=cpu` via `os.environ.setdefault`, which only applies if
  the variable isn't already set, so exporting it before pytest starts
  overrides the CPU default without touching `conftest.py` — the rest of
  the suite stays CPU-only and unaffected (`32 passed, 2 skipped` when run
  without the override, `2 passed` when run with it).
  - `test_lorenz_lambda1_matches_published_value_on_gpu`: the exact same
    system/tolerances as `test_lyapunov_core.py`'s CPU version, run on
    GPU — same published-value agreement (`λ1` within `0.08` of `0.9056`),
    device asserted to actually be `cuda:0` (not silently falling back to
    CPU).
  - `test_large_network_matrix_free_path_runs_on_gpu`: the 200-node
    Kuramoto network from `test_network.py`'s M6 matrix-free benchmark,
    run on GPU — finite `k=5` output, `0.434s` warm (vs. the `0.24s`
    CPU dense-vs-jvp comparison number logged above; not a controlled
    CPU-vs-GPU comparison, just confirms the jvp/vmap path itself works on
    GPU, which was the actual scope of this bullet — no correctness change
    expected or found).
- **Packaging (README polish, example notebooks) — postponed, not required
  for v1.** Decision this session: `examples/*.py` (sphinx-gallery format)
  already cover every capability with runnable, self-contained demos; a
  polished README and notebook conversion are documentation tasks, not
  functionality, and can wait until the API is otherwise stable.
- `notes/benchmark_report.md` (cross-tool validation against
  jitcode/jitcdde/ChaosTools.jl) — also postponed per this session's
  decision; already marked "draft skeleton, paused" in that file with its
  own "Open TODOs" resume list.

## M7 — Stretch goals (not required for v1)

- ~~Adaptive-step ODE integration (diffrax) + Benettin, as an alternative
  to fixed-step for stiff/multi-timescale ODE systems.~~ — promoted to a
  full milestone, see **M9**.
- State-dependent or distributed delays.
- ~~Sigmoidal/kuramoto coupling kernels in the DDE tangent path~~ — **done**
  as a side effect of M4's redesign: `coupling_fn` in
  `lyapax.simulator.make_step_fn` is the same plain-callable abstraction M3
  uses, so `lyapax.coupling.sigmoidal_coupling`/`kuramoto_coupling` already
  work against delayed networks with no extra code (exercised by
  `test_delayed_network_benchmark_scale`, a delayed Kuramoto network).

## M8 — Package review follow-ups (deferred from notes/package_review.md, 2026-07-02)

An external structured review (`notes/package_review.md`) covered 9
dimensions; most actionable findings were fixed directly in the same
session (stale `__init__.py` docstring/exports, README rewrite, x64
runtime warning, `renorm_every` validation, `lyapax.dde.tau_eff` +
rounding warning, `history` sort-order docs, `check_finite` opt-in on both
engines, GitHub Actions CI, ruff config + cleanup, and a new Rössler
divergence-identity test). The remaining items are lower-priority or
genuine design decisions, tracked here instead of done inline:

- [ ] `exec()` in `build_jax_dfun` (`simulator/step.py`) has no docstring
  note that `dfun_str` is trusted code — expressions are spliced directly
  into generated source with no sanitization, so a spec built from
  untrusted input is a real code-injection path, not just a style nit.
  Add a docstring warning; consider stricter allowed-name validation or
  an expression parser if lyapax ever accepts model specs from untrusted
  sources.
- [ ] dt-convergence regression tests: only one DDE test compares two
  `dt` values; the Lorenz/Rössler chaotic-flow tests each use a single
  fixed `dt`. Add a convergence-vs-`dt` test for at least one ODE
  (Lorenz or Rössler) and one more DDE case, per the review's Section 2
  "Improvement".
- [ ] Diffrax adapter for adaptive/stiff ODE integration — was tracked as
  an M7 stretch goal; promoted to a full milestone (not yet implemented),
  see **M9** below.
- [ ] `pmap`/`shard_map` examples for multi-device parameter sweeps —
  M6's `sweep.py` is single-device `jax.vmap` only; document that
  explicitly and add a `jit(vmap(...))` example before considering
  multi-device.
- [ ] Type `Protocol`s for `StepFn`/`CarryStepFn`/`CouplingFn` and
  parameter pytrees, replacing the current broad `Callable`/`dict`
  annotations — mostly a discoverability/IDE-support improvement, not a
  correctness gap.
- [ ] A convergence-helper utility summarizing last-window drift in
  `LyapunovResult.history` (relative/absolute change over the tail of
  the run) — nice-to-have, no test currently depends on it.
- [ ] Vectorized per-coupling-variable delayed gather in
  `_read_delayed_coupling` (currently a small static Python loop) — only
  worth it if multi-cvar delayed systems become common; current loop
  count is static and small.
- [ ] Dependency pinning / a tested-version lockfile — deferred while
  still at `0.0.1`; broad `jax>=0.10`/`numpy>=1.23` bounds are fine for
  now.
- [ ] Move milestone-history comments (`-- see notes/milestones.md, M4`
  etc.) out of source into docs, keeping code comments focused on
  current invariants — style-only, low priority.
- [ ] Optional GPU CI job / periodic manual validation record — CI added
  in this session is CPU-only per the review's own recommendation;
  `tests/test_gpu.py` stays opt-in and manually run for now.

## M9 — Adaptive-step integration (ODE via diffrax, DDE hand-rolled) — not started

Branch: `adaptive`. Design basis: `notes/stepping_accuracy_review.md` Part A
(the original sketch) and `notes/open_issues.md` item 4 (2026-07-07 decision
recording *why* the split is forced, not a preference) and item 5 (why
combined adaptive-step DDE stays out of scope). This milestone turns those
decisions into an implementation plan.

### Why ODE and DDE split here, restated briefly

`diffrax` has no DDE support (`patrick-kidger/diffrax#406`, open since April
2024, re-checked 2026-07-07 — still unresolved). So:

- **ODE: depend on `diffrax`.** Hand-rolling an adaptive, differentiable
  integrator means re-deriving embedded Butcher tableaus, error estimators,
  PID step-size control, and adjoint-safe gradients through accept/reject
  branching — solved, battle-tested work; not worth competing with.
- **DDE: no library option exists.** If adaptive-step DDE is ever built, it
  has to be ours. This milestone scopes *whether* to build it at all (see
  M9.3) separately from the ODE work, which starts now.

### M9.1 — Dependency and scope decision

- [ ] Add `diffrax` as an optional dependency (`pyproject.toml`'s
  `[project.optional-dependencies]`, a new `adaptive` extra — not a core
  dependency, so `lyapax`'s fixed-step path keeps zero new required deps).
  Pin a floor version compatible with the `jax>=0.10` floor already in use;
  verify against the dedicated `lyapax` dev environment
  (`/home/ziaee/envs/lyapax`, not `vbienv`) before pinning a number.
- [ ] Confirm `diffrax`'s minimum supported JAX version doesn't force
  bumping `lyapax`'s own `jax>=0.10` floor; record whatever version was
  actually tested against in the dependencies comment, matching the
  existing "Verified against jax==0.10.2" convention in `pyproject.toml`.
- [ ] Decide the public entry point shape: reuse `ode_problem(rhs, state0,
  dt, integrator=...)` with a new `integrator` value (e.g.
  `integrator="dopri5"` or an `AdaptiveODE(solver=..., rtol=..., atol=...)`
  object), vs. a separate `adaptive_ode_problem(...)` constructor. Leaning
  towards reusing `ode_problem` per `notes/stepping_accuracy_review.md`
  Part A's sketch — same downstream `lyapunov_spectrum(problem,
  n_steps=...)` call shape either way, so this is purely an ergonomics
  choice, not an architectural one.

### M9.2 — ODE: diffrax-backed adaptive stepping

Three structural changes are required on `lyapax`'s side of the boundary
(diffrax itself needs no changes) — all identified in
`notes/open_issues.md` item 4, expanded here into concrete edits:

1. **Renorm cadence: step-count → elapsed-time.**
   `core.py`'s `_run_renorm_scan` currently drives a fixed-length
   `jax.lax.scan` of `renorm_every` *raw steps* per QR block — correct
   because fixed-step `step_fn` always advances by exactly `dt`. Adaptive
   stepping doesn't take a fixed number of internal steps per unit of
   sampled time, so the per-block advance becomes "run accepted diffrax
   steps until elapsed time reaches the next renorm boundary
   (`renorm_every * dt`)" — a bounded `jax.lax.while_loop` nested inside
   (or replacing one iteration of) the outer `lax.scan`, not a second fixed
   -length scan.
   - [ ] New `_advance_adaptive_block(carry, target_dt)` (or equivalent) in
     `core.py`, gated behind the adaptive integrator, that wraps a
     `while_loop` over `solver.step(...)` calls (see M9.2.2) until
     accumulated `t` reaches `target_dt`, clamping the final internal step
     so it lands exactly on the boundary (diffrax solvers support a
     `t1` clamp per step; confirm the exact API when implementing).
   - [ ] Existing fixed-step path (`_run_renorm_scan` as used by
     `ode_problem`/`network_problem`/`dde_problem`) must be byte-for-byte
     unchanged — this is purely a new code path selected by the
     `integrator` value, not a rewrite of the existing one.
2. **Tangent propagation rides diffrax's per-step primitive, not
   `diffeqsolve`.** Today's tangent step is `jax.jvp(step_fn, (state,),
   (y_col,))` per tracked column, `jax.vmap`-batched over `k` (M6's
   matrix-free win, the partial-spectrum cost advantage this whole engine
   is built around). `diffeqsolve` runs its own internal loop end-to-end
   and does not compose with that per-step jvp trick — it would force
   re-running the *whole* adaptive trajectory once per tracked tangent
   column, discarding the O(k) cost model entirely.
   - [ ] Use diffrax's lower-level stepping API (`solver.step(terms, t0,
     t1, y0, args, solver_state, made_jump)` — the primitive
     `diffeqsolve` itself is built on) as the atomic unit inside `lyapax`'s
     own `while_loop`, so `jax.jvp` still wraps one bounded call at a
     time, exactly like the fixed-step case.
   - [ ] Confirm empirically (not just by reading diffrax's source) that
     `jax.jvp` through `solver.step` produces a finite, correct tangent —
     add a unit test isolating just this (a linear system with a known
     tangent action, comparing `jvp`-through-`solver.step` against the
     analytic fundamental-matrix action), *before* wiring it into the full
     Benettin loop, so a diffrax-internals surprise is caught in isolation.
3. **`stop_gradient` on the step-size decision.** Gradient must flow
   through the accepted state update but not through the controller's
   accept/reject/step-size choice (a discrete, non-differentiable
   decision) — `diffeqsolve`'s own adjoint handles this internally, but
   using the lower-level `solver.step` API directly means `lyapax` owns
   getting the placement right.
   - [ ] Identify exactly which quantity from diffrax's step-size
     controller (likely the proposed next `dt`/`h`) needs
     `jax.lax.stop_gradient`, and confirm via a differentiation test that
     `jax.grad`/`jax.jacrev` of a downstream scalar (e.g. `lambda_max`,
     tying this directly to item 6's differentiate-through-the-exponent
     goal in `notes/open_issues.md`) doesn't silently produce all-zero or
     NaN gradients across an accept/reject boundary.

### M9.3 — DDE: adaptive stepping is *not* being built in this milestone

Per `notes/open_issues.md` item 5: combined adaptive-step DDE needs history
interpolation over an *irregular* grid (an adaptive integrator's own
accepted-step times aren't evenly spaced, unlike the uniform ring buffer
`interpolate=True` Hermite scheme from `notes/stepping_accuracy_review.md`
Part B), which is substantially harder than either the ODE work above or
the existing fixed-step DDE interpolation alone.

- [ ] **Decision recorded, not deferred by omission:** DDE gets fixed-step
  only in this milestone. Fixed-step + `interpolate=True` + `rk4` already
  reaches the Hermite-interpolant accuracy ceiling (`notes/open_issues.md`
  item 1's resolution, ~order 4 until the interpolant itself caps it) —
  there is no known concrete need (stiff/multi-timescale delayed system)
  currently motivating the extra complexity.
- [ ] If a concrete need appears later, the shape of the work (for
  scoping a future milestone, not to be started now): replace the uniform
  -stride ring buffer (`lyapax/dde.py`'s `constant_history_buf0` /
  `_resolve_horizon` / the `tau_steps`-indexed read in
  `lyapax/simulator/step.py`) with a variable-stride history store keyed
  by actual accepted-step times, and generalize the cubic Hermite read
  (currently `theta = frac((t - tau) / dt)` against two *grid* points) to
  interpolate between two *arbitrary* stored times — a new data structure,
  not a parameter change to the existing one.
- [ ] No code changes to `lyapax/dde.py` in this milestone; if M9.2's
  `ode_problem` gains a new `integrator` value, confirm `dde_problem` /
  `network_dde_problem` explicitly reject (or simply never accept) an
  adaptive `integrator`, so a caller can't silently request something
  unimplemented — a clear `ValueError`, not silent fixed-step fallback.

### M9.4 — API surface (target shape, confirm against M9.1's decision)

```python
import diffrax
integrator = diffrax.Dopri5()  # or a small lyapax wrapper picking rtol/atol defaults
problem = lyapax.ode_problem(rhs, state0, dt=0.1, integrator=integrator)
result = lyapax.lyapunov_spectrum(problem, n_steps=...)
```

No visible change to `lyapunov_spectrum`'s signature or `LyapunovResult`.
The one internal casualty: `ODEProblem.step_fn`'s current single-function
contract (`state -> new_state`) doesn't carry enough state for adaptive
stepping (needs a threaded diffrax `solver_state` and the current `t`) —
this becomes an internal carry shape difference selected by `integrator`,
not a user-facing API change.

### M9.5 — Validation plan

- [ ] Convergence check: for a system with a known exact Lyapunov exponent
  (e.g. the existing Lorenz `λ1 ≈ 0.9056` reference, or a linear system
  with exact eigenvalues), sweep `rtol`/`atol` tighter and confirm the
  estimate converges monotonically toward the reference, the same
  dt-sweep-hygiene pattern already used for fixed-step (`notes/
  validation_systems.md`).
- [ ] Cross-check: a very tight `rtol`/`atol` should reproduce the existing
  fixed-step `rk4`/`rk6` answer on the same system to within both methods'
  combined error bars — confirms the new path isn't silently solving a
  different problem, not just that it converges to *something*.
- [ ] Differentiability check per M9.2.3 — `jax.grad` of an exponent
  w.r.t. a `rhs` parameter is finite and matches a finite-difference
  reference away from any accept/reject boundary.
- [ ] Document the accept/reject non-smoothness caveat from `notes/
  stepping_accuracy_review.md` Part A explicitly in the new integrator's
  docstring — worth knowing, not worth blocking on.
- [ ] New demo, `examples/plot_14_adaptive_ode.py` (next free number after
  `plot_13_dde_history_interpolation.py`) — a new capability needs a
  runnable `examples/` demo, not just unit tests; self-contained, with no
  `notes/`/milestone-number references inside the example file itself.
- [ ] New regression tests: `tests/test_adaptive_ode.py` — convergence,
  cross-check, and differentiability tests above, plus a test that
  `dde_problem`/`network_dde_problem` reject an adaptive `integrator`
  (M9.3's last bullet).

### Definition of done for M9

- `diffrax`-backed adaptive ODE integration available through
  `lyapax.ode_problem`/`lyapax.lyapunov_spectrum` with no change to
  existing fixed-step call sites or results.
- Validation plan (M9.5) fully passing, including the differentiability
  check tying back to `notes/open_issues.md` item 6.
- DDE stays fixed-step only, with an explicit, tested rejection of
  adaptive `integrator` values rather than silent fallback.
- `notes/open_issues.md` item 4 marked resolved/implemented (currently
  "decision recorded... not started work"); item 5 stays open, now
  pointing at M9.3's scoping notes instead of being a bare "out of scope"
  line.

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
- [x] GPU: was unusable on the dev machine (cudnn/driver mismatch, see M0);
      confirmed fixed in M6 (environment change, independent of this
      codebase) — `tests/test_gpu.py` runs real Lyapunov computations on
      `cuda:0` with correct output, opt-in via `JAX_PLATFORMS=cuda`.
- [x] Whether M1's Jacobian should default to `jacfwd` (dense, simple) or
      `jvp`-per-column from the start — went dense for M1/M3 as recommended
      here, and that was the right call: M4 needed matrix-free `jvp`
      propagation for real (large augmented `(state,buf)` dimension from
      the ring buffer), built it there once there was a concrete need, and
      it worked (`test_delayed_network_benchmark_scale`, `d_total>300`).
      Still open for M1/M3's plain flat-state case specifically (dense
      `jacfwd`, no concrete large-non-delayed-network need yet) — see M6.
