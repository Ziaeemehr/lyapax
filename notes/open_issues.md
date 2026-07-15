# Open Issues

Not-yet-resolved items, as of the stepping-accuracy work in
`notes/stepping_accuracy_review.md`. For older, unrelated follow-ups
(package-review deferrals: `exec()` trust docs, dt-convergence test
coverage, `Protocol` types, dependency pinning, etc.), see
`notes/milestones.md`'s M8 checklist instead -- this file only tracks the
items below.

## 1. DDE Lyapunov exponents converge at only ~O(dt) -- RESOLVED

**Root cause found: two independent O(dt) errors were stacked, and every
experiment removed only one of them at a time.**

The two errors:

1. **Delayed history frozen across RK stages** (this file's item 3 /
   the review note's Part C): each step read the delayed value once, at
   the step's start time, and held it fixed across all internal stages.
   That's an O(dt) error in the delayed argument of the RHS -- global
   order 1 regardless of integrator, exactly like frozen zero-delay
   coupling was.
2. **The ring-buffer write off-by-one** (item 2 below): the state at
   physical time `(t+1)*dt` was stored under slot `t`, silently shifting
   the effective delay by one full step -- an O(dt) bias in *which system
   was being solved*.

Why it looked unexplainable: the per-stage-read fix for (1) was attempted
*while (2) was still present* (commit `bc9a089` has the attempt's context
and still writes under slot `t`), so (2)'s O(dt) bias masked any gain and
the attempt was reverted as "ineffective". Later, (2) was fixed and the
convergence order re-measured -- but only with the *shipped* frozen-read
code, where (1) still capped it at order 1. Each fix was individually
"ruled out" by a measurement in which the other bug was binding.

The fix (both together): with the corrected write convention already in
place, `interpolate=True` now re-reads the delayed history at each
integrator stage's own intra-step time (`step + c`, `c` the Butcher node)
via the cubic Hermite interpolant, instead of once per step. See
`lyapax/simulator/step.py` (`_coupling_at`).

Measured after the fix (scalar linear DDE, `x' = -a x(t-tau)`, `a=0.5`,
`tau=0.317`, vs the Lambert-W analytic exponent):

- `heun`: error 9.2e-6 -> 1.4e-7 across `dt` = 2e-2 -> 2.5e-3, order
  2.00, 2.00, 2.00 (was ~1e-3-class errors at order 1.00 before).
- `rk4`: 2.1e-11 at `dt=2e-2`, order ~4.4 until the estimation noise
  floor (~5e-14).
- `rk6`: capped at ~order 4 -- the cubic Hermite interpolant's own
  accuracy ceiling, the expected theoretical limit. `rk4` is now the
  sweet spot for DDEs; `rk6`'s extra stages buy nothing past the
  interpolant.

Two caveats, both inherent rather than bugs, verified by re-running the
same sweep with a smooth exact-solution history (`x(t)=exp(lambda*t)`,
which has no breaking points -- rk4 then measures ~4.4 down to float64
roundoff on the *trajectory*, not just the exponent):

- **The grid-snapped default (`interpolate=False`) remains O(dt) by
  construction** -- without stored derivatives there is no sub-step
  history to read, and its `tau` is rounded to the grid besides. Pass
  `interpolate=True` to get integrator-order convergence.
- **Constant-history trajectories cap at ~order 2 near t=0.** The
  constant history has slope 0 but the true solution's right-derivative
  at `t=0` doesn't, and the slot-0 stored derivative can only represent
  one of the two one-sided slopes -- a one-time O(dt^2) error injection
  (the classic DDE breaking-point issue; a full fix needs breaking-point
  handling, out of scope). Lyapunov exponents are unaffected in practice:
  the mandatory transient discards the breaking-point region, which is
  why `rk4` measures ~4.4 on the exponent itself.

Regression tests: `tests/test_dde_interpolation.py`
(`test_interpolate_heun_converges_at_second_order`,
`test_interpolate_rk4_reaches_hermite_accuracy_floor`). Demo:
`examples/plot_13_dde_history_interpolation.py` (now also plots the rk4
curve). A side effect worth knowing: `interpolate=True` now requires
`tau >= dt` (`tau_steps >= 1`, validated with a clear error) -- the
per-stage reads at intra-step times up to `step + 1` must only touch
ring-buffer slots already finalized, which fails for sub-step delays
(that configuration was silently unsound before, too: the stored-
derivative computation already read at `t + 1`).

### Practical impact on package usage (updated)

- **Integrator choice matters for DDEs again** with `interpolate=True`:
  `heun` gives order 2, `rk4` order ~4; `rk6` is not worth its extra
  stages (Hermite ceiling ~4). With `interpolate=False`, everything still
  caps at O(dt) -- prefer `interpolate=True` whenever accuracy matters.
- **`dt` sweeps remain good hygiene** (2-3 halvings to confirm the
  exponent has stabilized), but with `interpolate=True` + `rk4` each
  halving now buys ~16x, not 2x.
- **Ring-buffer cost still scales as `~tau/dt`** -- but since accuracy no
  longer requires brute-force `dt` shrinking, the pressure on memory/step
  count is much lower for the same target error.

## 2. Ring-buffer read/write off-by-one -- confirmed and fixed

`_write_ring`/`_write_ring_interp` (`lyapax/simulator/step.py`) wrote the
newly integrated state -- the value at physical time `(t + 1) * dt` --
into ring-buffer slot `t`, not `t + 1`. Confirmed directly in code, not
just inferred from a synthetic test: the `interpolate=True` branch already
computed the stored derivative by reading coupling *at time `t + 1`*
(`_coupling_at(new_state, 0.0, buf, t + 1, params)`), then stored the
result under slot `t` -- an internal inconsistency between the time a
value was computed for and the slot index used to label it.

Fixed by writing under `t + 1` in both `_write_ring` and
`_write_ring_interp`. Verified two ways:
- `tests/test_ring_buffer_time_convention.py`: a deterministic,
  Lyapunov-free check (per the plan below) that `read(step=t,
  tau_steps=m) == value_at_time((t - m) * dt)` for every reachable `(t,
  m)`, for both the plain and Hermite-interpolated read/write pairs.
- `tests/test_delayed_networks.py::test_per_edge_delay_near_zero_recovers_m3_eigenvalues`
  actually caught the bug independently: at `tau_steps=1`, the *old* code
  matched the zero-delay eigenvalues almost exactly (the off-by-one was
  silently zeroing out a "1 step" delay). After the fix, the same
  `tau_steps=1` case shows a real, nonzero deviation from the zero-delay
  eigenvalues that scales linearly with `dt` (~4.5e-3 at `dt=1e-3`,
  ~4.5e-4 at `dt=1e-4`) -- exactly the `O(dt)` bias a genuine one-step
  delay should produce. The test's `dt` was tightened (1e-4, from 1e-3)
  to keep the delay->0 limit it's named for meaningfully small, rather
  than loosening its tolerance to admit a real nonzero delay.

**Half of item 1, in hindsight.** The `interpolate=True` DDE convergence
order was measured before and after this fix and was unchanged (~1.0 both
times), so this bug was initially "ruled out" as item 1's cause -- but
both of those measurements used the frozen per-step history read, which
was *also* O(dt)-binding on its own. Item 1 was these two errors stacked;
see its resolution above. Independently of that, this was a real
correctness bug (it silently shifted every DDE/per-edge-delay delay one
step short).

Full writeup: `notes/stepping_accuracy_review.md`, Part C;
`notes/possible_solution_to_open_issues.md` for the original investigation
plan this followed.

## 3. Coupling frozen across RK stages -- fixed for zero-delay and DDE

Every integrator used to read coupling once per step and hold it fixed
across that step's internal RK stages, rather than recomputing it fresh
at each stage the way a correct RK implementation of the full right-hand
side would. This capped accuracy at `O(dt)` for any genuinely coupled
system, delayed or not, regardless of the base method's nominal order.

- **Zero-delay networks: fixed and validated.** `coupling_at(y_stage, c)`
  is now called fresh at each stage. A coupled linear network's error
  dropped ~5 orders of magnitude at the same `dt`, and `rk4`/`rk6` no
  longer give identical errors.
- **DDE (`interpolate=True`): fixed.** The delayed history lookup is
  recomputed at each stage's own intra-step time via the Hermite
  interpolant. The first attempt at this looked ineffective only because
  the ring-buffer off-by-one (item 2) was still present at the time --
  see item 1's resolution above.
- **DDE (`interpolate=False`): O(dt) by construction, left as is.**
  Without stored derivatives there is no sub-step history to read.

Full writeup: `notes/stepping_accuracy_review.md`, Part C.

## 4. Adaptive ODE integration -- RESOLVED (implemented via diffrax, M9)

**Implemented 2026-07-08, `lyapax.adaptive.diffrax_adaptive_step`** (see
`notes/milestones.md` M9 for the full writeup). Summary of what changed
relative to the plan below: no `core.py` changes were needed at all (the
adaptive `step_fn` satisfies the exact same `state -> new_state` contract
`_advance` already assumes); a diffrax solver must be constructed with
`scan_kind="lax"` or `jax.jvp` through it raises `TypeError` (a diffrax-
internals surprise, found via an isolated test before wiring anything up);
diffrax's `PIDController` already `stop_gradient`s its own step-size
decisions internally, so `lyapax` needed no extra `stop_gradient` calls;
and differentiating a Lyapunov exponent through this integrator works with
`jax.jacfwd` (forward-mode) but **not** `jax.grad`/`jax.jacrev` (reverse-
mode can't replay a dynamic-trip-count `lax.while_loop` backward at all --
a more fundamental limitation than the "non-smooth at accept/reject
boundaries" caveat anticipated below). DDE stays fixed-step only, per
item 5, with an explicit rejection if an adaptive integrator is passed to
`dde_problem`/`network_dde_problem`.

<details>
<summary>Original planning note (2026-07-07, pre-implementation)</summary>

Not started; decided to depend on diffrax, not hand-roll.

A design exists (`notes/stepping_accuracy_review.md`, Part A: an
`integrator` value backed by `jax.lax.while_loop`, `dt` kept as the
outer sampling interval rather than the integrator's own step size), but
no implementation has been written. That note left open "hand-rolled
embedded pair vs. depending on `diffrax`" -- discussion below (2026-07-07)
resolves that question in favor of depending on `diffrax`, and records
what the integration work actually involves.

### Decision: depend on diffrax (as a library), don't hand-roll

Re-checked 2026-07-07: `diffrax` still has no DDE support (open feature
request `patrick-kidger/diffrax#406`, unresolved since April 2024) --
confirms the M0-era finding still holds. So the ODE/DDE split is forced,
not a preference:

- **ODE: use diffrax.** Hand-rolling an adaptive, *differentiable*
  integrator means re-deriving embedded RK Butcher tableaus, error
  estimators, PID step-size control, and -- the part that matters most --
  adjoint-safe gradients through adaptive stepping (this note's own
  flagged risk: "non-smoothness at accept/reject boundaries when
  differentiating w.r.t. a system parameter"). That's deep, error-prone
  numerical work diffrax has already solved and battle-tested; the
  strategic move is to depend on it, not compete with it. This also
  reinforces item 6's "differentiate through the exponent itself"
  priority, rather than working against it.
- **DDE: still ours.** No JAX library covers this. Fixed-step +
  `interpolate=True` + rk4 already reaches the Hermite-interpolant
  accuracy ceiling (item 1's resolution) -- keep full adaptive-step DDE
  gated on a concrete need (stiff/multi-timescale delayed systems) per
  item 5, rather than building it preemptively just because ODE gets
  adaptivity.

### What "depend on diffrax" actually requires (still real, but bounded)

The integration isn't a drop-in swap, because lyapax's Benettin engine
isn't shaped like diffrax's typical usage (`diffeqsolve` over a whole
interval). Three structural changes, all on lyapax's side of the
boundary:

1. **Renorm cadence: step-count -> elapsed-time.** `core.py`'s
   `_advance` runs a fixed-length `lax.scan` of `renorm_every` raw steps
   between QR renormalizations. Adaptive stepping doesn't take a fixed
   number of steps per unit time, so this becomes "run accepted steps
   until elapsed time reaches the next renorm boundary" -- a bounded
   `lax.while_loop`, not a fixed-length `scan`.
2. **Tangent propagation rides diffrax's per-step primitive, not
   `diffeqsolve`.** Currently `jax.jvp(step_fn, (state,), (y_col,))` per
   tracked tangent column, vmapped over k (the k << d partial-spectrum
   cost advantage). `diffeqsolve` runs its own internal loop end-to-end
   and doesn't compose with that per-step jvp trick. Use diffrax's
   lower-level `solver.step()` (the primitive `diffeqsolve` itself is
   built on) as the atomic unit inside our own loop instead.
3. **`stop_gradient` on the step-size decision.** Gradient must flow
   through the accepted RK update but not through the controller's
   accept/reject/dt choice. `diffeqsolve`'s adjoint handles this
   internally; using the lower-level stepping API means lyapax owns
   getting that `stop_gradient` placement right.

### API impact: none visible to callers

`ode_problem(rhs, state0, dt, integrator="dopri5")` (or a separate
`adaptive_ode_problem(...)`) feeding the same
`lyapunov_spectrum(problem, n_steps=...)` call shape. The one internal
casualty is `ODEProblem.step_fn`'s single-function contract (`state ->
new_state`) -- adaptive stepping needs an extra threaded carry (solver
state, current `t`), so that becomes an internal detail, not a
user-facing function.

Not yet turned into an implementation plan or milestone entry -- this is
a design decision recorded for later, not started work.

</details>

## 5. Combined adaptive-step DDE -- correctly out of scope for now

Would need history interpolation over an *irregular* grid (since an
adaptive integrator's own sample times aren't evenly spaced), which is
substantially harder than either item 3/4 alone. Not worth attempting
until items 1 and 4 are each resolved independently.

## 6. Feature roadmap: next-level capabilities (added 2026-07-07)

Not a bug-tracking item like 1-5 above -- a strategy note on where to grow
next, prompted by the question of whether lyapax should chase feature
parity with ChaosTools.jl/DynamicalSystems.jl.

### Recommendation: stay narrow, don't chase ChaosTools.jl's breadth

ChaosTools.jl is one piece of a decade-old, many-contributor ecosystem
(DynamicalSystems.jl: DelayEmbeddings, RecurrenceAnalysis,
ComplexityMeasures.jl, Attractors.jl, ...) covering recurrence
quantification, fractal/correlation dimensions, entropy/complexity
measures, periodic-orbit detection, basin-of-attraction mapping, delay
embeddings, surrogate testing. None of that shares lyapax's
tangent-propagation core (`core.py`'s Benettin/QR scan) -- adding it
would mean bolting a second, weaker product onto the Lyapunov engine
rather than extending it, and users who want that breadth already have
Julia. lyapax's real advantage is JAX itself -- `jit`/`vmap`/GPU, and
above all differentiability through the exponent computation, which
ChaosTools.jl cannot offer (no reverse-mode AD through their solver
stack). "Next level" should mean depth along that axis, not breadth
toward a Julia clone.

### Fits the existing engine -- worth prioritizing

1. **Differentiate through the Lyapunov exponent itself -- audited
   2026-07-15, works with a real caveat.** `lyapunov_spectrum` is built
   from `jax.lax.scan` (static trip count) + `jnp.linalg.qr`, both
   reverse-mode-compatible, so `jax.grad`/`jax.jacrev` do not raise here
   (unlike `lyapax.adaptive`'s diffrax integrator, item 4). Confirmed on a
   non-chaotic linear system: `jax.grad`, `jax.jacfwd`, and a central
   finite-difference reference all agree to the analytic answer
   (`d(lambda_max)/d(gamma) == 1` exactly) to `<1e-6`. But on a genuinely
   chaotic system (Lorenz), the same call returns a large, finite number
   that is *not* the useful sensitivity -- `step_fn` closes over the
   differentiated parameter, so autodiff walks through the entire unrolled
   chaotic trajectory and the "gradient" inherits its exponential
   sensitivity to perturbation, empirically scaling like
   `exp(lambda_max * horizon)` (observed: already `~1e2`-`1e4` within a
   few hundred steps, `>1e8` by a few thousand -- see
   `tests/test_differentiability.py`). Reverse- and forward-mode agree
   with each other even in the diverging case, confirming this is the
   real value of this particular (naive, trajectory-unrolling) estimator's
   gradient, not an AD-mode-specific bug. This matches a known
   chaotic-sensitivity-analysis result (why shadowing-based methods, e.g.
   least-squares shadowing, exist in that literature) rather than being a
   lyapax defect. **Net: shipped as a documented, tested capability with a
   scope limit** -- reliable for non-chaotic/short-horizon systems (e.g.
   gradient-based tuning of a parameter toward a target exponent while
   staying off an attractor), not for gradients through long chaotic
   trajectories without independent verification. Documented in
   `lyapax.core`'s module docstring and
   `docs/background/capabilities.md`; demoed in
   `examples/18_differentiate_lyapunov_exponent.py`. Shadowing-based
   methods for the genuinely-chaotic case are a possible future direction,
   not attempted here -- substantial additional machinery (not a
   `lyapunov_spectrum`-level change) for a need that hasn't arisen yet.
2. **Kaplan-Yorke (Lyapunov) dimension -- done, 2026-07-15.**
   `lyapax.core.kaplan_yorke_dimension(exponents, d_total=None)`: pure
   post-processing of `LyapunovResult.exponents` (walk the cumulative sum,
   interpolate the fractional part at the zero crossing), no new
   tangent-propagation or QR machinery. Found while implementing: two
   byte-for-byte identical private `_kaplan_yorke_dimension` helpers
   already existed (`tests/test_dde.py`, `benchmarks/lyapax/mackey_glass.py`)
   -- both retired in favor of the new public function.
   **Correctness addition beyond the original ad hoc helpers:** an
   optional `d_total` guard -- if the tracked spectrum is a partial one
   (`k < d_total`) and its cumulative sum never goes negative, the true
   crossing point lies beyond what's tracked, so returning `k` (the old
   helpers' silent behavior in this case) would understate the real
   dimension; passing `d_total` now raises `ValueError` instead. Tested
   in `tests/test_kaplan_yorke.py` (hand-checked crossing, all-negative,
   full-spectrum edge case, the `d_total` guard, and a cross-check against
   Lorenz's published exponents matching the literature's ~2.06). Demoed
   in `examples/19_kaplan_yorke_dimension.py` (Lorenz + Rossler, plus a
   plot of the cumulative-sum-crossing-zero mechanics the formula is
   built on).
3. **Covariant Lyapunov Vectors (CLVs).** A natural extension of the
   forward QR pass already in `core.py`'s `_advance`/`_renorm_block` --
   needs a backward pass through the stored `R` factors (Ginelli et
   al.'s algorithm). Gives Oseledets-direction/hyperbolicity information
   beyond the exponents themselves; reuses the existing scan structure
   rather than introducing a new one.
4. **Finite-time / local Lyapunov exponents.** `LyapunovResult.history`
   already carries the running per-block estimate; formalizing a
   windowed/local variant (rather than only the long-horizon average) is
   a small delta and useful for spotting intermittency or regime changes
   along a trajectory.
5. **Adaptive-step ODE integration (diffrax).** Already tracked as an
   M7/M8 stretch goal in `notes/milestones.md`; still the most-cited gap
   in `docs/background/capabilities.md` ("No adaptive or stiff ODE
   integration").
6. **State-dependent delays.** Ambitious, already an explicit non-goal
   for v1 (`notes/milestones.md`). Revisit only after item 4 above (in
   this file's numbering) and diffrax integration have landed.

### Explicitly not chasing

Recurrence quantification analysis, fractal/correlation dimension
estimators, entropy/complexity measures, delay-embedding reconstruction,
periodic-orbit search, basin-of-attraction mapping -- all mature and
well-served by ChaosTools.jl/DynamicalSystems.jl already. Weaker
reimplementations wouldn't help users who have Julia access, and would
dilute lyapax's actual identity: a JAX-native, differentiable Lyapunov
engine, not a general nonlinear-timeseries toolkit.
