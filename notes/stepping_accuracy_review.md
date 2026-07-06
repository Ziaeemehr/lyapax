# Stepping Accuracy Review: Adaptive ODE Integration and DDE History Interpolation

## Status

| Item | State |
|---|---|
| DDE Hermite-interpolated history reads (`interpolate=True`) | **Shipped** |
| Coupling recomputed per RK stage, zero-delay networks | **Shipped** |
| Adaptive ODE integration | **Not started** (design below) |
| Coupling/history recomputed per RK stage, DDE case | **Shipped** -- an earlier attempt only looked ineffective because the ring-buffer off-by-one was masking it |
| Ring-buffer read/write off-by-one | **Fixed** -- and retroactively identified as half of the DDE O(dt) mystery |

## Summary

Two accuracy limitations share one root cause: both the ODE and DDE
engines assume every quantity they need lands *exactly* on a fixed,
uniform `dt` grid.

- For ODEs: no adaptive step-size control. `lyapax.integrators`' methods
  (`euler`, `heun`, `rk4`, `rk6`) all take one fixed `dt` per call, with no
  mechanism to take smaller internal steps where the trajectory is
  fast-changing and larger ones where it's slow.
- For DDEs: no sub-step delay interpolation (now fixed, see Part B) --
  `resolve_tau_steps` used to round every physical delay `tau` to the
  nearest integer number of `dt` steps, and every delayed read pulled an
  exact stored ring-buffer sample, never a value *between* two samples.

Both were documented, known tradeoffs, not oversights: the DDE case is
"risk #4" in `notes/milestones.md`; `notes/api_design_review.md`'s
"Adaptive Integrators" section scopes the ODE case as future work. This
note works out how to resolve each, and why -- despite the shared root
cause -- they're solved separately, not as one combined "fully adaptive
DDE" feature (see the closing section).

A third, related issue surfaced while validating Part B: coupling and
delayed-history lookups were being read once per integrator step and held
fixed across that step's internal stages, silently capping every
integrator (including `rk6`) at first-order accuracy for any genuinely
coupled or delayed system. That's Part C -- fixed for zero-delay networks,
still open for DDEs.

## Part A: Adaptive ODE Integration (not started)

### Current behavior

`lyapax.ode_step(rhs, dt, integrator=...)` builds a `state -> new_state`
map that advances by exactly `dt` using a fixed number of internal stages.
`lyapunov_spectrum` doesn't know or care how `step_fn` gets from `state`
to `state` `dt` later -- it only needs `step_fn` to be a pure,
differentiable function of `state`, called once per raw step inside a
fixed-length `jax.lax.scan` (`renorm_every` raw steps per QR block). That
decoupling of outer sampling from inner stepping is what makes adaptive
integration tractable here without touching `core.py` at all.

### Proposed design

An adaptive integrator is just another `integrator` value (a callable
`(rhs, dt) -> step_fn`), except internally `step_fn(state)` runs a
`jax.lax.while_loop`: take an internal step of a proposed size `h`,
estimate its local error against an embedded lower-order companion
formula, accept or shrink-and-retry, and keep going until the accumulated
internal time reaches `dt` exactly. `dt` keeps today's meaning -- the
fixed interval between renormalization samples -- it just stops also
being "the integrator's own step size."

```python
integrator = lyapax.integrators.AdaptiveRK45(rtol=1e-6, atol=1e-9)
step = lyapax.ode_step(rhs, dt=0.1, integrator=integrator)   # dt = sampling interval
result = lyapax.lyapunov_spectrum(step, state0, dt=0.1, n_steps=...)
```

### Why this fits better than expected

The classic objection to adaptive-step + autodiff is that a
variable-trip-count loop is hard to differentiate: reverse-mode AD
through `jax.lax.while_loop` needs to replay an a priori unknown number of
steps backward, which JAX doesn't support directly. But this codebase's
tangent engine is `jax.jvp`-based, not `jacrev`/`vjp`-based (a deliberate
choice, per `core.py`'s module docstring). Forward-mode JVP through a
`while_loop` is not the hard case: the primal trajectory runs concretely
(its accept/reject decisions resolve to actual numbers during the forward
pass), and the tangent rides along the same accepted path -- no new
autodiff machinery needed. `while_loop`'s trip count is a runtime value,
not a compile-time shape, so it doesn't trigger recompilation either.

### Open questions

- **Non-smoothness at accept/reject boundaries.** A parameter change
  could shift which steps get accepted, making the exponent non-smooth in
  that parameter at a measure-zero set of points. Worth documenting, not
  worth blocking on.
- **Hand-rolled vs. dependency.** `diffrax` already solves this robustly
  (JAX-native, differentiable, dense output), but this repo's stated
  philosophy is to vendor a minimal simulator rather than import one (see
  `src/lyapax/simulator/NOTICE.md`) -- a real choice to make explicitly.
- **Validation plan.** An empirical convergence check as `rtol`/`atol`
  shrink, against a system with a known exact answer, plus a cross-check
  that a very tight `rtol`/`atol` reproduces the existing fixed-step `rk6`
  answer on the same system.

### Recommendation

A scoped follow-up: one embedded pair (e.g. Bogacki-Shampine RK23 or
Dormand-Prince RK45) behind `jax.lax.while_loop`, validated per the plan
above, before deciding whether `diffrax` is worth wiring in for harder
cases (stiffness, dense output).

## Part B: DDE History Interpolation (shipped)

### Why

`resolve_tau_steps(tau, dt)` rounds a physical delay to the nearest
integer number of `dt` steps; every downstream spectrum is computed for
that rounded `tau_eff`, not the `tau` requested. Two consequences: (1)
refining accuracy requires shrinking `dt` globally -- there's no way to
improve delay accuracy independently of integration accuracy; (2) the
rounding error is an `O(dt)` bias in *what system is actually being
simulated*, not a shrinking truncation error, so it doesn't improve at
the rate a higher-order integrator's own error does.

### What's implemented

Cubic Hermite interpolation over the ring buffer, built from the value
*and derivative* at the two grid points bracketing `t - tau`, instead of
snapping to the nearer one -- the same approach established adaptive-step
DDE solvers (`jitcdde`) use. `make_step_fn(..., interpolate=True)`,
surfaced through `dde_problem(..., interpolate=True)` /
`network_dde_problem(..., interpolate=True)`; default
`interpolate=False` preserves the original grid-snapped behavior
byte-for-byte.

- **Ring buffer stores value + derivative.** The coupling-variable
  derivative `d(cvar_state)/dt` is exactly the relevant slice of
  `dfun(state, coupling, params)`, already computed each step -- just not
  previously saved. Buffer shape: `(horizon, 2, n_cvar, n_nodes)`.
- **Read side interpolates**, using the standard cubic Hermite basis
  (`h00, h10, h01, h11`) against `theta = frac((t - tau) / dt)`. `tau` no
  longer needs to be a whole number of `dt` steps.
- **Tangent propagation needs no new machinery.** The interpolant is a
  fixed (`theta` never depends on `state`), smooth, linear function of
  four buffer entries -- `jax.jvp` differentiates through it the same way
  it already differentiates every other buffer read. The tangent buffer's
  per-slot size doubles (value + derivative), so `d_buf`/QR cost grow
  accordingly, though `horizon` itself may be able to shrink for the same
  delay since interpolation error, not grid density, is now the
  bottleneck.

### Scope limits

- **Uniform delay only.** Wired up for the uniform-`tau_steps` +
  custom-`coupling_fn` path. The legacy per-edge `delay_steps` matrix path
  would need a *different* `theta` per edge -- a real generalization, not
  a drop-in change; raises `ValueError` if requested there.
- **Still assumes constant, known-ahead-of-time `tau`.** State-dependent
  or time-varying delays are out of scope, as everywhere else in this
  codebase.
- **Doesn't touch precision-limited systems.** Fixes the `tau`-rounding
  bias only; doesn't address the float32/float64 or `renorm_every`
  overflow risks already documented in `notes/milestones.md`.

### Validated

`tests/test_dde_interpolation.py` / `examples/plot_13_dde_history_interpolation.py`:
exact-integer-`tau` reduces to the grid-snapped answer to ~1e-10 (as it
must, `theta=0` at every read); at a `tau` not a multiple of any swept
`dt`, grid-snapping's error is confirmed non-monotonic (e.g. `tau_eff`
comes out `0.320, 0.320, 0.320, 0.315, 0.3175` across five `dt` values --
worse at the finer `dt=0.005` than at `dt=0.01`), while interpolation uses
`tau` exactly at every `dt` and decreases smoothly and monotonically.
That smooth, predictable convergence is what shipped; the *rate* of
convergence is now the integrator's own order (up to the Hermite
interpolant's ~4th-order ceiling), see Part C.

## Part C: Coupling/History Frozen Across RK Stages

### The issue

Every integrator (`_euler`/`_heun`/`_rk4`/`_rk6` in
`lyapax.simulator.step`) read coupling *once* per step and held it fixed
across that step's internal stages, rather than recomputing it as the
true right-hand side would need. This is not standard RK practice --
textbook RK evaluates the complete right-hand side fresh at each stage's
own `(time, state)` argument -- and it silently caps accuracy at `O(dt)`
for *any* genuinely coupled or delayed system, regardless of the base
method's nominal order. Confirmed directly: on a coupled linear network
(`G=0.5`), `rk4` and `rk6` gave *identical* errors at every `dt` before
the fix, both converging at order ~1 -- `rk6`'s 6th-order accuracy was
never actually realized for any coupled system, only for uncoupled ones.

### Zero-delay: fixed

`_euler`/`_heun`/`_rk4` now take a `coupling_at(y_stage, c)` callable,
called fresh at every stage with that stage's own intra-step state
estimate; `rk6_combine` (`lyapax/integrators.py`) was generalized from
`f(y) -> dy` to `f(y, c) -> dy` so `_rk6` does the same, reusing its own
already-verified `c_i` values (`RK6_STAGE_C`). Validated: on the same
coupled network, at `dt=0.02`, RK4's error dropped from `3.1e-2` to
`4.0e-7`, RK6's from `3.1e-2` to `1.8e-7` -- roughly five orders of
magnitude, and RK4/RK6 no longer give identical errors. The residual
`~1e-7` is finite-run/QR statistical noise, not truncation error.

### DDE: fixed (and a post-mortem of why it looked unfixable)

The analogous change -- reconstructing the delayed history lookup at each
stage's own intra-step *time* (`step + c`, via the same interpolant, which
already accepts non-integer time arguments) instead of once per step --
is now shipped for `interpolate=True`, and restores the integrator's real
convergence order. Measured on the scalar linear DDE (`a=0.5`,
`tau=0.317`) against the Lambert-W analytic exponent:

- `heun`: order 2.00, 2.00, 2.00 across `dt` halvings from 2e-2 (errors
  9.2e-6 -> 1.4e-7); previously order ~1.0 with 1e-3-class errors.
- `rk4`: order ~4.4 (2.1e-11 at `dt=2e-2`) down to the estimation noise
  floor ~5e-14.
- `rk6`: capped at ~order 4 -- the cubic Hermite interpolant's own
  ceiling, the expected theoretical limit for this history
  representation. `rk4` is the DDE sweet spot; `rk6` buys nothing more.

**Why the first attempt at exactly this change looked ineffective**: it
was made while `_write_ring_interp` still wrote the state at time
`(t+1)*dt` under slot `t` (the off-by-one below) -- a one-full-step shift
of the effective delay, itself an O(dt) bias in *which system was being
solved*. Two independent O(dt) errors were stacked; removing either one
alone leaves the measured order at ~1. The attempt removed only the
frozen read (order still 1, blamed on the read fix, reverted); the later
off-by-one fix was validated only against the reverted, frozen-read code
(order still 1, "ruled out as the cause"). Only both together lift the
cap. The earlier "ruled out" list was accurate but incomplete evidence:
the interpolation formula, the transient, and the QR machinery were
indeed all innocent.

Two inherent limits remain, verified by re-running the sweep with a
smooth exact-solution history (`x(t) = exp(lambda t)`, no breaking
points), under which the *trajectory* itself measures ~4.4 with `rk4`
down to float64 roundoff:

- `interpolate=False` stays O(dt) by construction -- one stored sample
  frozen across the step, no sub-step history to read, plus grid-rounded
  `tau`.
- Constant-history *trajectories* cap at ~order 2 from the `t=0`
  breaking point (the slot-0 stored derivative can represent only one of
  the two one-sided slopes there -- a one-time O(dt^2) injection).
  Exponents don't feel this: the mandatory transient discards the
  breaking-point region, so `rk4` reaches its ~4th order on the exponent.

`interpolate=True` now validates `tau_steps >= 1` (`tau >= dt`) with a
clear error: per-stage reads at intra-step times up to `step + 1` must
only land on already-finalized ring-buffer slots. (Sub-step delays were
silently unsound before, too -- the stored-derivative computation already
read at `t + 1`.)

Regression coverage: `tests/test_dde_interpolation.py`
(`test_interpolate_heun_converges_at_second_order`,
`test_interpolate_rk4_reaches_hermite_accuracy_floor`,
`test_interpolate_rejects_sub_step_delay`).

## Why Adaptive ODE and DDE Interpolation Are Related but Solved Separately

Both are instances of "stop assuming everything lands exactly on a fixed,
uniform grid, and add a principled way to reconstruct values in between."
That's the extent of the connection. Combining them (an adaptive-step
DDE) is substantially harder than either alone: the ring buffer's whole
simplification -- a delayed read looks up a *stored, evenly-spaced*
sample, or (with interpolation) a value between two evenly-spaced samples
-- stops holding the moment the primal trajectory's own sample times
become irregular. An adaptive-step DDE needs history interpolation over
an *irregular* grid, which is what full continuous-history solvers like
`jitcdde` do, and is a meaningfully bigger jump than either piece here.

Recommended order, if both are pursued: DDE interpolation first (done),
adaptive ODE integration second (independent, not started), and only
after both are independently validated, revisit whether a combined
adaptive-step DDE is worth the effort, or whether "fixed-step ODE +
Hermite-interpolated DDE history" is the right permanent scope boundary.
