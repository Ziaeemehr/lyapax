# Stepping Accuracy Review: Adaptive ODE Integration and DDE History Interpolation

## Summary

Two open accuracy limitations share one root cause: both the ODE and DDE
engines currently assume every quantity they need lands *exactly* on a
fixed, uniform `dt` grid.

- For ODEs, that assumption shows up as "no adaptive step-size control" --
  `lyapax.integrators`' methods (`euler`, `heun`, `rk4`, `rk6`) all take one
  fixed `dt` per call, with no mechanism to take smaller internal steps
  where the trajectory is fast-changing and larger ones where it's slow.
- For DDEs, the same assumption shows up as "no sub-step delay
  interpolation" -- `resolve_tau_steps` rounds a physical delay `tau` to
  the nearest integer number of `dt` steps, and every delayed read in
  `lyapax.simulator.step` (`_read_uniform_delayed_cvar`,
  `_read_delayed_coupling`) pulls an exact stored ring-buffer sample,
  never a value *between* two stored samples.

Both are documented, known tradeoffs, not oversights: the DDE case is
"risk #4" in `notes/milestones.md`, explicitly flagged as "revisit if the
convergence-vs-`dt` test... shows the rounding error is unacceptable," and
`notes/api_design_review.md`'s "Adaptive Integrators" section scopes the
ODE case as future work. This note works out *how* to actually resolve
each one, and why -- despite the shared root cause -- they should be
solved separately rather than as one combined "fully adaptive DDE"
feature.

## Part A: Adaptive ODE Integration

### Current behavior

`lyapax.ode_step(rhs, dt, integrator=...)` builds a `state -> new_state`
map that advances by exactly `dt` using a fixed number of internal stages
(1 for Euler, 2 for Heun, 4 for RK4, 8 for RK6). `lyapunov_spectrum`
doesn't know or care how `step_fn` gets from `state` to `state` `dt`
later -- it only needs `step_fn` to be a pure, differentiable function of
`state`, called once per raw step inside a fixed-length `jax.lax.scan`
(`renorm_every` raw steps per QR block).

This means the *outer* structure -- fixed-`dt` sampling, fixed-length
scan, QR every `renorm_every` samples -- is already decoupled from what
happens *inside* one `step_fn` call. That decoupling is exactly what
makes adaptive integration tractable here without touching `core.py` at
all.

### Proposed design

An adaptive integrator is just another `integrator` value (a callable
`(rhs, dt) -> step_fn`, same as `rk4_step`/`rk6_step`), except that
internally, `step_fn(state)` runs a `jax.lax.while_loop`: take an internal
step of a proposed size `h`, estimate its local error against an embedded
lower-order companion formula, accept or shrink-and-retry, and keep going
until the accumulated internal time reaches `dt` exactly. `dt` keeps
today's meaning -- the fixed interval between renormalization samples --
it just stops also being "the integrator's own step size."

```python
integrator = lyapax.integrators.AdaptiveRK45(rtol=1e-6, atol=1e-9)
step = lyapax.ode_step(rhs, dt=0.1, integrator=integrator)   # dt = sampling interval
result = lyapax.lyapunov_spectrum(step, state0, dt=0.1, n_steps=...)
```

### Why this fits better than expected

The classic objection to adaptive-step + autodiff is that a
variable-trip-count loop is hard to differentiate: reverse-mode AD
through `jax.lax.while_loop` needs to replay an a priori unknown number of
steps backward, which JAX does not support directly. But this codebase's
tangent engine is `jax.jvp`-based, not `jacrev`/`vjp`-based --
`lyapax/core.py`'s module docstring is explicit that this was a deliberate
choice (cost: O(k) forward passes per raw step, not O(d) for a dense
Jacobian). Forward-mode JVP through a `while_loop` is not the hard case:
the primal trajectory runs concretely (its accept/reject decisions are
resolved to actual numbers during the forward pass), and the tangent
simply rides along the same accepted path. Nothing about the tangent
direction can change which steps get accepted, so there is no new
autodiff machinery to build -- just more arithmetic per `step_fn` call.

`jax.lax.while_loop`'s trip count is a runtime value, not a compile-time
shape, so a varying number of internal substeps does not by itself
trigger recompilation under `jax.jit`.

### What's still genuinely open

- **Non-smoothness at accept/reject boundaries.** A system parameter could
  in principle shift exactly which internal steps get accepted, making
  "exponent as a function of that parameter" non-smooth at a measure-zero
  set of points. Worth documenting, not worth blocking on -- the same
  caveat already applies to any jitted control flow with data-dependent
  branching, and is typically negligible in practice.
- **Hand-rolled vs. dependency.** `diffrax` already solves this problem
  robustly (adaptive Dormand-Prince/Tsit5, JAX-native, differentiable,
  dense output). Depending on it would be far less risky than hand-rolling
  an embedded RK pair and step-size controller from scratch -- but this
  repo's stated philosophy is to vendor a minimal simulator rather than
  import one (see `src/lyapax/simulator/NOTICE.md`), so this is a real
  choice to make explicitly, not default silently either way.
- **Validation plan.** Whichever is chosen, verify the same way RK6 was
  verified in this session: (1) a symbolic stability-function check
  (`R(z)` vs. `e^z`) is not applicable to an adaptive method the same way,
  so instead (2) an empirical convergence check as `rtol`/`atol` shrink,
  against a system with a known exact answer (e.g. the linear
  eigenvalue check in `plot_01_linear_ode.py`, or the Lorenz
  sum-of-exponents invariant in `plot_07_speed_and_accuracy.py`), plus (3)
  a cross-check that a very tight `rtol`/`atol` reproduces the existing
  fixed-step RK6 answer on the same system.

### Recommendation

Worth pursuing, but as a scoped follow-up: a short implementation of one
embedded pair (e.g. Bogacki-Shampine RK23 or a Dormand-Prince RK45) behind
`jax.lax.while_loop`, validated per the plan above, before deciding
whether it's worth also wiring in `diffrax` as an optional dependency for
the harder cases (stiffness, dense output).

## Part B: DDE History Interpolation

### Current behavior

`resolve_tau_steps(tau, dt)` (`src/lyapax/dde.py`) rounds a physical delay
to `tau_steps = max(1, round(tau / dt))`, and warns if the relative gap
between `tau` and `tau_eff = tau_steps * dt` exceeds `warn_tol`. Every
downstream Lyapunov spectrum is computed for `tau_eff`, not the exact
`tau` requested. The ring buffer (`lyapax.simulator.step`) then reads the
delayed coupling-variable state via exact integer indexing
(`_read_uniform_delayed_cvar`, `_read_delayed_coupling`) -- there is no
sub-step interpolation, so a delayed value always comes from a stored
sample, never a reconstruction of the value *between* two samples.

This is deliberate and already documented (`notes/milestones.md`, risk
#4): "simpler and fully autodiff-friendly, but means `tau` is only exact
up to `dt` rounding." `tests/test_dde.py::test_linear_scalar_dde_dt_convergence`
already measures the practical size of this: the same physical `tau` at
`dt=2e-2` vs. `dt=1e-2` must agree to within `0.01` on the Lyapunov
exponent (a loose bound, chosen *because* of this rounding error, not
despite it) -- concrete evidence that the current scheme's accuracy is
capped by `tau` rounding, not by the underlying Euler/Heun/RK integrator's
own order.

### The problem this creates

Because `tau` is snapped to the nearest grid point, two things are true
that wouldn't be true for a genuinely continuous-history DDE solver:

1. Refining accuracy requires shrinking `dt` globally. There is no way to
   improve *just* the delay accuracy independently of the ODE integration
   accuracy -- they're coupled through the same `dt`, even though `rk6`
   already makes the ODE side's own error negligible at practical `dt`.
2. The rounding error in `tau_eff` is an `O(dt)` bias in *what system is
   actually being simulated* (a slightly different delay than requested),
   not a shrinking truncation error in solving the same system more
   accurately -- so it doesn't improve at the rate a higher-order
   integrator's error does, and can dominate the total error budget even
   when using `rk6`.

### Implemented: Hermite-interpolated history reads

This is exactly what established adaptive-step DDE solvers do (the
`jitcdde`/`jitcdde_lyap` precedent this repo's `dde.py` module docstring
already compares itself against): reconstruct the delayed value from a
**cubic Hermite interpolant** built from the value *and derivative* at the
two grid points bracketing `t - tau`, rather than snapping to the nearer
one. Implemented as `make_step_fn(..., interpolate=True)`, and surfaced
through `dde_problem(..., interpolate=True)` /
`network_dde_problem(..., interpolate=True)` -- default `interpolate=False`
preserves today's grid-snapped behavior exactly, byte-for-byte.

Concretely:

1. **Store more per ring-buffer slot.** Alongside the coupling-variable
   state `cvar_state` at each step, also store its time-derivative
   `d(cvar_state)/dt` -- available for free, since it's exactly the
   relevant slice of `dfun(state, coupling, params)` already computed that
   step, just not currently saved. The buffer shape grows from
   `(horizon, n_cvar, n_nodes)` to two same-shaped arrays (value and
   derivative), or one `(horizon, 2, n_cvar, n_nodes)` array.
2. **Read side: interpolate, don't snap.** For a real-valued
   `t_frac = (t - tau) / dt`, take `i0 = floor(t_frac)`, `i1 = i0 + 1`,
   `theta = t_frac - i0 in [0, 1)`, look up `(y0, y0')` at `buf[i0]` and
   `(y1, y1')` at `buf[i1]`, and evaluate the standard cubic Hermite
   basis:

   ```
   H(theta) = h00(theta)*y0 + h10(theta)*dt*y0'
            + h01(theta)*y1 + h11(theta)*dt*y1'
   ```

   using `H(theta)` as the delayed coupling-variable value instead of
   `buf[(step - tau_steps) % horizon]`. `tau` no longer needs to be
   rounded to an integer number of steps at all -- `theta` absorbs the
   fractional part exactly.
3. **Tangent propagation needs no new machinery.** `H` is a fixed
   (`theta` depends only on `tau`, `dt`, and the integer step counter --
   never on `state`), smooth, linear function of four buffer entries.
   `jax.jvp` already differentiates through it automatically, the same
   way it already differentiates through every other read of `buf`. The
   only change is that the *tangent* buffer's per-slot dimension also
   doubles (value + derivative), so `d_buf` (and therefore `d_total`,
   `k`'s practical ceiling, and QR cost `O(d_total * k^2)`) roughly
   doubles *per stored horizon slot* -- but `horizon` itself may be able
   to shrink for the same delay, since interpolation error, not grid
   density, becomes the accuracy bottleneck. Net effect on cost is not
   obviously negative.

### Scope limits worth stating explicitly

- **Uniform delay first.** The uniform-`tau_steps` coupling path
  (`_read_uniform_delayed_cvar`, used whenever a custom `coupling_fn` is
  given) is the natural first target -- one `theta` per read, shared
  across all nodes. The legacy per-edge `delay_steps` matrix path
  (`_read_delayed_coupling`) would need a *different* `theta` per edge
  (since each edge's delay can differ), which is a real generalization,
  not a drop-in change -- consistent with `make_step_fn`'s existing
  documented scope split between these two paths.
- **Still assumes constant, known-ahead-of-time `tau`.** State-dependent
  or time-varying delays are a much larger design change and are not
  addressed by interpolation alone; out of scope here as they are
  everywhere else in this codebase today.
- **Does not, by itself, help precision-limited systems.** Interpolation
  fixes the `tau`-rounding bias; it does nothing about the existing
  float32/float64 or `renorm_every` overflow risks already documented in
  `notes/milestones.md`.

### Validated: what actually happens, and an important related finding

`tests/test_dde_interpolation.py` and
`examples/plot_13_dde_history_interpolation.py` check this the same way
`rk6` was validated: an exact-integer-`tau` case (interpolation must
reduce to the grid-snapped answer bit-for-bit, since `theta=0` at every
read -- confirmed to ~1e-10) and an empirical convergence-order sweep on
the scalar linear DDE against its Lambert-W exact answer, at a `tau`
deliberately not a multiple of any swept `dt`.

The measured result is *not* what the design above predicted. Grid-
snapping's error vs. `dt` is confirmed non-monotonic exactly as expected
(e.g. at `tau=0.317`: `tau_eff` comes out `0.320, 0.320, 0.320, 0.315,
0.3175` at `dt = 0.04, 0.02, 0.01, 0.005, 0.0025` -- getting *worse* at
`dt=0.005` than at `dt=0.01`, and the exponent error follows suit).
Interpolation fixes exactly that: `tau_eff` is `0.317` exactly at every
`dt`, and the error decreases smoothly and monotonically. But its
empirical order is close to **1**, not the higher order (`O(dt^3)`-
`O(dt^4)`) a Hermite interpolant built from exact derivatives should give.

The reason turned out to be unrelated to the interpolation formula
itself, or to how the stored derivative is computed (reusing the coupling
already read at the start of the step vs. re-reading it at the new time
after advancing were both tried; both give the same ~order-1 result).
It's a **separate, pre-existing limitation, shared by every integrator in
this package**: `_euler`/`_heun`/`_rk4`/`_rk6` (`lyapax.simulator.step`)
all read coupling *once* per step and hold it fixed across that step's
internal stages. For a zero-delay, non-trivially-coupled network (`G !=
0`), this was checked directly: sweeping `dt` on the same linear 4-node
network from `plot_04_linear_network.py` with `G=0.5`, `rk4` and `rk6`
give **identical errors to 4+ significant figures at every `dt`**, both
converging at order ~1 -- RK6's nominal 6th-order accuracy is not actually
realized for any coupled (delayed or not) system today, only for
uncoupled ones (`G=0`, which is what this session's earlier `rk6`
validation happened to use). Interpolation inherits this same cap rather
than introducing a new one -- it still delivers exactly what it promises
(an arbitrary `tau` used exactly, with smooth, predictable convergence),
just capped at first order like everything else that involves nonzero
coupling, delayed or not.

Fixing the underlying cap means recomputing coupling at each integrator
substage instead of freezing it once per step -- `_euler`/`_heun`/`_rk4`
gained a `coupling_at(y_stage, c)` callable (called fresh at each stage,
`y_stage` = that stage's own intra-step state, `c` = its Butcher node),
and `rk6_combine` (`lyapax/integrators.py`) was generalized from
`f(y) -> dy` to `f(y, c) -> dy` so `_rk6` could do the same, reusing
`rk6_combine`'s own already-verified `c_i` values (`RK6_STAGE_C`) rather
than hand-deriving a second set.

### Follow-up: fixed for zero-delay, not yet for DDE

For the **zero-delay** case, this is fully validated: `coupling_at` is
called with that stage's own updated state, so a genuinely coupled
network (`G != 0`) is now integrated as honestly as an uncoupled one.
Re-running the same coupled-network sweep from above: at `dt=0.02`, RK4's
error dropped from `3.1e-2` to `4.0e-7`, RK6's from `3.1e-2` to `1.8e-7`
-- roughly **five orders of magnitude**, and RK4/RK6 no longer give
identical errors (RK6 is now measurably better, as its nominal order
implies). At that point the remaining ~`1e-7` residual is dominated by
finite-run/QR statistical noise, not integration truncation error, so
finer `dt` doesn't shrink it further -- the coupling term is no longer
the bottleneck at all.

For the **delayed** case, the analogous change was attempted --
recomputing the delayed history lookup at each stage's own intra-step
time (`step + c`, using the already-built Hermite interpolant, which
already accepts non-integer time arguments) instead of once per step --
but it did **not** reduce the empirical convergence order below ~1, and
was reverted rather than shipped half-understood. Before reverting, three
explanations were ruled out:

- **The interpolation formula itself**: tested in isolation (a known
  smooth function, `sin(t)`, with exact stored value/derivative pairs at
  synthetic grid points) it recovers close to its expected 4th order.
- **Initial-transient / discontinuity effects**: DDE constant-history
  initial conditions are known to create a derivative discontinuity at
  `t=0` that can locally degrade interpolation accuracy for a few delay
  cycles -- but the ~order-1 behavior was identical whether the
  comparison point was `t=5`, `t=25`, or `t=50` (tens of delay cycles
  past the initial condition), so it isn't a lingering transient effect.
- **The Lyapunov/QR machinery**: the *primal trajectory* alone (no
  tangent propagation, no QR renormalization at all -- just running
  `step_fn` forward and comparing against a `dt=1e-4` reference) shows the
  same ~order-1 convergence, so the limitation is in the state
  integration itself, not in how the exponent is estimated from it.

One more thing was found but not conclusively tied to the above: a
synthetic test of the raw ring-buffer read/write pair (writing known
values at successive integer steps, then reading back a chosen offset)
showed what looks like an off-by-one between which time a write's slot
index is supposed to represent and which time a matching read actually
retrieves. A direct end-to-end check (a small `tau_steps` where a
one-step shift would be easily visible in the resulting exponent) did
*not* cleanly match a simple "effective delay is one step short"
hypothesis, though, so this may be a real, separate, pre-existing
indexing subtlety worth its own dedicated investigation -- or it may not
be the explanation for the order-1 ceiling at all. Recorded here as an
open lead, not a diagnosis.

**Current state**: `has_delays=True` (both `interpolate=True` and the
grid-snapped default) still reads coupling once per step, unchanged from
before this follow-up -- i.e., DDE Lyapunov exponents in this package
remain capped at ~O(dt) regardless of integrator or interpolation choice,
for a reason that is not yet fully understood. This is a genuine open
problem, not a scheduling choice deferred for convenience like the rest
of this note.

## Why These Two Are Related but Shouldn't Be Solved Together

Both problems are instances of "stop assuming everything lands exactly on
a fixed, uniform grid, and add a principled way to reconstruct values
in between." That's the extent of the connection -- combining them (an
adaptive-step DDE) is substantially harder than either alone, because the
ring buffer's whole simplification (a delayed read is always looking up a
*stored, evenly-spaced* sample or, with the interpolation above, a value
between two evenly-spaced samples) stops holding the moment the primal
trajectory's own sample times become irregular. An adaptive-step DDE
would need history interpolation over an *irregular* grid, which is
exactly what full continuous-history solvers like `jitcdde` do, and is a
meaningfully bigger jump than either piece here.

Recommended order, if both are pursued:

1. DDE Hermite interpolation, on the existing fixed-step engine (Part B)
   -- self-contained, directly answers an already-flagged, already
   partially-measured accuracy risk, and does not require deciding
   anything about adaptive integration first.
2. Adaptive ODE integration (Part A), independently, on the plain
   (non-delayed) engine.
3. Only after both are independently validated: revisit whether a
   combined adaptive-step DDE is worth the substantially larger design
   effort, or whether "fixed-step ODE integration + Hermite-interpolated
   DDE history" is simply the right permanent scope boundary for this
   package.
