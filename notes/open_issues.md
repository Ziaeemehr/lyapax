# Open Issues

Not-yet-resolved items, as of the stepping-accuracy work in
`notes/stepping_accuracy_review.md`. For older, unrelated follow-ups
(package-review deferrals: `exec()` trust docs, dt-convergence test
coverage, `Protocol` types, dependency pinning, etc.), see
`notes/milestones.md`'s M8 checklist instead -- this file only tracks the
items below.

## 1. DDE Lyapunov exponents converge at only ~O(dt)

The scalar linear DDE (`lyapax.dde`), tested under every integrator
(`euler`/`heun`/`rk4`/`rk6`) and both `interpolate` settings, shows
empirical convergence order ~1 as `dt` shrinks -- far below what `rk6`
achieves on the same system without delay. Root cause unknown.

Ruled out:
- The Hermite interpolation formula itself (tests at ~4th order in
  isolation, against a known smooth function).
- Initial-transient / discontinuity effects from the constant-history
  convention (order-1 behavior is identical at `t=5`, `25`, and `50` --
  tens of delay cycles past the initial condition).
- The Lyapunov/QR estimation machinery (the *primal trajectory* alone,
  compared against a `dt=1e-4` reference with no tangent propagation at
  all, shows the same ~order-1 convergence).
- Coupling frozen across RK stages, for the zero-delay case (fixed
  separately, see item 3 below) -- the analogous per-stage fix for the
  delayed lookup was attempted and did **not** improve DDE's order.

Not yet tried: instrumenting a step-by-step comparison of the DDE
trajectory against a very fine reference to see exactly where per-step
error is introduced, rather than only measuring aggregate convergence
order.

Full writeup: `notes/stepping_accuracy_review.md`, Part C.

## 2. Suspected ring-buffer read/write off-by-one (unconfirmed)

A synthetic test of `_write_ring`/`_read_uniform_delayed_cvar`
(`lyapax/simulator/step.py`) -- write known values at successive integer
steps, then read back a chosen offset -- showed what looks like a
mismatch between which time a write's slot index is supposed to represent
and which time a matching read actually retrieves.

A direct end-to-end check (a small `tau_steps` where a one-step shift in
the effective delay would be clearly visible in the resulting Lyapunov
exponent) did **not** cleanly confirm a simple "effective delay is one
step short" hypothesis. May or may not be related to item 1. Needs a
dedicated investigation: trace the exact write/read index correspondence
through a full step, independent of any interpolation or per-stage
coupling changes, to confirm or rule this out on its own.

Full writeup: `notes/stepping_accuracy_review.md`, Part C.

## 3. Coupling frozen across RK stages -- fixed for zero-delay, not DDE

Every integrator used to read coupling once per step and hold it fixed
across that step's internal RK stages, rather than recomputing it fresh
at each stage the way a correct RK implementation of the full right-hand
side would. This capped accuracy at `O(dt)` for any genuinely coupled
system, delayed or not, regardless of the base method's nominal order.

- **Zero-delay networks: fixed and validated.** `coupling_at(y_stage, c)`
  is now called fresh at each stage. A coupled linear network's error
  dropped ~5 orders of magnitude at the same `dt`, and `rk4`/`rk6` no
  longer give identical errors.
- **DDE: attempted, reverted.** The analogous fix (recompute the delayed
  history lookup at each stage's own intra-step time) did not improve
  DDE's convergence order -- this is the same open problem as item 1, not
  a separate bug.

Full writeup: `notes/stepping_accuracy_review.md`, Part C.

## 4. Adaptive ODE integration -- not started

A design exists (`notes/stepping_accuracy_review.md`, Part A: an
`integrator` value backed by `jax.lax.while_loop`, `dt` kept as the
outer sampling interval rather than the integrator's own step size), but
no implementation has been written. Open questions there: hand-rolled
embedded pair vs. depending on `diffrax`; non-smoothness at accept/reject
boundaries when differentiating w.r.t. a system parameter.

## 5. Combined adaptive-step DDE -- correctly out of scope for now

Would need history interpolation over an *irregular* grid (since an
adaptive integrator's own sample times aren't evenly spaced), which is
substantially harder than either item 3/4 alone. Not worth attempting
until items 1 and 4 are each resolved independently.
