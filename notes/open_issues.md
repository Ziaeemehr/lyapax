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
- The ring-buffer write/read off-by-one, item 2 below -- confirmed and
  fixed, but the `interpolate=True` convergence order measured cleanly at
  ~1.0 (1.01, 1.00, 1.00 across three `dt` halvings) both before and after
  the fix. Ruled out, not the explanation.

Not yet tried: instrumenting a step-by-step comparison of the DDE
trajectory against a very fine reference to see exactly where per-step
error is introduced, rather than only measuring aggregate convergence
order.

Full writeup: `notes/stepping_accuracy_review.md`, Part C.

### Practical impact on package usage

Only the DDE path (`lyapunov_spectrum_dde`, `dde_problem`,
`network_dde_problem`) is affected. Plain ODEs and zero-delay coupled
networks are not -- those get their integrator's real nominal order (item
3, fixed).

- **Integrator choice doesn't buy accuracy for a DDE.** `euler`/`heun`/
  `rk4`/`rk6` all cap at the same ~O(dt) error once a delay is involved,
  because the delayed-history lookup, not the ODE part, is the bottleneck.
  Picking `rk6` for a delayed system costs 8 stages/step for the same
  asymptotic accuracy `heun` gives at the same `dt`.
- **A single run at one `dt` is not enough for real precision.** Error
  shrinks linearly, not at the interpolation formula's own ~4th order.
  Halving `dt` roughly halves the error, not the usual "extra digit per
  10x-smaller step." Always sweep `dt` (2-3 halvings) and check the
  exponent has stabilized to the digit you need, the way
  `tests/test_linear_scalar_dde_dt_convergence` and
  `examples/plot_13_dde_history_interpolation.py` already do, rather than
  trusting one run.
- **Cost compounds with accuracy.** The ring buffer (and its tangent/QR
  counterpart) is sized `~tau/dt`, so shrinking `dt` for accuracy also
  linearly grows memory and step count -- there's no higher-order
  shortcut around this the way there is for zero-delay systems.
- **Existing examples/tests are fine, but only to the precision they
  claim.** `examples/plot_08`/`plot_09` (dt=1e-3-1e-2, tau~0.3) and the
  Lambert-W/Mackey-Glass tests use tolerances (0.01-0.02) already loose
  enough to cover this bias -- not evidence that finer precision is free.
  Treat any single-`dt` DDE result as good to about one significant digit
  unless you've verified convergence yourself at your own `tau`/`dt`.

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

**Does not explain item 1.** The `interpolate=True` DDE convergence order
was measured directly before and after this fix and is unchanged (~1.0
both times) -- see item 1's ruled-out list. This was a real, independent
correctness bug (it silently shifted every DDE/per-edge-delay delay by
one step short), just not the cause of the O(dt) convergence cap.

Full writeup: `notes/stepping_accuracy_review.md`, Part C;
`notes/possible_solution_to_open_issues.md` for the original investigation
plan this followed.

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
