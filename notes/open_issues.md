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
