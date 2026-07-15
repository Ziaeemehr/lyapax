# Adaptive integration: does it give a wall-clock speedup?

**Status: investigated, not (yet) formalized as a permanent benchmark.** This
answers the open question `notes/package_review.md` flagged under
Performance/§14: current public benchmarks never characterized adaptive
integration's overhead or compared it against fixed-step integration at
*matched accuracy* rather than matched nominal `dt`. Numbers below are from
ad-hoc scripts (not committed), run on this machine. Results 1-3 are CPU
(`jax.default_backend() == "cpu"`); Result 4 covers GPU, once a
`LD_LIBRARY_PATH` environment issue unrelated to `lyapax` (Anaconda CUDA
libs shadowing the venv's own) was identified and worked around — see
Result 4 for the fix and why GPU turned out to make things worse, not
better, for the small system tested there.

## Answer

**No. Not on any system tested, on either backend, including the one kind
of system adaptive integration exists for.** Adaptive integration
(`lyapax.adaptive.diffrax_adaptive_step`) is 2-4x slower than fixed-step
RK4/RK6 at matched accuracy on a small chaotic system (Lorenz) on CPU, and
~4.5x slower on GPU (Result 4) — GPU's kernel-launch latency makes the
accept/reject loop's per-step control flow even more expensive relative to
a fixed-step scan, since a 3-dimensional system has essentially nothing for
the GPU to parallelize over. On CPU, the overhead ratio shrinks as per-step
cost grows (larger state dimension), approaching rough parity at
`d=1500-3000` — and adaptive was marginally faster at loose tolerances
there — but never produced a clear win. Even on a relaxation oscillator
(Van der Pol, Result 5) — the textbook case for adaptive stepping, where
fixed-step must resolve fast transitions everywhere even during long slow
stretches — RK4 was still ~4x faster at matched accuracy, because the
"wasted" fixed-step work was cheap enough (small state dimension) that
avoiding it saved nothing worth the control-flow overhead. **A real win
would need sharply time-varying stiffness *and* an expensive per-step cost
together — not tested, and this package's typical workloads (Lorenz/Rössler/
network-scale chaotic systems, not large stiff ODEs) don't obviously call
for testing it further.**

## Methodology

"Matched accuracy" means: for a range of `dt` (fixed-step) or `rtol`
(adaptive) values, run `lyapunov_spectrum` over the same physical time span,
measure the resulting exponent error against an exact/published reference,
then compare *wall-clock time at comparable error levels* — not at the same
`dt`/`rtol` value, which isn't a fair comparison across integrators with
different meanings of "step size." Timing is warm (second call, so JIT
trace/compile cost is excluded), following `benchmarks/lyapax/_common.py`'s
existing convention.

## Result 1 — small system (Lorenz, `d=3`)

Reference: `sum(lambda) = -(sigma + 1 + beta)` (exact structural identity,
same as `tests/test_lyapunov_core.py`'s Lorenz tests).

| target error | RK4 (warm) | RK6 (warm) | adaptive/Dopri5 (warm) |
|---|---|---|---|
| ~1e-3 | 0.34s | 0.56s | 1.32s |
| ~1e-5 | 0.38s | 0.60s | 1.37s |
| ~1e-7 | 0.38s | 0.70s | 1.71s |

Adaptive is **3-4x slower than RK4**, **~2x slower than RK6**, at every
accuracy level tested.

**Why:** a same-tolerance `diffrax.diffeqsolve` proxy run (same solver,
controller, `rtol`/`atol`, over the same physical span) shows the internal
accept/reject loop taking multiple internal steps per outer `dt`:

| rtol | accepted | rejected | internal steps per outer `dt` |
|---|---|---|---|
| 1e-4 | 8795 | 2997 | ~1.76 |
| 1e-6 | 21009 | 5177 | ~4.20 |
| 1e-8 | 50506 | 6908 | ~10.10 |
| 1e-10 | 125812 | 8670 | ~25.16 |

Each internal step carries real fixed overhead (PID controller update,
`solver.step` bookkeeping, the `jax.lax.while_loop` iteration itself) that a
fixed-step call simply doesn't pay per raw step. For a 3-dimensional system,
that overhead is not amortized by anything.

## Result 2 — does overhead matter less for large systems?

Swept a dense linear system's dimension `d` at *matched nominal* `dt=1e-2`
(not matched accuracy — just characterizing the overhead ratio itself),
100 outer steps, `k=5`:

| d | RK4 (warm) | adaptive (warm) | ratio (adaptive/RK4) |
|---|---|---|---|
| 10 | 0.21s | 0.74s | 3.5x |
| 50 | 0.18s | 0.74s | 4.1x |
| 200 | 0.23s | 0.74s | 3.2x |
| 1000 | 0.29s | 1.01s | 3.5x |
| 3000 | 4.10s | 8.38s | **2.0x** |

The ratio shrinks once the per-step matmul cost (`O(d^2)` per raw step)
starts to dominate the fixed control-flow overhead — but adaptive is still
~2x slower even at `d=3000`.

## Result 3 — matched accuracy at large `d`

Built a `d=1500` dense linear system with an *exactly known* top-3 spectrum
by construction: `A = Q @ diag(eigs) @ Q.T` for a random orthogonal `Q` and
hand-picked `eigs`, so the tracked Lyapunov exponents equal `eigs[:3]`
exactly (a linear time-invariant system's exponents are its Jacobian's
eigenvalues) — no need for a separately-converged reference.

| target error | RK4 (warm) | adaptive (warm) |
|---|---|---|
| ~5e-8 | 1.87s (`dt=0.02`, err=5.6e-8) | 2.21s (`rtol=1e-6`, err=4.1e-8) |
| ~3e-7 | 1.87s (`dt=0.02`, err=5.6e-8, over-converged) | **1.72s** (`rtol=1e-3`, err=3.4e-7) |

At `d=1500`, the gap nearly closes: adaptive is close to parity with RK4,
and at looser tolerances it was marginally *faster*. Still not a decisive
win at any point tested.

## Result 4 — GPU, and why the CPU-only conclusion above was wrong

The "GPU isn't usable" claim in an earlier version of this doc was incomplete
— see below. GPU is reachable; the fix is unrelated to `lyapax` itself.

### The CUDA environment issue

The RTX A5000 and its driver are fine. `$LD_LIBRARY_PATH` on this machine
points at an Anaconda-installed CUDA stack
(`/home/ziaee/anaconda3/lib/python3.11/site-packages/nvidia/.../lib`, ...)
*ahead of* the CUDA libraries `pip install`ed into the `lyapax` venv itself.
JAX's `cuSPARSE` load resolves against the wrong (Anaconda) copy and fails,
so it silently falls back to CPU — `JAX_PLATFORMS` alone doesn't fix this,
since the failure happens before backend selection, not because of it.

Confirmed:

```bash
env -u LD_LIBRARY_PATH /home/ziaee/envs/lyapax/bin/python -c \
  "import jax; print(jax.default_backend(), jax.devices())"
# -> gpu [CudaDevice(id=0)]
```

### GPU vs CPU, matched accuracy, Lorenz

Reran Result 1's exact comparison with `env -u LD_LIBRARY_PATH`:

| Method | GPU warm | CPU warm (Result 1) | error |
|---|---|---|---|
| RK4, `dt=0.01` | 1.77s | 0.36s | 1.02e-4 |
| RK4, `dt=0.005` | 3.00s | 0.38s | 6.16e-6 |
| Adaptive Dopri5, `rtol=1e-6` | 7.39s | 1.32-1.37s | 3.87e-4 |

**GPU is slower than CPU for both integrators here** — expected for a
3-dimensional system: GPU kernel-launch latency dominates when there's
almost no per-step work to parallelize over. Adaptive is worse off
proportionally too: **~4.5x slower than an already-more-accurate RK4 run**
on GPU, versus ~3-4x on CPU (Result 1) — GPU kernel-launch latency
compounds with the accept/reject loop's extra sequential dispatches per
outer step. So Result 2's finding (adaptive's overhead *ratio* shrinks as
per-step cost grows) doesn't rescue it here either: this system is too
small for either integrator to benefit from GPU at all, so there's no
"more expensive per-step work" for the fixed overhead to be amortized
against.

### Where the overhead comes from (verified against the source)

`lyapax.adaptive.diffrax_adaptive_step` (`src/lyapax/adaptive.py`) does not
call `diffrax.diffeqsolve` — it drives `solver.step` and `PIDController`
directly inside a `jax.lax.while_loop` (see the module's own docstring on
why: `diffeqsolve`'s default `scan_kind` rejects forward-mode `jax.jvp`).
Consequence: every outer `dt` advance re-runs `controller.init`,
diffrax's initial-step-size heuristic, and `solver.init` from scratch, then
starts a fresh `while_loop` — no state (step-size history, solver internals)
carries over between one outer step and the next. `lyapax.core`'s `_advance`
then applies `jax.jvp` through that entire per-outer-step adaptive solve for
tangent propagation.

Isolating the restart cost specifically (same physical span, same QR
cadence in time — `renorm_every * dt = 1.0` either way — only the outer
`dt`, and hence how often the controller restarts, differs), on GPU:

| configuration | warm |
|---|---|
| outer `dt=0.1`, `renorm_every=10` (10x more restarts) | 7.84s |
| outer `dt=1.0`, `renorm_every=1` | 6.25s |

Fewer restarts save about 20% — a real but secondary effect. The dominant
cost is still the accept/reject loop itself (stage evaluations, PID
bookkeeping) rather than the re-initialization overhead.

**Caveat on Result 1's step-count table:** that table's numbers came from
a direct `diffrax.diffeqsolve(t0=0, t1=500)` call that runs the controller
continuously across the whole span — unlike `lyapax`'s actual wrapper, which
restarts it every outer `dt`. That table is still a reasonable proxy for
*why* more accept/reject steps happen at tighter `rtol`, but it understates
the wrapper's real per-call overhead by not including restart cost.

### A more specialized integration design (proposal, not implemented)

A general-purpose hand-rolled Dopri5/PID replacement for diffrax is
unlikely to help — it would still need the same stage evaluations, the same
sequential accept/reject decisions, and the same dynamic loop, for
substantial correctness/maintenance cost and only marginal (pytree/generic-
controller) overhead removed. A structurally different integration boundary
looks more promising, though unverified and not attempted here:

1. Integrate one full QR/renormalization block adaptively (carrying
   controller/solver state across it), instead of restarting per raw
   outer step.
2. Propagate `(state, tangent matrix)` together as the augmented
   variational system `dx/dt = f(x)`, `dY/dt = J_f(x) Y`, with QR only at
   the renormalization boundary — closer to what the algorithm actually
   needs than an adaptive solver hidden inside what `lyapunov_spectrum`
   otherwise treats as one fixed raw step.
3. Decide explicitly whether the adaptive error norm should include the
   tangent matrix, not just the primal state — see the second subtlety
   above (primal-only error control can silently under-resolve the
   tangent direction).
4. Expose accepted/rejected step counts in any permanent matched-accuracy
   benchmark, not just this one-off investigation.

This would be a real redesign (changes `lyapax.adaptive`'s integration
boundary, not just its parameters), not a drop-in fix — flagging it here as
a direction, not proposing to build it as part of this investigation.

## Result 5 — relaxation oscillator (the textbook case for adaptive stepping)

Results 1-4 all use systems with roughly *uniform* stiffness along the
trajectory — the worst case for adaptive integration to show a benefit,
since a human picking one fixed `dt` up front loses nothing when the right
`dt` barely varies. The actual case adaptive stepping exists for is a
trajectory with *separated timescales*: long slow stretches punctuated by
fast transitions, where fixed-step must use the smallest `dt` needed
anywhere, wasting steps everywhere else. Van der Pol at large `mu` is the
standard example (`x' = y`, `y' = mu*(1 - x^2)*y - x`).

Used `mu=30`, reference `lambda1` from a fine fixed-step RK4 run
(`dt=2e-4`, ~1.5M steps, 1.6s — cheap regardless, this is only a
2-dimensional system) rather than a closed form (none exists for Van der
Pol's Lyapunov exponents):

| target error | RK4 (warm) | adaptive (warm) |
|---|---|---|
| ~2e-8 | **0.31s** (`dt=0.002`) | 1.28s (`rtol=1e-9`) |

RK4 is still **~4x faster** at matched accuracy, on the one system type
adaptive stepping is supposed to be good at. The reason: RK4 reached this
accuracy in only 150,000 steps, and because Van der Pol is a 2-dimensional
system those steps cost almost nothing (0.31s total) *regardless* of
whether the fixed `dt` was "wasteful" during the slow phase — there was
never enough absolute cost in the wasted steps for adaptive stepping's
smarter step selection to recover, let alone beat, its own fixed per-step
control-flow overhead. Loosening `rtol` didn't help either: `rtol=1e-3` and
`rtol=1e-6` gave *larger* errors (1.22e-1, 3.95e-2) than RK4's coarsest
point tested (`dt=0.02` -> 5.01e-2), and were still slower in absolute
terms (1.12s, 1.20s) than RK4's finest, most-accurate point.

**Conclusion:** a real win for `lyapax.adaptive` needs sharply time-varying
stiffness *and* an expensive per-step cost (large state dimension) at the
same time — separately, neither condition alone was enough in this
investigation (Result 3 for large-but-uniform, Result 5 for
small-but-stiff). Not tested here, and not the shape of problem this
package's validation suite otherwise targets (Lorenz/Rössler/network-scale
chaotic systems), so not chased further.

## Two real subtleties found along the way (not benchmark artifacts)

1. **`lyapax.adaptive` doesn't work with `network_problem` at all**, not
   just with DDEs. `lyapax/simulator/step.py`'s guard
   (`getattr(integrator, "_lyapax_adaptive_ode_only", False)`) rejects any
   adaptive integrator unconditionally — including a plain, non-delayed
   network (`has_delays=False`) — but its error message says "not supported
   for DDEs," which is misleading when triggered by a network call. This
   wasn't a benchmark artifact; it's why the large-`d` experiments above use
   a plain dense `ode_problem` (`state -> A @ state`) rather than
   `lyapax.network`'s `Network`/`network_problem` machinery — the intended
   "large network" test isn't actually possible with adaptive integration
   today.
2. **The step-size controller sizes steps off the *primal* trajectory's
   error, not the tangent/JVP that produces the Lyapunov exponent.** An
   early version of the `d=1500` test used a fast-decaying primal state
   (bulk eigenvalues around -20 to -40); over a 20-time-unit transient the
   primal state's norm underflowed to ~4e-11. Once the primal is
   numerically zero, `rtol*|y| + atol` collapses to just `atol`, which is
   trivially satisfied — so the controller happily took large steps that
   were far too coarse to accurately propagate the *tangent* direction,
   producing an exponent error of ~2.3 (nonsense) regardless of how tight
   `rtol` was set. This is specific to systems whose primal trajectory
   decays much faster than expected relative to the exponents being
   tracked — not a concern for bounded attractors (Lorenz, Kuramoto) or
   other realistic Lyapunov workloads, but worth knowing if a decaying
   transient is ever unusually long.

## Sample code

Condensed versions of the three checks above — not the literal scratchpad
scripts (which also did warm/first-call timing, printed every sweep point,
etc.), but enough to reproduce each result table's shape. All three assume
`os.environ["JAX_PLATFORMS"] = "cpu"` set *before* importing `jax`, and
`jax.config.update("jax_enable_x64", True)`.

**Result 1 — Lorenz, matched accuracy:**

```python
import time
import jax, jax.numpy as jnp
from lyapax import systems
from lyapax.core import lyapunov_spectrum, ode_problem
from lyapax.adaptive import diffrax_adaptive_step

sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
rhs = systems.lorenz(sigma, rho, beta)
state0 = jnp.array([1.0, 1.0, 1.0])
expected_sum = -(sigma + 1.0 + beta)
renorm_every = 10

def warm_time(fn, *a, **kw):
    fn(*a, **kw)  # first call: pays JIT trace/compile cost, discarded
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    jax.block_until_ready(r)
    return time.perf_counter() - t0, r

def run_fixed(dt, integrator):
    problem = ode_problem(rhs, state0=state0, dt=dt, integrator=integrator)
    n_steps = (int(round(500.0 / dt)) // renorm_every) * renorm_every
    return lyapunov_spectrum(
        problem, n_steps=n_steps, renorm_every=renorm_every,
        t_transient=int(round(100.0 / dt)) * dt)

def run_adaptive(rtol, dt=0.1):
    integrator = diffrax_adaptive_step(rtol=rtol, atol=rtol * 1e-2)
    problem = ode_problem(rhs, state0=state0, dt=dt, integrator=integrator)
    n_steps = (int(round(500.0 / dt)) // renorm_every) * renorm_every
    return lyapunov_spectrum(
        problem, n_steps=n_steps, renorm_every=renorm_every,
        t_transient=int(round(100.0 / dt)) * dt)

from lyapax.integrators import rk4_step
for dt in [4e-2, 2e-2, 1e-2, 5e-3]:
    warm_s, result = warm_time(run_fixed, dt, rk4_step)
    err = abs(float(jnp.sum(result.exponents)) - expected_sum)
    print(f"rk4 dt={dt}: err={err:.2e} warm={warm_s:.2f}s")

for rtol in [1e-4, 1e-6, 1e-8, 1e-10]:
    warm_s, result = warm_time(run_adaptive, rtol)
    err = abs(float(jnp.sum(result.exponents)) - expected_sum)
    print(f"adaptive rtol={rtol}: err={err:.2e} warm={warm_s:.2f}s")
```

Internal accepted/rejected step counts (the table explaining *why* adaptive
is slower) came from calling `diffrax.diffeqsolve` directly, bypassing
`lyapunov_spectrum` entirely, with the same solver/controller/tolerances:

```python
import diffrax

def term(t, y, args):
    return rhs(y)

for rtol in [1e-4, 1e-6, 1e-8, 1e-10]:
    controller = diffrax.PIDController(rtol=rtol, atol=rtol * 1e-2)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(term), diffrax.Dopri5(), t0=0.0, t1=500.0, dt0=0.1,
        y0=state0, stepsize_controller=controller, max_steps=200_000,
        saveat=diffrax.SaveAt(t1=True),
    )
    print(rtol, sol.stats["num_accepted_steps"], sol.stats["num_rejected_steps"])
```

**Result 2 — size-scaling ratio at matched nominal `dt`:**

```python
import numpy as np

gamma, G = -2.0, 0.5

def ring_matrix(d):
    A = np.full((d, d), 0.0)
    idx = np.arange(d)
    A[idx, idx] = gamma
    A[idx, (idx - 1) % d] = G
    A[idx, (idx + 1) % d] = G
    return jnp.asarray(A)

for d in [10, 50, 200, 1000, 3000]:
    A = ring_matrix(d)
    rhs_d = lambda state, A=A: A @ state
    state0_d = jnp.asarray(np.random.default_rng(0).normal(size=d) * 0.1)

    problem_rk4 = ode_problem(rhs_d, state0=state0_d, dt=1e-2, integrator="rk4")
    warm_rk4, _ = warm_time(
        lyapunov_spectrum, problem_rk4, n_steps=100, renorm_every=10, k=5)

    integrator = diffrax_adaptive_step(rtol=1e-6, atol=1e-8)
    problem_ad = ode_problem(rhs_d, state0=state0_d, dt=1e-2, integrator=integrator)
    warm_ad, _ = warm_time(
        lyapunov_spectrum, problem_ad, n_steps=100, renorm_every=10, k=5)

    print(f"d={d}: rk4={warm_rk4:.2f}s adaptive={warm_ad:.2f}s ratio={warm_ad/warm_rk4:.2f}")
```

**Result 3 — matched accuracy at large `d`, with an exact reference.** The
key design point (see "Two real subtleties" below): `top_eigs` must be
*closer to zero* than `bulk` (so QR genuinely converges to `top_eigs`, not
to some larger bulk eigenvalue), spaced ~1 apart (so QR distinguishes
adjacent tracked directions quickly), and `t_transient` long enough relative
to the *smallest* gap involved — get any of these wrong and the error
floors at some constant regardless of `dt`/`rtol`, an artifact of incomplete
QR alignment, not integration accuracy:

```python
k = 3
top_eigs = np.array([-0.1, -1.1, -2.1])          # gaps ~1 apart, closest to 0
d = 1500
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
bulk = rng.uniform(-10.0, -5.0, size=d - k)       # clear gap *below* top_eigs
eigs = np.concatenate([top_eigs, bulk])
A = jnp.asarray((Q * eigs) @ Q.T)                 # symmetric -> real eigenvalues
rhs_big = lambda state: A @ state
state0_big = jnp.asarray(np.random.default_rng(1).normal(size=d) * 0.1)

T_TRANSIENT = 15.0  # >> 1 / (min gap) so tangent alignment actually converges

for dt in [2e-2, 1e-2, 5e-3]:
    problem = ode_problem(rhs_big, state0=state0_big, dt=dt, integrator="rk4")
    n_steps = (int(round(3.0 / dt)) // 5) * 5
    warm_s, result = warm_time(
        lyapunov_spectrum, problem, n_steps=n_steps, renorm_every=5, k=k,
        t_transient=T_TRANSIENT)
    err = float(jnp.max(jnp.abs(result.exponents - jnp.asarray(top_eigs))))
    print(f"rk4 dt={dt}: err={err:.2e} warm={warm_s:.2f}s")

for rtol in [1e-3, 1e-6, 1e-9]:
    integrator = diffrax_adaptive_step(rtol=rtol, atol=rtol * 1e-2)
    problem = ode_problem(rhs_big, state0=state0_big, dt=0.1, integrator=integrator)
    n_steps = int(round(3.0 / 0.1))
    warm_s, result = warm_time(
        lyapunov_spectrum, problem, n_steps=n_steps, renorm_every=5, k=k,
        t_transient=T_TRANSIENT)
    err = float(jnp.max(jnp.abs(result.exponents - jnp.asarray(top_eigs))))
    print(f"adaptive rtol={rtol}: err={err:.2e} warm={warm_s:.2f}s")
```

## What wasn't checked

- **A stiff/relaxation-type system that is also large (`d` >> 2-3).**
  Result 3 (large, uniform-stiffness) and Result 5 (small, sharply
  time-varying stiffness) each tested one of the two conditions
  identified as necessary for a win, but not both together — the only
  remaining case with a plausible path to a real speedup. Non-linear
  large systems don't have an equally cheap exact reference the way
  Result 3's constructed linear system does, and (per the first subtlety
  above) aren't reachable through `network_problem` with the adaptive
  integrator today regardless — so this would need a large single-`ode_problem`
  stiff system (e.g. a stiff reaction network or coupled relaxation
  oscillators expressed as one flat state vector) and a non-exact,
  fine-dt reference the way Result 5 used. Not attempted here; the
  investigation stopped once both individual conditions were confirmed
  insufficient on their own.
- **GPU at large `d`.** Result 4's GPU numbers are Lorenz/Van-der-Pol-only
  (small systems); Results 2/3's large-`d` findings were CPU-only. A GPU
  rerun of Result 3 at `d=1500` would show whether GPU's per-step
  parallelism changes the near-parity finding there — not done here.

## Bottom line for the package review

`notes/package_review.md`'s Performance section (§7) already listed this as
open ("current public benchmarks do not characterize this overhead"). This
investigation confirms the suspicion the review only asserted qualitatively:
adaptive integration is not a general speed win over fixed-step RK4/RK6 for
`lyapax`'s Lyapunov workload — not on small systems, not on large ones, not
on either backend, and not even on the relaxation-oscillator case adaptive
stepping is specifically designed for (Result 5). Its value proposition is
convergence guarantees / tolerance-driven accuracy control and decoupling
step size from renormalization interval, not raw throughput — the module
docstring (`src/lyapax/adaptive.py`) and README now say so plainly, instead
of only implying it's a speed-oriented alternative to fixed-step
integration. Recommendation: keep the feature (optional, isolated, already
tested/documented — removing working code with zero cost to non-users is
pure loss) and do not invest in the redesign sketched in Result 4 without a
concrete large-and-stiff use case motivating it; the value it provides
(tolerance control, forward-mode-differentiable adaptive stepping) doesn't
require a speedup to justify existing. Separately, this environment's
`LD_LIBRARY_PATH` issue (Anaconda CUDA libs shadowing the venv's own) is
worth fixing regardless of this investigation — it silently made every
prior GPU-gated check (`tests/test_gpu.py`, the
`examples/14_gpu_acceleration.py` demo) run on CPU without any error.

If this should ship as a permanent, reproducible benchmark
(`benchmarks/lyapax/`, following the existing `_common.py`/`emit()`
convention) or an `examples/`-style demo, that's a follow-up, not done
here.

My conclusion: if adaptive integration was added exclusively for speed, the current evidence does not justify it as the default fast path. Its real value is tolerance-based accuracy and handling trajectories with strongly varying time scales. For smooth non-stiff systems and especially GPUs, fixed-step RK methods with static, fusible control flow are likely to remain faster. A specialized Lyapax adaptive path is worth prototyping, but replacing Diffrax itself should be the last step, only if profiling that redesigned path shows Diffrax-specific overhead is still dominant.