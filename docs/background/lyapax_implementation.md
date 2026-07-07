# How lyapax is implemented

This page covers the design decisions specific to lyapax: matrix-free
tangent propagation, the integrator-agnostic engine, coupling as a plain
callable, the DDE ring buffer with its two history-read modes, batched
parameter sweeps, and the numerical-precision requirements that follow
from the algorithm. For the algorithm itself, see
{doc}`lyapunov_exponents`.

(matrix-free-tangent)=
## Matrix-free tangent propagation

The Benettin/QR method needs the action of the step map's Jacobian on
$k$ tangent vectors, where $k \le d$ is the number of exponents tracked.
lyapax never forms the dense $d \times d$ Jacobian: instead each of the
$k$ tangent columns is propagated by one {func}`jax.jvp` (forward-mode
directional derivative) call, and the $k$ calls are batched together
with {func}`jax.vmap`.

Cost is therefore $O(k)$ forward-mode passes per raw step, not $O(d)$
for a dense Jacobian. This is the entire benefit of tracking a *partial*
spectrum: a `jax.jacfwd`-based implementation always computes all $d$
Jacobian columns regardless of how many are used, so asking for the
leading $k$ exponents would save nothing. For DDE systems the gap is
larger still, because there the differentiated carry includes the whole
history ring buffer, and its dimension grows with both network size and
buffer depth.

## Any fixed-step map, integrator-agnostic

The Lyapunov engine (`lyapunov_spectrum`) only ever sees a plain,
differentiable `state -> new_state` function. It does not know or care
whether that function came from a fixed-step Euler / Heun / RK4 / RK6
integrator, a hand-written discrete map (logistic, Hénon), or a coupled
network step. The same QR/Benettin code serves every example in the
gallery without a system-specific branch anywhere in the engine. For a
discrete map, pass `dt=1.0` and read the exponents as per-iterate rates.

## Coupling as a plain callable

Network dynamics are specified as two functions:

- `dfun(state, coupling, params) -> dstate` — the per-node dynamics;
- `coupling(cvar_state, weights, params) -> coupling` — how each node
  aggregates its neighbours' coupling variables.

Built-in linear, sigmoidal, and Kuramoto coupling rules are provided,
but any user-written callable with the same signature works — there is
no registry or dispatch layer to extend. The front-door constructors
`network_problem` / `network_dde_problem` just wire a `Network`
(topology and weights), a `dfun`, and a coupling callable together.

(precision-requirements)=
## Precision requirements: use float64

Lyapunov exponents are long-horizon averages of accumulated log-growth
rates. Under JAX's default float32 the estimates silently degrade: small
per-step rounding errors accumulate over the run, and the QR diagonals
of fast-shrinking directions underflow well before float64 would. Enable
x64 before creating any arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

lyapax warns (rather than raising) when it detects a float32 state with
x64 disabled, since a caller computing a deliberately short, coarse
estimate may accept the precision loss — but the float32 default is very
rarely what you actually want here.

(choosing-renorm-every)=
## Choosing `renorm_every`

Between QR renormalizations, each tangent column grows or shrinks
roughly like $\exp(\lambda_i \cdot \texttt{renorm\_every} \cdot dt)$.
Larger `renorm_every` reduces QR overhead, but if
$\exp(|\lambda_{\max}| \cdot \texttt{renorm\_every} \cdot dt)$
approaches the float64 range, tangent vectors overflow (or the slowest
direction underflows) between renormalizations and the run produces
non-finite estimates. Keep the exponent argument comfortably small; when
in doubt, renormalize more often. `lyapunov_spectrum(...,
check_finite=True)` raises as soon as any running estimate goes
non-finite (only usable outside `jax.jit`).

## DDE support: an augmented `(state, ring buffer)` carry

For systems with a transmission delay, `lyapunov_spectrum_dde`
generalizes the same Benettin/QR idea to a carry containing both the
current state and a fixed-depth ring buffer of recent history, and
differentiates through both jointly. This follows Farmer's classic
treatment of delay systems (see {doc}`lyapunov_exponents`): a delayed
sensitivity $\partial f / \partial x(t - \tau)$ is exactly as real as
the instantaneous one, and dropping it gives wrong exponents, not just
imprecise ones.

(dde-history-interpolation)=
### Grid-snapped vs. interpolated history reads

Two history-read modes exist:

- **Grid-snapped (default).** The delay $\tau$ is rounded to the nearest
  whole number of `dt` steps, and every history read pulls one exact
  stored ring-buffer sample. Simple and fully differentiable — but the
  delay actually simulated ($\tau_\text{eff}$) is not exactly the
  requested $\tau$. This is an $O(dt)$ bias in *which system is being
  simulated*, not a truncation error, so it does not shrink smoothly or
  monotonically as `dt` is refined (lyapax warns when the rounding is
  material, and reports $\tau_\text{eff}$).
- **Cubic-Hermite interpolated (`interpolate=True`).** The ring buffer
  stores (value, derivative) pairs, so any intra-step history read can
  be reconstructed with a cubic Hermite interpolant — the same approach
  established adaptive-step DDE solvers use. $\tau$ is used exactly, no
  rounding, and convergence in `dt` becomes smooth and monotone at the
  integrator's own order (up to the interpolant's ~4th-order ceiling).
  The cost: each buffer slot stores and differentiates twice as much
  data.

(dde-rk-stage-order)=
### Delays, coupling, and Runge–Kutta stage order

A subtlety worth knowing about because it silently affects accuracy in
naive implementations: a Runge–Kutta method only achieves its nominal
order if the *complete* right-hand side — including network coupling and
delayed-history terms — is re-evaluated fresh at each internal stage's
own intra-step state and time. Freezing the coupling or the history read
once per step caps convergence at first order for any coupled or
delayed system, regardless of the base method's nominal order (RK4 and
RK6 produce identical, order-1 errors).

lyapax therefore recomputes coupling at every stage's own intra-step
state estimate, and (with `interpolate=True`) reconstructs the delayed
history at each stage's own intra-step time via the Hermite interpolant.
In grid-snapped mode the history read is still once-per-step — one more
reason to prefer `interpolate=True` when accuracy in `dt` matters.

## Batched parameter sweeps

The network/DDE step carries `params` as data rather than closing over
it, so a whole grid of parameter values can be swept as one
{func}`jax.vmap`-batched XLA call (`sweep_lyapunov_spectrum`) instead of
one Python-level engine call per grid point.

## GPU execution

Nothing in lyapax is backend-specific: JAX picks the backend, and the
same code runs on CPU or GPU with zero changes. Whether the GPU is
*faster* is a separate, size-dependent question — small problems lose to
per-call overhead; see the GPU example in the gallery for measurements.
