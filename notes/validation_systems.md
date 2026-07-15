# Validation Systems - Reference Values for Lyapunov Exponent Tests

Companion to `notes/milestones.md`. Every test here is anchored to an
independent source - analytic derivation, a structural invariant of the
equations, or a published literature value - **not** to any number produced
by the code in `lyapunov-master/`. That existing code is useful only as a
historical reference for algorithm shape (variational equation + QR /
Gram-Schmidt renormalization); its numeric outputs are explicitly not to be
trusted as ground truth (per session guidance).

Tiers are ordered by how strong/independent the reference value is, and
roughly track which milestone first needs them.

---

## Tier 0 - Exact analytic values (no simulation ambiguity at all)

Use these first - if these fail, the bug is in the LE engine itself, not in
integration accuracy or chaos being hard to resolve.

### 0.1 Linear ODE - eigenvalues of the system matrix

For `ẋ = A x` with constant `A`, the Lyapunov spectrum is **exactly** the
real parts of the eigenvalues of `A`, for *any* initial condition (no
transient needed, no chaos, no QR renormalization even required in
principle - though the engine should still reproduce it via the normal QR
path).

- Pick `A` with distinct real eigenvalues, e.g. `A = diag(-1, -2, -5)` →
  expected spectrum `(-1, -2, -5)` exactly.
- Also test a case with a complex-conjugate pair, e.g. a damped 2D rotation
  `A = [[-0.1, 1], [-1, -0.1]]` → both real-part exponents equal `-0.1`
  (repeated), to check the engine handles near-degenerate exponents.
- **Tolerance:** should match to numerical integration error only (e.g.
  `1e-4`–`1e-6` depending on `dt`), since there's no chaos amplifying error.
- **Maps to:** M1.

### 0.2 Chaotic 1D maps with a closed-form Lyapunov exponent

Discrete maps avoid integration-scheme error entirely - best for isolating
bugs in the QR/renormalization bookkeeping itself.

- **Logistic map**, `x_{n+1} = r x_n (1 - x_n)`, at `r = 4`: exactly
  conjugate to the tent map / doubling map, giving
  **λ = ln 2 ≈ 0.6931472** (exact, provable via the conjugacy with the
  invariant measure `1/(π√(x(1-x)))`).
- **Tent map**, `x_{n+1} = 2x_n` if `x_n < 0.5` else `2(1-x_n)`: also
  **λ = ln 2** exactly (slope magnitude 2 everywhere).
- **Tolerance:** these should match to ~4-5 significant digits with a
  long-enough run (`N ~ 1e5`–`1e6` iterates); slow convergence is `O(1/√N)`
  since it's an ergodic average, not an integration-error issue.
- **Maps to:** M1/M2 - also the cheapest possible smoke test since there's
  no ODE integrator involved at all; consider implementing the map case
  first, before touching `lax.scan`-based ODE stepping.

### 0.3 Hénon map - sum-of-exponents invariant

`x_{n+1} = 1 - a x_n² + y_n`, `y_{n+1} = b x_n`, at the classical
`a = 1.4, b = 0.3`:

- The map's Jacobian determinant is constant: `det(J) = -b`. Therefore
  **λ1 + λ2 = ln|b| = ln(0.3) ≈ -1.203973** - exact, independent of any
  chaos-specific literature value.
- Individually, widely-cited numeric values (e.g. Hénon 1976, and standard
  references since) are approximately `λ1 ≈ 0.419`, `λ2 ≈ -1.623` (their
  sum ≈ `-1.204`, matching the exact invariant above).
- **Tolerance:** the *sum* should match `ln(0.3)` to near machine precision
  (it's an algebraic identity, not a statistical average); individual
  values within ~1% of the cited figures.
- **Maps to:** M1/M2 - good test for a 2-exponent (not just 1-exponent) map.

---

## Tier 1 - Structural invariants for continuous flows

These don't require trusting any external chaos-LE number; they follow from
the divergence theorem applied to the vector field, so they're a strong,
nearly-free correctness check on any new flow you validate against.

### 1.1 Lorenz system - constant-divergence check

`ẋ = σ(y-x)`, `ẏ = x(ρ-z) - y`, `ż = xy - βz`, classical parameters
`σ=10, ρ=28, β=8/3`:

- `trace(J) = -σ - 1 - β` is **constant** (state-independent) for this
  system, so:

  **λ1 + λ2 + λ3 = -(σ + 1 + β) = -(10 + 1 + 8/3) = -13.6̄ ≈ -13.6667**

  exactly, regardless of attractor geometry. This is the single strongest,
  cheapest check for a 3D chaotic flow: it doesn't depend on trusting a
  literature λ1 value at all.
- **Tolerance:** should match to the integration error, e.g. `1e-3`–`1e-4`
  for `dt = 1e-3`, over a long-enough averaging window past transient.
- **Maps to:** M1.

### 1.2 Rössler system - divergence check (state-dependent, needs a mean)

`ẋ = -y-z`, `ẏ = x+ay`, `ż = b + z(x-c)`, classical `a=b=0.2, c=5.7`:

- `trace(J) = a + (x - c)`, which is **not** constant (depends on the
  trajectory's mean `x`), so the check is:

  **λ1 + λ2 + λ3 = a - c + ⟨x⟩_t**

  where `⟨x⟩_t` is the time-average of `x` along the same trajectory used
  for the LE run. Compute both from one simulation and compare - still an
  independent check (no external LE number needed), just a slightly weaker
  one than Lorenz's because it needs the trajectory average as an
  intermediate quantity.
- **Maps to:** M1, as a secondary check alongside the Tier 2 literature
  value below.

---

## Tier 2 - Published literature values (continuous chaotic flows)

Use these as the primary "does this look like a real chaotic attractor"
check, with a tolerance band rather than digit-matching - published values
vary slightly (3rd–4th significant digit) across sources depending on
integration scheme, `dt`, and averaging horizon. Cross-check with an
independent tool if you want a second opinion beyond literature digits
(e.g. `nolds`/TISEAN on a simulated trajectory, or a small from-scratch
reference script - deliberately **not** anything from `lyapunov-master`).

- **Lorenz** (`σ=10, ρ=28, β=8/3`): commonly cited
  `λ ≈ (0.905, 0, -14.57)` (e.g. Wolf et al. 1985; rigorously validated
  numerics such as Viswanath 2004 give `λ1 ≈ 0.9056`). `λ2 ≈ 0` reflects
  the flow direction and should come out close to zero (within noise of the
  averaging, e.g. `< 0.01`) - a useful secondary sanity check since it's
  expected structurally for any autonomous continuous flow, not
  Lorenz-specific.
  - **Tolerance:** `λ1` within ~1-2% of `0.9056`; `|λ2| < 0.01`; sum matches
    Tier 1.1 exactly.
- **Rössler** (`a=b=0.2, c=5.7`): commonly cited `λ1 ≈ 0.07`
  (values reported in the 0.06–0.09 range depending on source/method,
  since Rössler's λ1 is much smaller than Lorenz's, making it more
  sensitive to averaging horizon). Use as a looser sanity check (positive,
  right order of magnitude) rather than a tight digit match; lean on the
  Tier 1.2 divergence identity as the tighter check for this system.
  - **Tolerance:** `λ1 ∈ [0.05, 0.09]`; `|λ2| < 0.01`.

Chapter reference: both systems (and the Lyapunov-exponent chapter's
treatment of the divergence/contraction identity used in Tier 1) appear in
Cvitanović et al., *ChaosBook* - cross-check the exact parameter set used
there before finalizing tolerances, since ChaosBook sometimes uses slightly
different Rössler constants than the "classical" `(0.2, 0.2, 5.7)` set.

---

## Tier 3 - Coupling/network correctness (exact, isolates the coupling code path)

### 3.1 Linear coupled network

Using a linear per-node model `ẋ_i = γ x_i + G Σ_j W_ij x_j` (i.e. the
`linear` `ModelSpec` already present in `vbi.simulator.models.linear`, used
here only as a reference equation, not as an imported dependency): the
whole system is `ẋ = A x` with `A = γI + G·W` (or `γI - G·L` if using a
graph Laplacian), a **constant** matrix.

- Exactly like Tier 0.1: **the full Lyapunov spectrum equals the real parts
  of the eigenvalues of `A`**, computable directly with `numpy.linalg.eigvals`
  independent of the LE engine entirely.
- Pick `γ < 0` (stable in isolation) and a small network (3-5 nodes) with a
  known weight matrix so `eigvals(A)` is easy to compute and check by hand.
- **Tolerance:** near machine/integration precision - this is testing
  wiring, not chaos.
- **Maps to:** M3. This is the test that catches "coupling term applied to
  the wrong axis," "Jacobian missing the off-diagonal coupling block," etc.,
  independent of anything related to chaotic dynamics.

---

## Tier 4 - DDE benchmarks

### 4.1 Mackey-Glass equation (primary chaotic DDE benchmark)

`ẋ(t) = β x(t-τ) / (1 + x(t-τ)^n) - γ x(t)`

Classic chaotic parameter set widely used in the nonlinear-dynamics and
time-series-prediction literature: `β=0.2, γ=0.1, n=10, τ=17`.

- This is the system Farmer (1982) originally used to introduce the
  discretized-map method for computing DDE Lyapunov spectra - i.e. exactly
  the method M4 implements via autodiff through the ring-buffer `step`. It
  remains the standard reference DDE for this kind of validation.
- Reported values: a small positive largest Lyapunov exponent (order
  `1e-2`–`1e-3` depending on normalization/units used), a near-zero second
  exponent, and a long tail of negative exponents (consistent with the
  system's infinite-dimensional phase space, truncated by the discretization
  horizon). Kaplan-Yorke dimension estimates in the literature for
  `τ=17` are commonly cited in the 2-3 range, increasing with `τ`
  (e.g. `τ=30` gives higher-dimensional chaos).
- **Tolerance:** treat this as a qualitative/order-of-magnitude check -
  positive `λ1`, `λ2` near zero, remaining spectrum negative, roughly the
  right Kaplan-Yorke dimension - not digit-matching. DDE LE values in the
  literature are noticeably less consistent across sources than the ODE
  cases above (different discretization horizons, different `τ` even within
  "the" chaotic Mackey-Glass parameter set), so pin down the exact source
  you're comparing against before setting a numeric tolerance in the test.
- **Maps to:** M4.

### 4.2 Linear scalar DDE (closer-to-analytic secondary check)

`ẋ(t) = -a x(t-τ)`:

- The characteristic equation `λ = -a e^{-λτ}` is transcendental but its
  dominant root (giving the top Lyapunov exponent for the linear case) can
  be solved numerically offline (e.g. via the Lambert W function:
  `λ = W(aτ)/τ` for suitable branch/sign conventions) as an independent,
  non-literature reference, analogous to Tier 0.1 but for a DDE.
- Use small `a·τ` first (non-oscillatory decay, easy to sanity check by
  eye) before pushing into the regime where this linear system oscillates
  or becomes marginally stable.
- **Maps to:** M4, as the "isolate the DDE tangent-propagation bug from
  Mackey-Glass's nonlinear chaos" check - prefer debugging failures here
  before debugging Tier 4.1.

### 4.3 Two-node delayed linear network (for M5)

A 2-node linear system with a single inter-node delay has a characteristic
equation that is still tractable to solve numerically (same transcendental
form as 4.2, generalized to a 2x2 system) - use as the delay-network analog
of Tier 3.1, i.e. an exact-ish reference that isolates "is the per-edge
delay wired correctly" from "is chaos resolved correctly."

---

## Cross-cutting test hygiene

- **Every** chaotic-flow test needs a documented transient-discard length
  and total integration horizon; report the running LE estimate
  (M2's convergence trace) alongside the final number in test output, not
  just a pass/fail, so a near-miss is diagnosable.
- Run each ODE test at two different `dt` values and confirm the LE
  estimate is stable (not `dt`-dependent) before trusting the literature
  comparison - this catches integration-scheme bugs that a single-`dt` test
  would miss.
- Run each DDE test at two different `dt` values (with `τ` held physically
  fixed) specifically to characterize the integer-step delay rounding error
  flagged as risk #4 in `notes/milestones.md`.
- Do **not** import or call anything from `lyapunov-master/` inside these
  tests, even for comparison - keep it out of the trust chain entirely per
  the session's guidance.
