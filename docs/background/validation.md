# Validation: how lyapax's results are checked

There is no ground-truth oracle for the Lyapunov spectrum of a generic
chaotic system, so every correctness claim in lyapax's test suite is
anchored to an *independent* source: an exact analytic result, a
structural invariant of the equations, or a published literature value.
No number produced by another from-scratch implementation is ever used
as a reference. This page lists the validation tiers, ordered by how
strong and independent the reference value is; each tier is also
demonstrated by an example in the gallery.

The structural invariants used below (sum rule, zero exponent along the
flow, symmetry-pinned zero exponents) are explained in
{doc}`lyapunov_exponents`.

(validation-tier-0)=
## Tier 0 — Exact analytic values

If these fail, the bug is in the Lyapunov engine itself, not in
integration accuracy or in chaos being hard to resolve.

(validation-tier-0-1)=
### 0.1 Linear ODE: eigenvalues of the system matrix

For $\dot{x} = Ax$ with constant $A$, the Lyapunov spectrum is exactly
the real parts of the eigenvalues of $A$, for any initial condition — no
transient, no chaos. The tests use matrices with distinct real
eigenvalues (e.g. $A = \operatorname{diag}(-1, -2, -5)$, spectrum
exactly $(-1, -2, -5)$) and a damped 2-D rotation with a
complex-conjugate pair (both exponents equal the shared real part),
which exercises near-degenerate exponents. Expected agreement is at the
level of the integration error alone. Demonstrated in
{ref}`sphx_glr_auto_examples_01_linear_ode.py`.

(validation-tier-0-2)=
### 0.2 Chaotic 1-D maps with closed-form exponents

Discrete maps avoid integration error entirely, isolating the
QR/renormalization bookkeeping:

- **Logistic map** at $r = 4$: exactly conjugate to the tent map, giving
  $\lambda = \ln 2 \approx 0.6931$ exactly [[1]](#references).
- **Tent map**: also $\lambda = \ln 2$ (slope magnitude 2 everywhere).

Convergence is the $O(1/\sqrt{N})$ rate of an ergodic average, so long
runs match to 4–5 significant digits. Demonstrated in
{ref}`sphx_glr_auto_examples_02_chaotic_maps.py`.

(validation-tier-0-3)=
### 0.3 Hénon map: sum-of-exponents invariant

For the Hénon map [[2]](#references) at the classical $a = 1.4$,
$b = 0.3$, the Jacobian determinant is constant, $\det J = -b$, so
$\lambda_1 + \lambda_2 = \ln|b| = \ln 0.3 \approx -1.20397$ — an
algebraic identity, checkable to near machine precision. The individual
values are additionally compared to widely cited figures
($\lambda_1 \approx 0.419$, $\lambda_2 \approx -1.623$
[[1]](#references)) within ~1%.

(validation-tier-1)=
## Tier 1 — Structural invariants of continuous flows

These follow from the divergence of the vector field and require no
trusted external exponent value.

- **Lorenz** ($\sigma = 10$, $\rho = 28$, $\beta = 8/3$):
  $\operatorname{tr} J = -\sigma - 1 - \beta$ is constant, so
  $\sum_i \lambda_i = -(10 + 1 + 8/3) \approx -13.6667$ exactly,
  regardless of attractor geometry.
- **Rössler** ($a = b = 0.2$, $c = 5.7$):
  $\operatorname{tr} J = a + (x - c)$ depends on the trajectory, so the
  check becomes $\sum_i \lambda_i = a - c + \langle x \rangle_t$, with
  the time average $\langle x \rangle_t$ computed from the same run.

(validation-tier-2)=
## Tier 2 — Published literature values

Used as a tolerance-band check (published values vary in the 3rd–4th
significant digit across sources, depending on integrator, `dt`, and
averaging horizon):

- **Lorenz**: $\lambda \approx (0.906,\, 0,\, -14.57)$
  [[1]](#references), [[3]](#references); $\lambda_2 \approx 0$ is the
  flow-direction zero expected structurally for any autonomous flow.
- **Rössler**: $\lambda_1 \approx 0.07$, with reported values in the
  0.06–0.09 range across sources [[1]](#references) — used as a looser
  order-of-magnitude check, with the Tier 1 divergence identity as the
  tighter one.

Both flows are demonstrated in
{ref}`sphx_glr_auto_examples_03_chaotic_flows.py`.

(validation-tier-3)=
## Tier 3 — Coupled-network wiring

A linear per-node model with linear coupling makes the whole network a
constant-matrix linear system, $\dot{x} = (\gamma I + G W)x$, so as in
Tier 0.1 the full spectrum equals the real parts of
$\operatorname{eig}(\gamma I + G W)$ — computable independently with
`numpy.linalg.eigvals`. This isolates coupling-path bugs ("coupling
applied to the wrong axis", "Jacobian missing the off-diagonal coupling
block") from anything related to chaos. Demonstrated in
{ref}`sphx_glr_auto_examples_04_linear_network.py`.

(validation-tier-4)=
## Tier 4 — Delay systems

- **Linear scalar DDE**, $\dot{x}(t) = -a\,x(t - \tau)$: the dominant
  root of the characteristic equation $\lambda = -a e^{-\lambda\tau}$
  is available through the Lambert W function [[4]](#references),
  giving an independent, near-analytic reference for the leading
  exponent of a genuine delay system.
- **Mackey–Glass** [[5]](#references) at the classic chaotic parameter
  set ($\beta = 0.2$, $\gamma = 0.1$, $n = 10$, $\tau = 17$): the system
  Farmer used to introduce the discretized-history method for DDE
  Lyapunov spectra [[6]](#references) — exactly the method lyapax
  implements. Checked qualitatively: a small positive leading exponent,
  a near-zero second exponent, and a negative tail, consistent with
  literature reports.

## Cross-cutting test hygiene

- Every chaotic-flow test documents its transient-discard length and
  averaging horizon, and reports the running estimate
  (`LyapunovResult.history`) alongside the final number, so a near-miss
  is diagnosable.
- ODE tests are run at two different `dt` values to confirm the
  estimate is `dt`-stable before comparing against literature — this
  catches integration-scheme bugs a single-`dt` test would miss.
- DDE tests are likewise run at two `dt` values with the physical
  $\tau$ held fixed, specifically to characterize the grid-snapping
  delay-rounding error (see {doc}`lyapax_implementation`).

## References

1. J. C. Sprott, *Chaos and Time-Series Analysis*, Oxford University
   Press, 2003. Appendix A tabulates Lyapunov spectra for the common
   benchmark systems used here.
2. M. Hénon, *A two-dimensional mapping with a strange attractor*,
   Communications in Mathematical Physics **50** (1976) 69–77.
   [doi:10.1007/BF01608556](https://doi.org/10.1007/BF01608556)
3. A. Wolf, J. B. Swift, H. L. Swinney, and J. A. Vastano, *Determining
   Lyapunov exponents from a time series*, Physica D **16** (1985)
   285–317.
   [doi:10.1016/0167-2789(85)90011-9](https://doi.org/10.1016/0167-2789%2885%2990011-9)
4. R. M. Corless, G. H. Gonnet, D. E. G. Hare, D. J. Jeffrey, and D. E.
   Knuth, *On the Lambert W function*, Advances in Computational
   Mathematics **5** (1996) 329–359.
   [doi:10.1007/BF02124750](https://doi.org/10.1007/BF02124750)
5. M. C. Mackey and L. Glass, *Oscillation and chaos in physiological
   control systems*, Science **197** (1977) 287–289.
   [doi:10.1126/science.267326](https://doi.org/10.1126/science.267326)
6. J. D. Farmer, *Chaotic attractors of an infinite-dimensional
   dynamical system*, Physica D **4** (1982) 366–393.
   [doi:10.1016/0167-2789(82)90042-2](https://doi.org/10.1016/0167-2789%2882%2990042-2)
