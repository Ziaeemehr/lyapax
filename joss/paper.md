---
title: 'lyapax: JAX-native Lyapunov exponent computation for ODEs and DDEs'
date: 7 July 2026
tags:
  - Python
  - JAX
  - dynamical systems
  - Lyapunov exponents
  - chaos
  - delay differential equations
authors:
  - name: Abolfazl Ziaeemehr
    orcid: 0000-0002-4696-9947
    affiliation: 1
    corresponding: true
affiliations:
  - name: Aix Marseille Univ, INSERM, INS, Institut de Neurosciences des Systèmes, Marseille, France
    index: 1
bibliography: paper.bib
---

# Summary

Lyapunov exponents quantify the average exponential rate at which nearby
trajectories of a dynamical system separate or converge, and are the
standard tool for detecting and characterizing chaos: a positive largest
exponent is the signature of sensitive dependence on initial conditions,
while the full spectrum yields the Kaplan–Yorke (Lyapunov) dimension of the
attractor. `lyapax` is a Python package that computes Lyapunov spectra of
ordinary differential equations (ODEs), iterated maps, and delay
differential equations (DDEs) using the Benettin/QR method
[@benettin1980lyapunov; @shimada1979numerical], built natively on JAX
[@jax2018github]. Rather than requiring the user to supply analytic
Jacobians or variational equations, `lyapax` propagates tangent vectors
through the same numerical step function the trajectory uses, via JAX
forward-mode automatic differentiation (`jax.jvp`) batched with
`jax.vmap`. This makes it possible to compute exponents for arbitrary
user-written right-hand sides — including coupled networks and
delay-coupled systems — with no hand-derived linearization, while
inheriting JAX's just-in-time compilation and transparent CPU/GPU
execution.

# Statement of need

Computing a Lyapunov spectrum reliably requires more than the textbook
formula: one must integrate the state and a set of orthonormalized tangent
vectors together, re-orthonormalize periodically (QR), and average the
logarithmic growth of each direction over a long horizon in a numerically
stable way. Implementing this correctly for a new model — especially a
high-dimensional network or a delay system — is error-prone, and existing
tools each cover only part of the space.

`lyapax` fills a specific gap: a Lyapunov-spectrum engine that (i) works on
any differentiable JAX map, so the model is ordinary Python/`jax.numpy`
code rather than a symbolic expression; (ii) is matrix-free, propagating
only the `k` tangent directions actually requested via `jax.jvp` instead of
forming a dense `d × d` Jacobian, so partial spectra of high-dimensional
systems are cheap; (iii) treats parameter sweeps as a first-class,
vectorized operation through `jax.vmap`; and (iv) runs unchanged on CPU or
GPU. It is aimed at researchers in nonlinear dynamics, computational
neuroscience, and physics who need Lyapunov spectra of custom ODE, map, or
DDE models — particularly coupled oscillator networks and delay-coupled
systems — without implementing or hand-linearizing the tangent-propagation
machinery themselves.

# State of the field

`jitcode` and `jitcdde` [@ansmann2018efficiently] compile a symbolic
right-hand side to C and are the established Python references for ODE and
DDE Lyapunov spectra respectively; they are fast and mature, but the model
must be expressed as a symbolic expression rather than arbitrary code, and
neither has a GPU backend. `ChaosTools.jl`, part of `DynamicalSystems.jl`
[@datseris2018dynamicalsystems], is a mature, broad-scope Julia toolbox for
nonlinear dynamics, including Lyapunov spectra, but it lives outside the
Python/NumPy ecosystem and is not designed around autodiff or GPU
execution. Ad hoc SciPy-based approaches (custom scripts using finite
differences, hand-derived Jacobians, or manually coded variational
equations) are common in practice but are one-off, unvalidated, and not
reusable across projects. `Diffrax` [@kidger2021neural] provides
differentiable ODE/SDE/CDE solvers natively in JAX, including adaptive-step
and stiff methods, but it is a general differential-equation solver
library, not a Lyapunov-spectrum package; it does not itself provide
tangent-space QR renormalization, partial-spectrum tracking, or DDE
ring-buffer handling. `lyapax` is, to the author's knowledge, the first
JAX-native package purpose-built for Lyapunov spectra that accepts
ordinary user-written JAX functions (rather than symbolic expressions) and
targets the autodiff/JIT/GPU workflow directly, while remaining a narrowly
scoped complement to — not a replacement for — these tools.

# Software design

For an autonomous system $\dot{x} = f(x)$, `lyapax` advances the state with
a fixed-step integrator (Euler, Heun, RK4, or RK6) and simultaneously
propagates a $(d, k)$ matrix $Q$ of tangent vectors through the *same* step
map. The tangent update is obtained by forward-mode automatic
differentiation of the step function — one `jax.jvp` per tracked column,
batched over columns with `jax.vmap` — so no analytic Jacobian is needed
and only the requested $k$ directions are ever propagated. Every
`renorm_every` steps the tangent matrix is QR-decomposed; the logarithms of
the diagonal of $R$ are accumulated, and the $i$-th Lyapunov exponent is
their sum divided by the elapsed time (Benettin's method). The cost scales
with $k$, not $d$, so a few leading exponents of a large network are
inexpensive to obtain.

The DDE engine (`lyapunov_spectrum_dde`) generalizes this to fixed-delay
systems by differentiating through an augmented `(state, ring_buffer)`
carry, following the discretized-map approach of @farmer1982chaotic. Delays
are resolved to an integer number of steps; a helper reports the effective
delay actually used so it can be converged toward the physical delay by
reducing `dt`. The package exposes problem-object constructors for plain
ODEs (`ode_problem`), coupled networks (`network_problem`) with pluggable
coupling functions, and delay and delay-network systems (`dde_problem`,
`network_dde_problem`), together with a small library of reference systems
(Lorenz, Rössler, logistic and Hénon maps, Kuramoto networks,
Mackey–Glass).

Because Lyapunov exponents are averages of log-growth rates accumulated
over many steps, single-precision arithmetic silently degrades
long-horizon estimates; `lyapax` therefore expects JAX's `float64` mode to
be enabled and warns when it is not.

Several design tradeoffs shape the current scope. **Fixed-step integrators
versus adaptive/stiff solvers:** `lyapax` uses fixed-step Euler/Heun/RK4/RK6
rather than an adaptive or implicit solver like those in `Diffrax`
[@kidger2021neural]; this keeps the tangent-propagation step map simple and
directly differentiable, at the cost of requiring the user to check
`dt`-convergence themselves and making stiff systems a poor fit. **Matrix-free
`jax.jvp`/`jax.vmap` versus a dense Jacobian:** propagating only the
requested $k$ tangent columns avoids ever materializing a $d \times d$
Jacobian, which is what makes partial spectra of high-dimensional systems
cheap, but it means the cost still scales linearly in $k$ and is not free
for full-spectrum ($k = d$) requests on very large systems. **Partial
spectra as a first-class option:** exposing `k` directly in the public API,
rather than always computing the full spectrum and truncating, is what
makes the matrix-free design pay off in practice. **DDE ring-buffer
dimension:** representing delay history as a fixed-size ring buffer appended
to the state keeps the DDE solver structurally identical to the ODE one
(same `jax.jvp`/`jax.vmap` machinery, same QR renormalization), but the
augmented state's dimension — and hence memory and compute cost — grows
with the delay-to-`dt` ratio, which is why very small `dt` relative to a long
delay `tau` can become expensive. **Plain callables over a coupling
registry:** networks and DDEs take an arbitrary user-supplied coupling
function rather than requiring registration in a fixed set of built-in
coupling types; a handful of common couplings (linear, sigmoidal, Kuramoto)
are provided as convenience defaults, but the API does not otherwise
constrain what a coupling function can compute.

# Validation

`lyapax` is validated against exact and published references and
cross-checked against independent tools. On systems with closed-form
answers it recovers them to near machine precision — the eigenvalues of
linear ODEs and of a coupled linear network, and $\ln 2$ for the logistic
and tent maps. On canonical chaotic systems it matches published values:
$\lambda_1 \approx 0.902$ for the Lorenz attractor (literature $\approx
0.9056$) with the exponent sum reproducing the exact $-(\sigma + 1 +
\beta)$, and $\lambda_1 \approx 0.07$ for the Rössler system. For delay
systems it recovers the Lambert-$W$ root of a linear scalar DDE and places
the Mackey–Glass largest exponent and Kaplan–Yorke dimension in their
established ranges.

These results agree with `jitcode`, `jitcdde`, and `ChaosTools.jl`
run on the same systems from independently transcribed equations, to within
the finite-time and finite-precision noise those tools show against each
other and against the literature. The matrix-free design also yields
practical speedups on the workloads it targets: computing five leading
exponents of a 200-node Kuramoto network via `jax.jvp`/`jax.vmap` is about
23× faster than forming a dense Jacobian with `jax.jacfwd`, and a
vectorized `jax.vmap` parameter sweep is about 3× faster than the
equivalent Python loop. The full accuracy and performance comparison,
including the exact settings and environment, is reproducible via the
scripts under `benchmarks/`.

# Research impact statement

*Placeholder — to be filled in before submission.* `lyapax` is newly
released with no external users, citations, or downstream projects yet.
The evidence of applicability currently available is internal: the
validation suite against exact/literature values and independent tools
described above, the runnable example gallery, and the benchmark reports
under `benchmarks/`. This section will be updated with concrete usage
evidence (adopting projects, presentations, or publications) as it
accumulates and before the package is submitted.

# AI usage disclosure

*Placeholder — to be filled in before submission.* This section will
describe how AI tools were used in developing `lyapax` (code, tests,
documentation, and/or this paper) and how correctness of any AI-assisted
contributions was checked, once the project is closer to submission.

# Limitations and future work

The current scope excludes adaptive/stiff ODE integration, state-dependent
or distributed delays, and stochastic/noise-driven Lyapunov exponents;
DDE problems with large delay-to-`dt` ratios are also limited by the
ring-buffer state's memory and compute cost. Planned directions include an
optional `Diffrax`-based [@kidger2021neural] adaptive-step integrator for
non-stiff-but-inconvenient systems, and continued work on DDE scalability
for long-delay problems.

# Acknowledgements

This project has received funding from the European Union's Horizon Europe
Programme and from a government grant managed by the Agence Nationale de la
Recherche (France 2030 program). The funders had no role in the design of
the software or the preparation of the manuscript.

# References
