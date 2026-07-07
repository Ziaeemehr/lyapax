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

# Statement of Need

Computing a Lyapunov spectrum reliably requires more than the textbook
formula: one must integrate the state and a set of orthonormalized tangent
vectors together, re-orthonormalize periodically (QR), and average the
logarithmic growth of each direction over a long horizon in a numerically
stable way. Implementing this correctly for a new model — especially a
high-dimensional network or a delay system — is error-prone, and existing
tools each cover only part of the space. `jitcode` and `jitcdde`
[@ansmann2018efficiently] compile a symbolic right-hand side to C and are
the established Python references for ODE and DDE Lyapunov spectra
respectively, but they require the model to be expressed symbolically and
have no GPU backend. `ChaosTools.jl`, part of `DynamicalSystems.jl`
[@datseris2018dynamicalsystems], is a mature Julia toolbox but lives
outside the Python/NumPy ecosystem.

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

# Method and Implementation

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

# Acknowledgements

This project has received funding from the European Union's Horizon Europe
Programme and from a government grant managed by the Agence Nationale de la
Recherche (France 2030 program). The funders had no role in the design of
the software or the preparation of the manuscript.

# References
