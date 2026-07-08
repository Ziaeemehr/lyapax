# Statement of need/field

Why `lyapax` exists, and how it relates to the other tools that compute
Lyapunov spectra. This mirrors the "Statement of need" and "State of the
field" sections of the JOSS paper (`joss/paper.md`), kept here so the
motivation is visible from the documentation site as well.

## Statement of need

Computing a Lyapunov spectrum reliably requires more than the textbook
formula: one must integrate the state and a set of orthonormalized tangent
vectors together, re-orthonormalize periodically (QR), and average the
logarithmic growth of each direction over a long horizon in a numerically
stable way. Implementing this correctly for a new model — especially a
high-dimensional network or a delay system — is error-prone, and existing
tools each cover only part of the space.

`lyapax` fills a specific gap: a Lyapunov-spectrum engine that

1. works on any differentiable JAX map, so the model is ordinary
   Python/`jax.numpy` code rather than a symbolic expression;
2. is matrix-free, propagating only the `k` tangent directions actually
   requested via `jax.jvp` instead of forming a dense $d \times d$
   Jacobian, so partial spectra of high-dimensional systems are cheap;
3. treats parameter sweeps as a first-class, vectorized operation through
   `jax.vmap`; and
4. runs unchanged on CPU or GPU.

It is aimed at researchers in nonlinear dynamics, computational
neuroscience, and physics who need Lyapunov spectra of custom ODE, map, or
DDE models — particularly coupled oscillator networks and delay-coupled
systems — without implementing or hand-linearizing the tangent-propagation
machinery themselves.

## State of the field

- **[jitcode](https://pypi.org/project/jitcode/) and
  [jitcdde](https://pypi.org/project/jitcdde/)** compile a symbolic
  right-hand side to C and are the established Python references for ODE
  and DDE Lyapunov spectra respectively. They are fast and mature, but the
  model must be expressed as a symbolic expression rather than arbitrary
  code, and neither has a GPU backend.
- **[ChaosTools.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/chaostools/stable/)**,
  part of `DynamicalSystems.jl`, is a mature, broad-scope Julia toolbox for
  nonlinear dynamics, including Lyapunov spectra, but it lives outside the
  Python/NumPy ecosystem and is not designed around autodiff or GPU
  execution.
- **Ad hoc SciPy-based approaches** (custom scripts using finite
  differences, hand-derived Jacobians, or manually coded variational
  equations) are common in practice but are one-off, unvalidated, and not
  reusable across projects.
- **[Diffrax](https://github.com/patrick-kidger/diffrax)** provides
  differentiable ODE/SDE/CDE solvers natively in JAX, including
  adaptive-step and stiff methods, but it is a general
  differential-equation solver library, not a Lyapunov-spectrum package;
  it does not itself provide tangent-space QR renormalization,
  partial-spectrum tracking, or DDE ring-buffer handling.

`lyapax` is, to the author's knowledge, the first JAX-native package
purpose-built for Lyapunov spectra that accepts ordinary user-written JAX
functions (rather than symbolic expressions) and targets the
autodiff/JIT/GPU workflow directly, while remaining a narrowly scoped
complement to — not a replacement for — these tools. A concrete,
reproducible cross-check of `lyapax` against `jitcode`, `jitcdde`, and
`ChaosTools.jl` on shared benchmark systems is in {doc}`benchmarks`.
