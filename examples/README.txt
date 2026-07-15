Examples
========

Runnable demos of lyapax's Lyapunov-exponent engine, in
`sphinx-gallery <https://sphinx-gallery.github.io/>`_ format (each file is
plain, directly-runnable Python, and also renders into the docs gallery --
see ``docs/``).

Demos cover: inspecting a model's raw simulated time series as a sanity
check before computing exponents (single systems and coupled networks
alike), benchmark systems with a known answer (Tiers 0-2 of
:doc:`/background/validation`), coupled networks (Tier 3), the extensibility
of custom coupling functions, a speed/accuracy characterization of the
current implementation (including RK4 vs RK6 convergence-order slopes),
delayed (DDE) networks -- both a per-edge linear
delay sweep against a closed-form (Lambert W) reference and a delayed
Kuramoto network showing what transmission delay does to synchronization
(Tier 4-5) -- a matrix-free (jvp/vmap) tangent-propagation speedup on
a large network, a jax.vmap parameter-sweep helper (one batched call
reproducing an earlier Python-loop G-sweep, faster and bit-for-bit
identical), the public front door (``ode_problem``, ``Network``/
``network_problem``, ``dde_problem``/``network_dde_problem``) that gives
plain, coupled, and delayed systems the same dynamics/network/coupling/
integrator/problem construction recipe, with ``state0``/``dt`` given once,
and grid-snapped vs. Hermite-interpolated DDE history reads
(``interpolate=True``) compared against a closed-form (Lambert W)
reference, showing why the latter converges smoothly as ``dt`` shrinks
and the former does not -- and a CPU-vs-GPU wall-time comparison across
growing network size, showing that ``lyapax``'s GPU support only pays off
once the per-step arithmetic is large enough to amortize a GPU's fixed
kernel-launch/transfer overhead, with the crossover point measured rather
than asserted -- and an adaptive-step ODE integrator
(``lyapax.adaptive.diffrax_adaptive_step``, backed by diffrax) that decouples
the renormalization sampling interval from the integrator's own internal
step size, with a tolerance-convergence sweep, a cross-check against
fixed-step rk4, and a demonstration that differentiating a Lyapunov exponent
through it requires ``jax.jacfwd`` rather than ``jax.grad`` -- and a
convergence diagnostic (``lyapax.core.convergence_drift``) that summarizes
how much a run's running exponent estimate moved over the tail of the run,
paired with ``result.checkpoint``/``lyapunov_spectrum(..., resume=...)``,
which continues a fixed-``n_steps`` run from where it left off instead of
restarting, letting a caller run in inspectable chunks and stop once
``convergence_drift`` says the estimate has settled -- and the same
run-inspect-resume loop for a DDE (``lyapax.dde.lyapunov_spectrum_dde(...,
resume=...)`` / ``lyapax.dde.DDECheckpoint``) on the Mackey-Glass chaotic
benchmark, showing that a DDE checkpoint must additionally carry the delay
ring buffer's state (not just trajectory state and tangent basis) for a
resumed run to continue correctly -- and differentiating a Lyapunov
exponent w.r.t. a system parameter with ``jax.grad``/``jax.jacfwd``,
demonstrating gradient-based tuning of a parameter toward a target
exponent on a non-chaotic system, then measuring how the same gradient
becomes numerically meaningless (grows by many orders of magnitude with
trajectory length rather than converging) once the underlying system is
genuinely chaotic.

New capability -> new demo: as engine features land (a new coupling kind, a
new delay structure, a performance change, ...), add a runnable example
that exercises it, not just unit-test coverage.
