Examples
========

Runnable demos of lyapax's Lyapunov-exponent engine, in
`sphinx-gallery <https://sphinx-gallery.github.io/>`_ format (each file is
plain, directly-runnable Python, and also renders into the docs gallery --
see ``docs/``).

Demos cover: inspecting a model's raw simulated time series as a sanity
check before computing exponents (single systems and coupled networks
alike), benchmark systems with a known answer (Tiers 0-2 in
notes/validation_systems.md), coupled networks (Tier 3), the extensibility
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
than asserted.

New capability -> new demo: as engine features land (a new coupling kind, a
new delay structure, a performance change, ...), add a runnable example
that exercises it, not just unit-test coverage.
