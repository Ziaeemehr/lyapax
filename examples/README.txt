Examples
========

Runnable demos of lyapax's Lyapunov-exponent engine, in
`sphinx-gallery <https://sphinx-gallery.github.io/>`_ format (each file is
plain, directly-runnable Python -- sphinx-gallery is not wired up yet, see
notes/milestones.md, but this directory is ready to point a
``sphinx_gallery_conf`` at once docs exist).

Demos cover: inspecting a model's raw simulated time series as a sanity
check before computing exponents (single systems and coupled networks
alike), benchmark systems with a known answer (Tiers 0-2 in
notes/validation_systems.md), coupled networks (Tier 3), the extensibility
of custom coupling functions, a speed/accuracy characterization of the
current implementation, delayed (DDE) networks -- both a per-edge linear
delay sweep against a closed-form (Lambert W) reference and a delayed
Kuramoto network showing what transmission delay does to synchronization
(Tier 4-5) -- a matrix-free (jvp/vmap) tangent-propagation speedup on
a large network, and a jax.vmap parameter-sweep helper (one batched call
reproducing an earlier Python-loop G-sweep, faster and bit-for-bit
identical).

New capability -> new demo: as engine features land (a new coupling kind, a
new delay structure, a performance change, ...), add a runnable example
that exercises it, not just unit-test coverage -- see notes/milestones.md
for which milestone each file corresponds to.
