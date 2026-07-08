lyapax
======

JAX-native Lyapunov exponent computation for ODEs and DDEs, via the
Benettin/QR method with ``jax.jvp``/``jax.vmap`` tangent propagation.

Installation
------------

.. code-block:: bash

   pip install lyapax

For development (running the test suite or examples), install from a clone
instead:

.. code-block:: bash

   pip install -e ".[dev]"      # core + pytest/scipy for the test suite
   pip install -e ".[examples]" # + matplotlib, to run examples/
   pip install -e ".[docs]"     # + sphinx/sphinx-gallery, to build docs/

Requires ``jax>=0.10``, Python ``>=3.11``.

.. toctree::
   :maxdepth: 1
   :caption: Background

   background/lyapunov_exponents

.. toctree::
   :maxdepth: 1

   background/lyapax_implementation
   background/validation
   background/capabilities
   background/jax_performance
   background/benchmarks
   auto_examples/index
   api
