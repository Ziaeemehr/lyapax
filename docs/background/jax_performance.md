# JAX performance notes

A short, practical guide to compiling and calling `lyapax` efficiently.
None of this changes results - it's about not paying more XLA
compilation cost than necessary.

## What's traced vs. what's Python

`lyapunov_spectrum`, `lyapunov_spectrum_dde`, and `sweep_lyapunov_spectrum`
are ordinary Python functions, not `jax.jit`-decorated ones. Internally
they use `n_steps`, `renorm_every`, `k`, and `t_transient` as plain Python
values to size `jax.lax.scan` lengths and pre-allocate the tangent matrix
(`(d, k)`) before entering traced code - none of that is itself run
through XLA tracing at the top level. `state0`, `dt`, `params`, and (for
DDEs) `buf0` are the values that flow into the traced/compiled inner
`jax.jvp`/`jax.vmap`/`jax.lax.scan` machinery.

## Wrapping a call in `jax.jit`

If you wrap `lyapunov_spectrum`/`lyapunov_spectrum_dde` in your own
`jax.jit` (e.g. to fuse it into a larger pipeline), mark the
shape/control-flow-determining arguments as static:

```python
import functools
import jax
from lyapax import lyapunov_spectrum

jitted = jax.jit(
    lyapunov_spectrum,
    static_argnames=["n_steps", "k", "renorm_every"],
)
```

`n_steps`, `k`, and `renorm_every` determine `jax.lax.scan` lengths and
array shapes (the `(d, k)` tangent matrix, the `n_steps // renorm_every`
loop trip count) - changing any of them triggers a full XLA recompile
under `jax.jit`, static or not. `t_transient` is used to compute a step
count too (`renorm_every * round(t_transient / dt / renorm_every)`), so
it is effectively shape-determining as well; treat it as static (or fixed)
in code paths you plan to call repeatedly.

In practice you often don't need to `jax.jit` the call yourself: the
`jax.jvp`/`jax.vmap`/`jax.lax.scan` calls inside `lyapunov_spectrum`
already get compiled by JAX on first use with a given set of static shapes
and are cached across subsequent calls with the same shapes - see
[07_speed_and_accuracy.py](../../examples/07_speed_and_accuracy.py) and
{doc}`benchmarks`, which report wall time *after* this warmup compilation.

## Compilation cost vs. runtime

The first call for a given `(d, k, n_steps, renorm_every, dt, integrator)`
combination pays an XLA tracing/compilation cost on top of the actual
integration; later calls with the same static shapes reuse the compiled
executable and only pay runtime. When benchmarking or reporting timings,
always report compile-time (first call) and steady-state runtime
separately, as {doc}`benchmarks` does - a single combined number
overstates per-call cost for any workload that calls `lyapunov_spectrum`
more than once with the same shapes, and understates it for a true
one-shot call.

## Avoiding accidental recompilation

- **Don't vary `n_steps`/`k`/`renorm_every` across a Python loop** if you
  can help it - each distinct value triggers a new compile. If you need
  to sweep a *value* (a parameter, an initial condition), sweep it as
  *data* instead: see `sweep_lyapunov_spectrum` and
  [11_vmap_parameter_sweep.py](../../examples/11_vmap_parameter_sweep.py),
  which batches a whole grid into one `jax.vmap`-compiled call rather than
  one Python-level call (and one compile) per grid point.
- **Reuse `dt`, `n_steps`, `k`, `renorm_every` across repeated calls**
  (e.g. inside an optimization loop or a notebook cell you re-run) so the
  compiled executable is reused instead of rebuilt.
- **Don't put `lyapunov_spectrum`/`lyapunov_spectrum_dde` calls inside a
  Python loop with changing static shapes** expecting `jax.jit` caching to
  save you - caching is keyed on the static shapes, and a different
  `n_steps` or `k` is a cache miss, not a fast path.
