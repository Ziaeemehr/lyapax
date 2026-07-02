# Vendored code notice

The files in this directory (`model_spec.py`, `coupling.py`, `step.py`) are
adapted, in trimmed form, from:

- `vbi/simulator/spec/model.py`
- `vbi/simulator/spec/coupling.py`
- `vbi/simulator/spec/connectivity.py`
- `vbi/simulator/backend/jax_/codegen.py`
- `vbi/simulator/backend/jax_/simulator.py`

from the `vbi` project (https://github.com/ins-amu/vbi), licensed under the
Apache License 2.0. Copyright the `vbi` authors (Abolfazl Ziaeemehr, Meysam
Hashemi, Marmaduke Woodman).

Changes made relative to the originals, per the decision recorded in
`notes/milestones.md` (M0 — vendor a trimmed copy rather than depend on
`vbi` at import time):

- Dropped monitors, stimuli, the Balloon-Windkessel/BOLD path, the
  parameter sweeper, and the `sigmoidal` / `kuramoto` coupling kinds —
  out of scope for `lyapax` v1.
- Dropped stochastic noise injection and state-bound clipping from the
  step function — both are non-smooth or not meaningfully compatible with
  a deterministic Lyapunov spectrum (see "risk #3" in
  `notes/milestones.md`).
- Dropped presentation-only `ModelSpec` methods (`describe`, `_repr_html_`,
  LaTeX rendering) — not needed for the numerical engine.
- Dropped `Connectivity.from_tvb` / `.from_file` / `.save` / `.load` and
  `ModelSpec`/`CouplingSpec` YAML/dict loading — `lyapax` builds these
  objects directly in Python; no config-file layer in v1.
- Did **not** vendor `CouplingSpec` at all (nor the `kind`-dispatched
  `build_coupling` factories in `vbi`'s numpy/JAX backends). Those dispatch
  on a fixed string enum via hardcoded `if/elif`, which means a user can't
  add a new coupling without patching `vbi`'s source. `lyapax.coupling`
  instead makes coupling a plain callable — see that module and the M3
  note in `notes/milestones.md`.
- Everything else (dfun codegen via `exec()`, the ring-buffer delayed
  coupling with a flat-index gather, the `step(carry, _)` shape used with
  `lax.scan`) is functionally the same as the `vbi` originals.
