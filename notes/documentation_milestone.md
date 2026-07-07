# M9 — Sphinx + Sphinx-Gallery Documentation

## Status

Planning only. This note lays out the content and the cleanup work for a
real Sphinx site; it does not itself add any `.rst`/`.md` pages under
`docs/`. Narrative pages (theory, user guide, etc.) are being written
separately and should treat the "Background" section below as source
material to adapt, not as a page to copy in verbatim.

Continues the doc work `notes/milestones.md` explicitly deferred at M6
("polished README and notebook conversion are documentation tasks, not
functionality, and can wait until the API is otherwise stable") and picks
up after M8's package-review follow-ups.

## What already exists

`docs/` is already a working Sphinx + `sphinx-gallery` scaffold, not a
blank slate:

- `docs/conf.py` — `sphinx.ext.autodoc`, `sphinx.ext.napoleon`,
  `sphinx.ext.viewcode`, `sphinx_gallery.gen_gallery`; theme is `furo`.
  `sphinx_gallery_conf` points `examples_dirs` at `../examples` and
  `gallery_dirs` at `auto_examples`, picking up every `plot_*.py`.
- `docs/index.rst` — toctree of `auto_examples/index` + `api`.
- `docs/api.rst` — `automodule` blocks for `core`, `dde`, `simulator.*`,
  `integrators`, `coupling`, `network`, `systems`, `sweep`, `utils`.
- `pyproject.toml`'s `[project.optional-dependencies].docs` already lists
  `sphinx`, `sphinx-gallery`, `furo`.
- A previous `sphinx-build` run left `docs/auto_examples/*` and
  `docs/sg_execution_times.rst` checked in. **These are stale** — they
  were generated before the front-door API migration (`ode_problem`,
  `network_problem`, `network_dde_problem`, `Network`; see
  `notes/api_design_review.md` and the examples-migration work in this
  session) and before `plot_12`–`plot_14` existed. Re-run `sphinx-build`
  once the docstring cleanup below lands; don't hand-edit the generated
  `auto_examples/` output.

So the immediate gap is not "set up Sphinx" — it's (1) narrative content
(background/theory, user guide, capability/limitation pages) and (2)
purging internal-dev-note cross-references from anything Sphinx will
publish (docstrings, example modules), per the policy below.

## Cross-reference policy: no `notes/*.md` links from published content

**Rule going forward:** docstrings (anything `autodoc` pulls in) and
`examples/plot_*.py` modules (anything `sphinx-gallery` renders) must
never point at `notes/*.md`. Those files are this repo's internal
design/review history — written for whoever is implementing the next
milestone, assuming access to the git working tree, not for a reader of
the published docs. If a docstring or example currently says "see
`notes/foo.md`", the fix is one of:

1. **Promote it.** The rationale is genuinely useful to an end user (why
   x64 matters, why DDE delays round to the nearest `dt`, what the
   validation tiers mean) → write it into a proper Sphinx background page
   and link `:doc:`/`:ref:` to that instead.
2. **Drop it.** The rationale is pure development history (which
   milestone added a function, an internal bug that got fixed, an
   abandoned design) → delete the pointer. That narrative belongs in git
   history / `notes/milestones.md`, not in a published docstring.
3. **Inline it, unlinked.** A short, self-contained clarification (e.g.
   "ring buffer slot `k % horizon` holds step `k`'s post-write state") can
   just stay as prose in the docstring with the `notes/...md` pointer
   removed — no doc page needed if there's nothing more to say.

Below is every current `notes/*.md` reference in code/examples (36
occurrences across 15 files, as of this milestone), bucketed by which of
the three fixes it needs. This is the checklist for the actual cleanup
pass (not done in this note).

### Bucket 1 — promote to a Background/Validation doc page

| Location | Points at | Promote to |
|---|---|---|
| `src/lyapax/core.py:42,54` | `milestones.md` risk #1 (float32 underflow) | Background: "Precision requirements" |
| `src/lyapax/core.py:204` | `milestones.md` risk #2 (`renorm_every` overflow) | Background: "Choosing `renorm_every`" |
| `src/lyapax/dde.py:28` | `stepping_accuracy_review.md` (Hermite interpolation cost/tradeoff) | Background: "Grid-snapped vs. interpolated DDE history" |
| `src/lyapax/integrators.py:8` | `validation_systems.md` Tier 2 | Background/Validation: tier table |
| `src/lyapax/integrators.py:54,70` | `stepping_accuracy_review.md` (per-stage history reads, order restoration) | Background: "DDE + Runge-Kutta stage order" |
| `src/lyapax/simulator/coupling.py:60` | `stepping_accuracy_review.md` | same as above |
| `src/lyapax/simulator/step.py:161,196,206,300,337,374,387` | `stepping_accuracy_review.md` (interpolation flag, per-stage coupling, freezing error) | same as above |
| `src/lyapax/systems.py:7` | `validation_systems.md` tiers | Background/Validation: tier table |
| `examples/plot_01..04` docstrings | `validation_systems.md` Tier 0.1/0.2-0.3/1-2/3.1 | link each demo at the matching row of the same tier table |

### Bucket 2 — drop (pure development history, no doc replacement)

| Location | Points at |
|---|---|
| `src/lyapax/core.py:7,18,266` | `milestones.md` (M1/M3/M4/M6 narrative) |
| `src/lyapax/coupling.py:3` | `milestones.md` M3 design note |
| `src/lyapax/dde.py:36,83,154` | `milestones.md` M4 narrative |
| `src/lyapax/__init__.py:5,6` | `milestones.md` + `validation_systems.md` (module docstring "see design history") |
| `src/lyapax/network.py:13` | `milestones.md` M5 cleanup note |
| `src/lyapax/sweep.py:5` | `milestones.md` M0 note |
| `src/lyapax/simulator/coupling.py:7` | `milestones.md` |
| `src/lyapax/simulator/step.py:318` | `milestones.md` M4 |
| `examples/plot_09_kuramoto_delayed_network.py:43` | `api_design_review.md` (now redundant — the front-door functions' own docstrings explain the "why") |

### Bucket 3 — inline, unlinked (internal detail, keep the one sentence, drop the pointer)

| Location | Points at |
|---|---|
| `src/lyapax/simulator/step.py:80-81` | `possible_solution_to_open_issues.md` + `open_issues.md` item 2 (ring-buffer slot convention) |

(`docs/api.rst`'s `automodule` directives will pull every one of these
docstrings in verbatim once built, so Bucket 2/3 items left unfixed would
otherwise ship internal-issue-tracker links straight into the public API
reference.)

## Background: how Lyapunov exponents are computed, and what lyapax does

This section is the actual "useful introduction" — write-once source
material for whatever Background/Theory page ends up in `docs/`.

### What a Lyapunov exponent is

For a dynamical system `dx/dt = f(x)` (or a discrete map `x_{n+1} =
F(x_n)`), the Lyapunov exponents measure the average exponential
rate at which infinitesimally close trajectories separate (positive
exponent) or converge (negative exponent) along each of `d` independent
directions in state space, `d` = state dimension. Formally, linearize
around a trajectory `x(t)`: an infinitesimal perturbation `delta_x(t)`
evolves under the variational (tangent) equation

```
d(delta_x)/dt = J(x(t)) delta_x,      J = Jacobian of f at x(t)
```

and the `i`-th Lyapunov exponent is the long-time average growth rate of
the `i`-th ordered singular direction of the tangent flow,

```
lambda_i = lim_{T->inf} (1/T) log( sigma_i(T) )
```

where `sigma_i(T)` are the singular values of the linearized flow map
over `[0, T]`. In practice nobody takes that limit or that SVD directly:
a single generic perturbation collapses onto the fastest-growing
direction almost immediately (everything overflows/underflows in finite
precision), so every practical algorithm periodically re-orthonormalizes
a whole basis of perturbation vectors instead of tracking one.

A few structural facts make Lyapunov spectra checkable, independent of
whether the underlying system is chaotic:

- **Sign meaning.** At least one positive exponent is the standard
  operational definition of chaos (sensitive dependence on initial
  conditions). All-negative spectra mean the trajectory converges to a
  fixed point or a stable limit cycle. A zero exponent along the flow
  direction is generic for any continuous-time attractor (perturbing
  along the trajectory itself neither grows nor shrinks).
- **Sum invariant.** `sum(lambda_i) = lim (1/T) integral trace(J(x(t))) dt`
  — for systems where `trace(J)` is constant (e.g. Lorenz), this is known
  exactly with no simulation at all; for others (e.g. Rössler) it reduces
  to a time-average of one state variable, checkable from an independent
  trajectory. `notes/validation_systems.md` (source of Bucket-1's tier
  table above) builds lyapax's whole test suite around checks like this,
  plus closed-form/published values, precisely because there is no
  ground-truth oracle to compare against otherwise.
- **Continuous symmetry -> exact zero exponent.** If the dynamics are
  invariant under a continuous transformation (e.g. Kuramoto phases under
  a global rotation `theta_i -> theta_i + c`), the generator of that
  symmetry is an exactly-marginal direction: one exponent is pinned to
  `0`, never negative, regardless of parameters — a model-specific but
  very sharp correctness check (used throughout the Kuramoto examples).

### The Benettin/QR method (what lyapax implements)

lyapax uses the standard variational/tangent-space approach (Benettin et
al. 1980), generalized with QR renormalization for the *full spectrum*,
not just the leading exponent:

1. Propagate the state `x` forward one step at a time under the chosen
   fixed-step map (Euler / Heun / RK4 / RK6).
2. Alongside it, propagate a `(d, k)` matrix `Y` of tangent vectors under
   the same step's linearization — `k <= d` is how many leading exponents
   are tracked; `k = d` gives the full spectrum.
3. Every `renorm_every` steps, QR-decompose `Y = Q R`. `Q` (orthonormal)
   replaces `Y` for the next stretch; `log|diag(R)|` for that stretch is
   accumulated per column.
4. Dividing the running sum by elapsed time gives each column's Lyapunov
   exponent estimate, converging as the initial-condition-dependent
   transient washes out (discarded via `t_transient`) and the running
   average smooths over the trajectory's fluctuations.

This is the direct JAX-native analogue of the classical Wolf/Sandri
algorithm, but note lyapax's method note in `lyapax/core.py`: unlike
older from-scratch implementations that inspired the design (see the
project history), the numeric results were never reused as ground truth
— every correctness claim is anchored independently, to analytic results,
structural invariants, or published literature values.

### What's specific to lyapax's implementation

- **JAX-native, matrix-free tangent propagation.** Instead of computing a
  dense `d x d` Jacobian (`jax.jacfwd`) and multiplying by `Y`, lyapax
  propagates each of the `k` tangent columns via one `jax.jvp`
  (forward-mode directional derivative) call, batched together with
  `jax.vmap`. Cost is `O(k)` per raw step, not `O(d)` — the entire benefit
  of tracking a *partial* spectrum (`k < d`) is otherwise lost if a dense
  Jacobian is computed anyway and only afterwards projected down to `k`
  columns.
- **Any fixed-time-step map, integrator-agnostic.** The Lyapunov engine
  itself (`lyapunov_spectrum`) only ever sees a plain `state -> new_state`
  function; it does not know or care whether that function came from
  Euler, Heun, RK4, RK6, a hand-written discrete map (logistic/Hénon), or
  a coupled network. This is why the same QR/Benettin code serves every
  example in `examples/` without a system-specific branch anywhere in the
  engine.
- **Coupling as a plain callable, not a fixed enum.** Network dynamics are
  `dfun(state, coupling, params) -> dstate` plus any
  `coupling(cvar_state, weights, params) -> coupling` callable — linear,
  sigmoidal, Kuramoto, or user-written, with no registry/dispatch layer to
  extend. `network_problem`/`network_dde_problem` (the front-door
  constructors from `notes/api_design_review.md`) just wire a `Network`
  (topology), a `dfun`, and a coupling callable together.
- **DDE support via an augmented `(state, ring_buffer)` carry.** For
  systems with a transmission delay, lyapax's DDE engine
  (`lyapunov_spectrum_dde`) generalizes the same Benettin/QR idea to a
  carry containing both the current state and a fixed-depth ring buffer
  of recent history, differentiating through both jointly (a delayed-value
  sensitivity, `d f / d x(t - tau)`, is exactly as real as the
  instantaneous one, and dropping it silently gives wrong exponents, not
  just imprecise ones). Two history-read modes exist:
  - **Grid-snapped (default):** `tau` is rounded to the nearest whole
    number of `dt` steps; every history read pulls one exact stored ring
    buffer sample. Fully differentiable, simple, but the delay actually
    simulated (`tau_eff`) is not exactly the requested `tau`, and that
    rounding error does not shrink smoothly as `dt` is refined.
  - **Hermite-interpolated (`interpolate=True`):** the ring buffer stores
    (value, derivative) pairs, letting any intra-step history read be
    reconstructed via a cubic Hermite interpolant — `tau` used exactly, no
    rounding, and each Runge-Kutta stage reads history at its own
    intra-step time instead of freezing one read across the whole step
    (fixing an O(dt) accuracy ceiling that otherwise caps every integrator
    at first order for delayed systems, regardless of its nominal order).
- **`jax.vmap`-batched parameter sweeps.** Because the network/DDE step
  carries `params` as data rather than closing over it
  (`network_step_parametrized`, `sweep_lyapunov_spectrum`), a whole grid
  of parameter values can be computed as one batched XLA call instead of
  one Python-level `lyapunov_spectrum` call per grid point.
- **Runs on GPU with zero code changes** — JAX picks the backend; nothing
  in lyapax is backend-specific. Whether that's *faster* is a separate,
  size-dependent question (small problems lose to per-call overhead; see
  `examples/plot_14_gpu_acceleration.py`).

## What lyapax can do

- Full or partial (`k <= d`) Lyapunov spectra for plain (uncoupled) ODEs,
  given either a hand-written JAX right-hand side or a `ModelSpec`/
  `build_jax_dfun`-compiled symbolic one.
- The same for coupled networks: arbitrary topology (`weights`), any
  coupling rule (built-in linear/sigmoidal/Kuramoto, or a user callable),
  fixed-step Euler/Heun/RK4/RK6 integration.
- Fixed-delay DDE networks, both a single *uniform* delay shared by every
  edge (`network_dde_problem(..., tau=...)`) and a genuine per-edge
  heterogeneous delay matrix (`Connectivity` + `lyapax.simulator.make_step_fn(...,
  delay_steps=...)`, the lower-level path `examples/plot_08_delayed_coupling.py`
  uses) — the latter currently only with the built-in hardcoded-linear
  coupling, see the explicit gap below.
- Grid-snapped (fast, simple) or Hermite-interpolated (exact `tau`,
  higher per-stage order) DDE history reads.
- Discrete chaotic maps (logistic, Hénon, or any user map) — same engine,
  `dt=1.0` per iterate, no integrator involved.
- Batched parameter/initial-condition sweeps via `jax.vmap`
  (`sweep_lyapunov_spectrum`), and transparent GPU execution.
- A validation suite anchored to independent sources (exact eigenvalues,
  structural invariants, published literature values — never against an
  unverified from-scratch reference implementation), so a correctness
  claim about lyapax's output always has an external check behind it.

## What lyapax cannot do (explicit non-goals and current gaps)

- **No adaptive/stiff ODE integration.** Every integrator is fixed-step
  (Euler/Heun/RK4/RK6); there is no `diffrax`-style adaptive-step or
  implicit solver. Exponents are for the numerical time-`dt` map, not the
  exact continuous flow — `dt`-convergence is the caller's responsibility
  to check (see `examples/plot_07_speed_and_accuracy.py`). Tracked as an
  M7 stretch goal, not started.
- **DDE delays must be known and fixed** — no state-dependent or
  distributed delays. Grid-snapped mode further restricts `tau` to
  (effectively) an integer multiple of `dt`; only `interpolate=True`
  removes that specific rounding restriction, and only for the
  uniform-delay, custom-`coupling_fn` path.
- **Per-edge delay matrix + custom `coupling_fn` is not wired up.**
  Heterogeneous per-edge delays currently only work with the vendored
  hardcoded-linear coupling formula (`coupling_fn=None`); combining a
  per-edge delay matrix with an arbitrary coupling callable (e.g. a
  delayed, per-edge Kuramoto network) needs an edge-aware coupling
  signature that doesn't exist yet.
- **No stochastic/noise-driven Lyapunov exponents.** Noise injection was
  deliberately dropped from the vendored step function (non-smooth /
  not meaningfully compatible with a deterministic spectrum) — an
  explicit non-goal, not a bug.
- **No PDEs / spatiotemporal chaos.** State is always a finite-dimensional
  vector (or `(n_state_vars, n_nodes)` array for networks); there is no
  spatial-discretization or infinite-dimensional support.
- **`history` column ordering is not stable near-degenerate exponents.**
  Columns are ordered once, by the final row; near-crossing exponents can
  swap order mid-run (`LyapunovResult.history`'s docstring has the detail).
- **`dfun_str` uses `exec()`-based codegen with no sanitization.** Fine for
  specs a caller writes themselves; not safe to build from untrusted
  input. (Tracked in `notes/milestones.md` M8 as a documentation gap to
  close — this milestone's docstring cleanup is the place to actually add
  that warning where a reader will see it.)

## Proposed Sphinx page structure

Concrete enough to start writing against, not final:

```
docs/
  index.rst                 (exists — toctree entry point)
  installation.md           (new — pip install variants, x64 requirement)
  background/
    lyapunov_exponents.md    (new — "What a Lyapunov exponent is" + Benettin/QR, from this note)
    lyapax_implementation.md (new — jvp/vmap, coupling-as-callable, DDE ring buffer + Hermite, vmap sweeps)
    validation.md            (new — the tier table from notes/validation_systems.md)
    capabilities.md          (new — the can/cannot-do lists above)
  api.rst                    (exists)
  auto_examples/             (generated by sphinx-gallery — do not hand-edit)
```

Once these pages exist with stable target names, Bucket 1 of the
cross-reference audit above can be done as a single pass: replace each
`notes/foo.md` string with the matching `:doc:`/`:ref:` role.

## Next steps (not done in this note)

- [ ] Write the four new `docs/background/*` pages from the "Background"
      section above (adapt, don't copy verbatim — this note is source
      material, not final prose).
- [ ] Do the Bucket 1/2/3 docstring and example cleanup pass per the
      tables above; re-run `sphinx-build -b html docs docs/_build/html`
      afterward and confirm no `notes/` string remains anywhere under
      `docs/_build/html` or in `src/lyapax`/`examples` source.
- [ ] Regenerate `docs/auto_examples/` (currently stale, predates the
      `ode_problem`/`network_problem`/`network_dde_problem` migration and
      `plot_12`–`plot_14`).
- [ ] Add the `dfun_str`/`exec()` trust-boundary warning called out above
      to `build_jax_dfun`'s docstring (M8 follow-up, do it as part of this
      pass since it's a docstring edit either way).
- [ ] Decide whether `README.md`'s "Further reading" section (currently
      linking `notes/*.md` directly) should also redirect to the built
      docs once they exist — out of strict scope here (README isn't
      autodoc'd or gallery-rendered) but likely wanted for consistency.
- [ ] Add `docs/installation.md` and wire all new pages into
      `docs/index.rst`'s toctree.
