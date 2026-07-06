# API Design Review: ODE/DDE Symmetry and Integrator Extensibility

## Status

Implemented, following this note's "Suggested Implementation Order"
almost exactly: integrator registry (`get_integrator`, `"euler"`/`"heun"`/
`"rk4"`, plus `"rk6"` added later), `use_heun` replaced by `integrator=`
(deprecated alias since removed entirely -- no backward-compat shim was
kept, per later direction), `Network` dataclass, `network_step`/
`network_problem`, `DDEProblem`/`dde_problem`/`network_dde_problem`,
`lyapunov_spectrum`/`lyapunov_spectrum_dde` accepting problem objects, and
examples/README updated to the new front door. Option A (two spectrum
functions, both accepting problem objects) was chosen over Option B (one
dispatcher) as recommended below. Adaptive ODE integrators were
deliberately left for later, per this note's own scoping -- see
`notes/stepping_accuracy_review.md` for that follow-up (not yet
implemented) and for a related, separate finding (coupling/history frozen
across RK stages) discovered while implementing DDE interpolation. The
rest of this document is the original proposal, kept for reference.

## Summary

The current API is already capable, but it exposes different mental models for
ODEs, DDEs, and networks:

- ODE users pass a simple `step_fn(state) -> new_state` into
  `lyapunov_spectrum`.
- DDE users pass a carry-style simulator step plus `state0`, `buf0`, and
  `params` into `lyapunov_spectrum_dde`.
- Network users construct a step through `make_network_step_fn` or
  `simulator.make_step_fn`, where model dynamics, coupling, graph topology,
  delays, time step, and integrator choice are all mixed into one factory.
- Integrator choice is currently not a first-class public concept. ODEs expose
  `rk4_step`, while the vendored simulator exposes `use_heun: bool`, which is
  too narrow once we want Euler, Heun, RK4, or future methods.

The main recommendation is to introduce one public simulation specification
layer that is shared by ODE and DDE, and one explicit integrator abstraction
that every step factory consumes. The low-level carry functions can remain, but
they should become implementation details or advanced APIs.

## Design Goals

1. Keep the simple ODE path simple.
2. Make DDE usage look structurally similar to ODE usage.
3. Make integrator selection explicit and extensible.
4. Keep adaptive methods possible, but do not force them into the v1 API until
   the Lyapunov semantics are clear.
5. Move network construction from a long positional factory into a small set of
   named objects.
6. Preserve JAX friendliness: pure functions, static shapes, `jit`/`vmap`
   compatibility, and differentiability through the step.

## Current API Shape

### ODE

The cleanest public path is:

```python
rhs = systems.lorenz(...)
step = rk4_step(rhs, dt)
result = lyapunov_spectrum(step, state0, dt, n_steps)
```

This is good because the step is just a differentiable time-`dt` map. The
Lyapunov function only needs `step_fn`, `state0`, `dt`, and run controls.

The limitation is that `rk4_step(rhs, dt)` is both the integrator and the step
factory. Adding more integrators means adding more top-level step factories
unless we introduce a shared abstraction.

### DDE

The DDE path is more explicit about implementation:

```python
step = make_scalar_delayed_step_fn(...)
result = lyapunov_spectrum_dde(step, state0, buf0, params, dt, n_steps)
```

This exposes the ring buffer and carry state directly. That is accurate, but it
is not parallel to the ODE API. A user has to understand more of the simulator
internals before they can compute a DDE spectrum.

The current DDE implementation is also tied to fixed integer-step delays. That
is a reasonable v1 constraint and should stay visible in the docs.

### Networks

`make_network_step_fn` is useful but overloaded. It asks for:

- `dfun`
- `weights`
- `cvar_indices`
- `params`
- `dt`
- `coupling_fn`
- `use_heun`

The delayed lower-level `make_step_fn` is more complex: it also takes
`has_delays`, `horizon`, `n_nodes`, `delay_steps`, `G_default`, `coup_a`,
`coup_b`, `tau_steps`, and `coupling_fn`.

The problem is not only the number of arguments. The factory combines four
separate concepts:

- model dynamics
- network topology
- coupling rule
- numerical integration

That makes the API hard to read and hard to extend.

## Recommendation: Public Concepts

The API should make these concepts explicit:

```python
System       # model dynamics: rhs/dfun, state shape, parameters
Network      # weights, delay steps, coupling variable indices
Coupling     # linear, sigmoidal, kuramoto, custom callable
Integrator   # euler, heun, rk4, future methods
Problem      # ODE or DDE initial data and delay/history information
```

These do not all need to be heavy classes. Small dataclasses or named
constructors are enough. The goal is to make function signatures short and
stable.

## Proposed Public API

### ODE, Minimal

```python
step = lyapax.ode_step(
    rhs,
    dt=0.01,
    integrator="rk4",
)

result = lyapax.lyapunov_spectrum(
    step,
    state0,
    dt=0.01,
    n_steps=50_000,
)
```

This preserves the current simple ODE workflow. `rk4_step(rhs, dt)` can remain
as a convenience alias.

### DDE, Minimal

```python
problem = lyapax.dde_problem(
    rhs_delayed,
    state0=state0,
    tau=2.0,
    dt=0.01,
    history=history,
    integrator="heun",
)

result = lyapax.lyapunov_spectrum(problem, n_steps=50_000)
```

or, if keeping separate functions is preferred:

```python
result = lyapax.lyapunov_spectrum_dde(
    problem,
    n_steps=50_000,
)
```

The important change is that `buf0`, `horizon`, `tau_steps`, and carry layout
are owned by the DDE problem object, not by the user-facing spectrum call.

### Network ODE

```python
network = lyapax.Network(
    weights=weights,
    cvar_indices=(0,),
)

step = lyapax.network_step(
    dfun=dfun,
    network=network,
    coupling=lyapax.coupling.Linear(),
    params=params,
    dt=0.01,
    integrator="rk4",
)
```

This is easier to read than `make_network_step_fn(...)` because topology,
coupling, and integration are named separately.

### Network DDE

```python
network = lyapax.Network(
    weights=weights,
    cvar_indices=(0,),
    delay_steps=delay_steps,
)

problem = lyapax.network_dde_problem(
    dfun=dfun,
    network=network,
    coupling=lyapax.coupling.Linear(),
    params=params,
    state0=state0,
    dt=0.01,
    history=history,
    integrator="heun",
)

result = lyapax.lyapunov_spectrum_dde(problem, n_steps=50_000, k=10)
```

This gives ODE and DDE the same construction pattern:

1. define dynamics,
2. define optional network/coupling,
3. define integrator,
4. compute Lyapunov spectrum.

## Integrator API

Integrator selection should not be a boolean.

Replace:

```python
use_heun: bool = True
```

with:

```python
integrator: str | Integrator = "heun"
```

Recommended first shape:

```python
class Integrator(Protocol):
    name: str

    def step(self, rhs, state, dt, params=None):
        ...
```

For network/DDE simulator internals, `rhs` may need to receive coupling and
params:

```python
def step(self, state, dfun, coupling, dt, params):
    ...
```

The exact internal callable signature can be private. The public point is that
users should say `integrator="rk4"` or `integrator=RK4()` instead of toggling
`use_heun`.

Initial fixed-step integrators:

- `"euler"`
- `"heun"`
- `"rk4"`

`rk4_step(rhs, dt)` can stay as a thin compatibility wrapper around the new
integrator machinery.

## Adaptive Integrators

Adaptive integration is possible for ODEs, but it should be treated carefully
for Lyapunov exponents.

The current Lyapunov algorithm assumes a repeated fixed-time map:

```text
state_n -> state_{n+1}
```

with a known elapsed time `dt` per step and QR renormalization every fixed
number of steps. Adaptive solvers break that assumption because the internal
substeps and accepted step sizes depend on the state and tolerance.

For ODEs, a reasonable future API could be:

```python
integrator = lyapax.integrators.Dopri5(rtol=1e-6, atol=1e-9)

step = lyapax.ode_step(
    rhs,
    dt=0.1,              # output sampling interval, not internal step size
    integrator=integrator,
)
```

In this design, `step` still advances by a fixed external interval `dt`, while
the adaptive integrator chooses internal substeps. That keeps
`lyapunov_spectrum` compatible because the outer map is still time-`dt`.

Open questions before adding this:

- Is differentiating through adaptive control flow acceptable for the target
  JAX version and performance goals?
- Should tolerances be static fields to avoid recompilation?
- Should rejected steps be differentiated through, or should the adaptive
  solver be wrapped as an opaque numerical method?
- How do we document that exponents are for the numerical adaptive flow map,
  not an exact continuous system?

For DDEs, adaptive integration is much harder. The current DDE implementation
uses integer-step ring buffers. Adaptive step sizes would require either:

- interpolation in the history buffer, or
- adaptive internal stepping while still writing to a fixed external history
  grid.

Both are real design changes. For now, DDE should stay fixed-step and
integer-delay. The API should not promise adaptive DDE support yet.

## What To Do With `make_network_step_fn`

Keep it temporarily, but demote it.

Recommended path:

1. Add `Network`, `Coupling`, and `Integrator` public constructors.
2. Add a clearer `network_step(...)` wrapper that calls the existing
   implementation.
3. Mark `make_network_step_fn` as a low-level compatibility function in the
   docs.
4. Eventually move it under an advanced namespace, for example
   `lyapax.lowlevel.make_network_step_fn`.

The new wrapper can be implemented without rewriting the simulator:

```python
def network_step(
    dfun,
    network: Network,
    coupling,
    params,
    dt: float,
    integrator: str | Integrator = "heun",
):
    ...
```

Internally it can still call the current `make_step_fn`. The API improvement
does not require a large numerical rewrite.

## Spectrum API: One Function Or Two?

There are two reasonable choices.

### Option A: Keep Two Spectrum Functions

```python
lyapunov_spectrum(step_fn, state0, ...)
lyapunov_spectrum_dde(problem, ...)
```

This is explicit and low risk. It also avoids hiding the fact that DDE uses an
augmented state dimension.

### Option B: One Dispatcher

```python
lyapunov_spectrum(problem_or_step, ...)
```

This gives the neatest public API, but it needs good type boundaries:

- `ODEProblem`
- `DDEProblem`
- plain `step_fn` for backward compatibility

Recommendation: use Option A first, but allow both functions to accept problem
objects. Add a single dispatcher later only if examples show it genuinely
reduces confusion.

## Backward Compatibility

Do not remove the existing functions immediately. They are useful for tests and
advanced users.

Keep:

- `rk4_step(rhs, dt)`
- `lyapunov_spectrum(step_fn, state0, dt, ...)`
- `lyapunov_spectrum_dde(...)`
- `make_network_step_fn(...)`
- `make_parametrized_network_step_fn(...)`

Add:

- `ode_step(rhs, dt, integrator="rk4")`
- `network_step(..., network=..., coupling=..., integrator="heun")`
- `dde_problem(...)`
- `network_dde_problem(...)`
- `Integrator` objects or a small integrator registry

Then update examples to prefer the new API while leaving old paths documented
as lower-level.

## Suggested Implementation Order

1. Add `integrators.py` registry support:
   `get_integrator("euler" | "heun" | "rk4")`.
2. Replace internal `use_heun` plumbing with `integrator`, while accepting
   `use_heun` as a deprecated compatibility argument.
3. Add `Network` dataclass.
4. Add `network_step(...)` wrapper around the existing network/simulator code.
5. Add `DDEProblem` dataclass that owns `state0`, `buf0`, `params`, `dt`,
   `horizon`, and the carry step.
6. Teach `lyapunov_spectrum_dde` to accept `DDEProblem`.
7. Update examples and README to show the new public path.
8. Only then consider adaptive ODE integrators.

## Main Review Finding

The numerical core does not need to be redesigned first. The main API issue is
that the public functions expose implementation details unevenly:

- ODE hides integration behind a simple step map.
- DDE exposes ring-buffer carry mechanics.
- Network setup exposes too many construction details at once.
- Integrator choice is not a first-class object.

A cleaner API should make ODE and DDE construction parallel, make integrators
replaceable, and keep the current carry/ring-buffer machinery behind problem
objects. Adaptive ODE support can fit later if the outer API preserves a
fixed-time map. Adaptive DDE support should remain out of scope until the
history interpolation and tangent semantics are designed explicitly.
