# Possible Solutions to Open Issues

This note collects package-level design ideas relevant to
`notes/open_issues.md`, especially the still-open DDE accuracy problem and
the suspected ring-buffer time-indexing issue.

## Summary

The closest external reference is JiTCDDE. Its DDE solver uses a continuous
history representation: after each accepted integration step, it stores an
anchor containing

```text
(time, state, derivative)
```

Delayed values are reconstructed from neighboring anchors with cubic Hermite
interpolation. This is conceptually close to `lyapax`'s current
`interpolate=True` path, which stores value and derivative in the DDE history
buffer.

The main difference is not the interpolation formula. The difference is the
history model:

- JiTCDDE stores anchors at explicit physical times.
- `lyapax` stores samples in a uniform ring buffer and infers physical time
  from the integer step index.

That difference matters because the current open issue is likely about time
semantics: which physical time a ring slot represents, and whether delayed
reads ask for the correct bracketing samples.

## JiTCDDE's Memory Strategy

JiTCDDE does **not** keep the whole trajectory forever. It uses a sliding
continuous-history window.

Internally, JiTCDDE's past is a cubic Hermite spline, backed by anchors. During
integration, it repeatedly calls `forget(max_delay)`. The corresponding
`chspy.CubicHermiteSpline.forget` implementation removes anchors that are out
of reach of the largest delay:

```python
threshold = self.t - delay
while self[1].time < threshold:
    self.pop(0)
```

So JiTCDDE keeps only enough anchors to interpolate any delayed value in the
interval

```text
[current_time - max_delay, current_time]
```

plus the extra bracketing anchor needed for Hermite interpolation at the left
edge of that window.

Memory scaling is therefore approximately

```text
O(number_of_accepted_steps_inside_max_delay_window * state_dimension * 2)
```

The factor of two comes from storing both state and derivative at each anchor.

This differs from `lyapax`'s ring buffer:

```text
O(ceil(max_delay / dt) * n_cvar * n_nodes)
```

The ring buffer is still the better fit for large, fixed-step network
simulations because its memory use is predictable and independent of total
simulation length. JiTCDDE's anchor history is also bounded, but the number of
anchors depends on adaptive step sizes. If the solver takes many small accepted
steps inside one delay window, memory grows accordingly.

## Practical Lesson for `lyapax`

The lesson is probably not "replace the ring buffer with a JiTCDDE-style list".
For large systems, the ring buffer is still a reasonable design.

The more useful lesson is:

```text
Keep the ring buffer, but make each slot's physical time convention explicit.
```

A JiTCDDE-like convention for the ring buffer would be:

```text
slot k stores the anchor at physical time k * dt

anchor value      = y(k * dt)
anchor derivative = f(y(k * dt), history at k * dt)
```

Then a delayed read during an RK stage should request

```text
history((t + c_i) * dt - tau)
```

where `t` is the current integer step and `c_i` is the RK stage's Butcher node.
The interpolated read should then choose the two ring slots bracketing that
physical delayed time.

This is the same conceptual operation JiTCDDE performs with explicit-time
anchors, but implemented over a fixed-size circular array.

## Suspected Ring-Buffer Off-by-One

The current implementation writes `new_state[cvar_idx]` after advancing one
step, but stores it using the old step index:

```python
new_state = integrate(...)
new_buf = _write_ring(buf, t, new_state[cvar_idx], horizon)
```

Semantically, `new_state` is the state at time `(t + 1) * dt`, not at
`t * dt`.

That makes this convention suspicious:

```python
return buf[(step - tau_steps) % horizon]
```

If slot `t` actually contains the value for time `t + 1`, delayed reads can be
shifted by one step even though all modular arithmetic is internally
consistent. Such a shift would be exactly the kind of error that can leave a
high-order interpolation formula correct in isolation while the full DDE
trajectory still converges at only `O(dt)`.

The first concrete investigation should therefore be independent of Lyapunov
exponents and QR:

1. Define the invariant: "slot `k % horizon` stores the value at physical time
   `k * dt`."
2. Write a small deterministic buffer test that advances several steps and logs
   the physical time assigned to every write.
3. For delayed reads, log the requested delayed physical time and the slot or
   slots used to reconstruct it.
4. Check that exact-grid reads satisfy

   ```text
   read(step=t, tau_steps=m) == value_at_time((t - m) * dt)
   ```

5. Only after that invariant is verified, retry stage-time delayed lookup with
   `step + c_i`.

A likely fix to test is writing the newly computed state under `t + 1`, not
under `t`, while ensuring initialization fills the history consistently:

```python
new_buf = _write_ring(buf, t + 1, new_state[cvar_idx], horizon)
```

The interpolated version would need the same convention:

```python
new_buf = _write_ring_interp(
    buf,
    t + 1,
    new_state[cvar_idx],
    new_deriv,
    horizon,
)
```

This change should not be made blindly; it needs targeted tests because initial
history seeding and `tau_steps` semantics must move together.

## Stage-Time Delayed Reads

For a correct RK treatment of a DDE, each internal stage should evaluate the
complete right-hand side at that stage's own time. For a delayed coupling term,
that means the delayed lookup should be evaluated at

```text
(step + c_i) * dt - tau
```

not only at

```text
step * dt - tau
```

This was already attempted and did not improve convergence, but the package
comparison suggests a possible reason: if the ring-buffer slots are shifted by
one step, stage-time interpolation is still using the wrong physical samples.

Recommended order:

1. Fix or rule out the ring-buffer time convention.
2. Add exact-grid delayed-read tests.
3. Add fractional-time Hermite delayed-read tests against a known smooth
   function.
4. Retry stage-time delayed reads only after the above pass.

## Julia Packages

Julia's DelayDiffEq follows the same broad architecture as JiTCDDE:

- represent DDE solving as ODE solving plus history interpolation;
- use `MethodOfSteps(ODEAlg())`;
- track declared lags so discontinuities can be handled more accurately.

This supports the same conclusion: delayed values need to be evaluated as
history values at precise physical times, not as vaguely indexed buffer slots.

ChaosTools and DynamicalSystems.jl are more relevant to the Lyapunov side than
to the DDE history side. Their Lyapunov-spectrum machinery follows the standard
tangent dynamics plus QR-renormalization pattern, which is consistent with the
current conclusion in `notes/open_issues.md`: the observed `O(dt)` behavior is
more likely in the primal DDE stepping than in the QR/Lyapunov estimator.

## Recommendation

Do not abandon the ring buffer. It is the right memory model for large
fixed-step network simulations.

Instead, import JiTCDDE's stronger invariant:

```text
Every stored history item has a precise physical time.
Every delayed read asks for a precise physical time.
Interpolation only maps from that requested time to neighboring stored times.
```

For `lyapax`, that likely means:

- document the ring-buffer time convention directly in `_write_ring`,
  `_read_uniform_delayed_cvar`, `_write_ring_interp`, and
  `_read_uniform_delayed_cvar_interp`;
- test the convention without any Lyapunov machinery;
- consider changing DDE writes from `t` to `t + 1`;
- only then reintroduce per-stage delayed reads using `step + c_i`.

If those tests confirm the suspected shift, this would explain why Hermite
interpolation tests pass in isolation while the full DDE trajectory remains
first order.

