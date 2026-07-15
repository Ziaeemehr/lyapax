"""Shared plumbing for the jitcode side of docs/background/benchmarks.md.

jitcode_lyap's `integrate(t)` returns *local* Lyapunov exponents (growth
over the just-completed sub-interval), not a running average -- the
standard usage (per jitcode's own docs/examples) is to call it repeatedly
at a fixed spacing, discard the sub-intervals inside the transient, and
average the rest. `run_lyap_spectrum` below does exactly that, with
`spacing`/`transient_time`/`total_time` chosen to mirror lyapax's
`renorm_every*dt` / `t_transient` / `t_transient + n_steps*dt` for the same
system, so the two are as close to an apples-to-apples comparison as two
different integrators (lyapax: fixed-step RK4: jitcode: adaptive dopri5 by
default) allow.
"""
import json
import time

import numpy as np


def run_lyap_spectrum(ODE, state0, dt, n_steps, renorm_every, t_transient):
    ODE.set_initial_value(state0, 0.0)
    spacing = renorm_every * dt
    transient_time = t_transient
    total_time = t_transient + n_steps * dt
    n_transient_blocks = int(round(transient_time / spacing))
    n_blocks = int(round(total_time / spacing))

    local_lyaps = []
    t = 0.0
    for _ in range(n_blocks):
        t += spacing
        _state, lyaps, _vectors = ODE.integrate(t)
        local_lyaps.append(lyaps)

    local_lyaps = np.vstack(local_lyaps)
    exponents = local_lyaps[n_transient_blocks:].mean(axis=0)
    return np.sort(exponents)[::-1]


def time_build_and_run(build_fn, run_fn):
    """build_fn() -> ODE (includes C compilation); run_fn(ODE) -> exponents.
    Returns (first_call_s, warm_s, exponents_from_warm_call)."""
    t0 = time.perf_counter()
    ODE = build_fn()
    exponents = run_fn(ODE)
    first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    exponents = run_fn(ODE)
    warm_s = time.perf_counter() - t0

    return first_s, warm_s, exponents


def emit(system, exponents, first_s, warm_s, **extra):
    payload = {
        "tool": "jitcode",
        "system": system,
        "exponents": [float(x) for x in exponents],
        "first_call_s": first_s,
        "warm_s": warm_s,
        **extra,
    }
    print(json.dumps(payload))
    return payload
