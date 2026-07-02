"""Shared plumbing for the jitcdde side of notes/benchmark_report.md.

jitcdde_lyap's `integrate(target_time)` returns `(state, local_lyaps,
integration_time)` -- the third element is the *actual* time advanced
(jitcdde integrates adaptively and interpolates back to `target_time`, so
this can differ from the nominal spacing), and jitcdde's own docs say local
exponents must be weighted by it when averaging, not just arithmetically
averaged. `run_lyap_spectrum` mirrors lyapax's `renorm_every*dt` /
`t_transient` / `t_transient + n_steps*dt` for the spacing/transient/total
time, same as benchmarks/jitcode/_common.py's analogous function.
"""
import json
import time

import numpy as np


def run_lyap_spectrum(DDE, past_value, dt, n_steps, renorm_every, t_transient):
    DDE.constant_past(past_value)
    DDE.step_on_discontinuities()

    spacing = renorm_every * dt
    transient_time = t_transient
    total_time = t_transient + n_steps * dt
    n_transient_blocks = int(round(transient_time / spacing))
    n_blocks = int(round(total_time / spacing))

    local_lyaps = []
    weights = []
    t = DDE.t
    for _ in range(n_blocks):
        t += spacing
        _state, lyaps, integration_time = DDE.integrate(t)
        local_lyaps.append(lyaps)
        weights.append(integration_time)

    local_lyaps = np.vstack(local_lyaps)
    weights = np.array(weights)
    exponents = np.average(local_lyaps[n_transient_blocks:], weights=weights[n_transient_blocks:], axis=0)
    return np.sort(exponents)[::-1]


def time_build_and_run(build_fn, run_fn):
    """build_fn() -> DDE (includes C compilation); run_fn(DDE) -> exponents.
    run_fn must call DDE.constant_past(...) itself (constant_past calls
    reset_integrator internally, so it's a cheap in-place reset -- no
    recompilation -- letting the same compiled DDE be reused for the warm
    call, mirroring jitcode's set_initial_value pattern).
    Returns (first_call_s, warm_s, exponents_from_warm_call)."""
    t0 = time.perf_counter()
    DDE = build_fn()
    exponents = run_fn(DDE)
    first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    exponents = run_fn(DDE)
    warm_s = time.perf_counter() - t0

    return first_s, warm_s, exponents


def emit(system, exponents, first_s, warm_s, **extra):
    payload = {
        "tool": "jitcdde",
        "system": system,
        "exponents": [float(x) for x in exponents],
        "first_call_s": first_s,
        "warm_s": warm_s,
        **extra,
    }
    print(json.dumps(payload))
    return payload
