"""Shared plumbing for the lyapax side of notes/benchmark_report.md.

Every benchmark script (lyapax, jitcode, jitcdde, chaostools) prints one
JSON object as its last stdout line, with at least
{"tool", "system", "exponents"} -- collect_results.py parses that line to
build the report's tables. Timing follows benchmark_report.md's
methodology section: first-call (includes JIT trace/compile) and warm
(steady-state) are reported separately.
"""
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)


def time_and_run(fn, *args, **kwargs):
    """Call fn twice, returns (first_call_seconds, warm_seconds, result)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    jax.block_until_ready(result)
    first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    jax.block_until_ready(result)
    warm_s = time.perf_counter() - t0

    return first_s, warm_s, result


def emit(tool, system, exponents, first_s, warm_s, **extra):
    # JAX_PLATFORMS is only a *default* (see setdefault above) -- a caller
    # (collect_results.py's GPU pass) can already override it before this
    # script runs. Tag the tool name with the backend actually used so a
    # GPU run and a CPU run of the same script land in different
    # results.json rows instead of one silently overwriting the other.
    if jax.default_backend() == "gpu" and not tool.endswith("-gpu"):
        tool = f"{tool}-gpu"
    payload = {
        "tool": tool,
        "system": system,
        "exponents": [float(x) for x in exponents],
        "first_call_s": first_s,
        "warm_s": warm_s,
        **extra,
    }
    print(json.dumps(payload))
    return payload


if __name__ == "__main__":
    sys.exit("this module is imported by other benchmarks/lyapax/*.py scripts, not run directly")
