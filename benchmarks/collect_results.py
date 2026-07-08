"""Runs every benchmark script under benchmarks/{lyapax,jitcode,jitcdde,chaostools}/
and writes benchmarks/results.json, so notes/benchmark_report.md's tables can be
regenerated with one command instead of manual copy-paste.

Usage: python benchmarks/collect_results.py
Requires: the lyapax dev environment, plus `pip install -e .[benchmark]` for
jitcode/jitcdde, plus a working `julia` on PATH with ChaosTools.jl installed
(see notes/benchmark_report.md's Environment table) -- chaostools scripts are
skipped with a warning, not a hard failure, if julia isn't found, since they're
not needed to validate the Python-side tools.

The lyapax scripts (only -- jitcode/jitcdde/ChaosTools.jl have no GPU backend)
are additionally re-run with `JAX_PLATFORMS=cuda`, so the performance table
also has a lyapax-on-GPU column, not just CPU. Like the julia check above,
this is a skip-with-warning, not a hard failure, if no working GPU is found --
`benchmarks/lyapax/_common.py` tags each result's "tool" with a "-gpu" suffix
based on the backend actually used (`jax.default_backend()`), so a GPU run and
a CPU run of the same script land in different results.json rows.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_PATH = HERE / "results.json"
TIMEOUT_S = 600

LYAPAX_SCRIPTS = [
    HERE / "lyapax" / "linear_ode.py",
    HERE / "lyapax" / "maps.py",
    HERE / "lyapax" / "lorenz.py",
    HERE / "lyapax" / "rossler.py",
    HERE / "lyapax" / "network.py",
    HERE / "lyapax" / "linear_scalar_dde.py",
    HERE / "lyapax" / "mackey_glass.py",
    HERE / "lyapax" / "network_scaling.py",
]

PY_SCRIPTS = LYAPAX_SCRIPTS + [
    HERE / "jitcode" / "linear_ode.py",
    HERE / "jitcode" / "lorenz.py",
    HERE / "jitcode" / "rossler.py",
    HERE / "jitcode" / "network.py",
    HERE / "jitcode" / "network_scaling.py",
    HERE / "jitcdde" / "linear_scalar.py",
    HERE / "jitcdde" / "mackey_glass.py",
]

JL_SCRIPTS = [
    HERE / "chaostools" / "linear_ode.jl",
    HERE / "chaostools" / "lorenz.jl",
    HERE / "chaostools" / "rossler.jl",
    HERE / "chaostools" / "network.jl",
    HERE / "chaostools" / "maps.jl",
    HERE / "chaostools" / "network_scaling.jl",
]


def parse_json_lines(stdout, stderr, label):
    results = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not results:
        print(f"WARNING: no JSON output from {label}", file=sys.stderr)
        print(stderr[-2000:], file=sys.stderr)
    return results


def run_python(script, extra_env=None):
    label = script.relative_to(HERE)
    if extra_env:
        label = f"{label} ({', '.join(f'{k}={v}' for k, v in extra_env.items())})"
    print(f"Running {label}...", file=sys.stderr)
    env = {**os.environ, **extra_env} if extra_env else None
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True,
        timeout=TIMEOUT_S, env=env,
    )
    return parse_json_lines(proc.stdout, proc.stderr, label)


def gpu_available():
    """Probe for a real, usable JAX GPU backend in a throwaway subprocess --
    `jax.devices()` can list a CudaDevice while every actual op still fails
    (a cudnn/driver mismatch, see tests/test_gpu.py), so this runs one tiny
    op rather than just checking device enumeration.
    """
    probe = (
        "import jax; jax.config.update('jax_enable_x64', True); "
        "x = jax.numpy.zeros(1) + 1; jax.block_until_ready(x); "
        "print(jax.default_backend())"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "JAX_PLATFORMS": "cuda"},
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "gpu"


def run_julia(script):
    julia = shutil.which("julia")
    if julia is None:
        print(f"SKIP {script.name}: julia not found on PATH", file=sys.stderr)
        return []
    print(f"Running {script.relative_to(HERE)}...", file=sys.stderr)
    proc = subprocess.run(
        [julia, str(script)], capture_output=True, text=True, timeout=TIMEOUT_S
    )
    return parse_json_lines(proc.stdout, proc.stderr, script.relative_to(HERE))


def main():
    all_results = []
    for script in PY_SCRIPTS:
        all_results.extend(run_python(script))

    if gpu_available():
        for script in LYAPAX_SCRIPTS:
            all_results.extend(run_python(script, extra_env={"JAX_PLATFORMS": "cuda"}))
    else:
        print("SKIP lyapax GPU pass: no working JAX GPU backend found "
              "(JAX_PLATFORMS=cuda probe failed)", file=sys.stderr)

    for script in JL_SCRIPTS:
        all_results.extend(run_julia(script))

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"Wrote {len(all_results)} results to {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
