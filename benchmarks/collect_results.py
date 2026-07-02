"""Runs every benchmark script under benchmarks/{lyapax,jitcode,jitcdde,chaostools}/
and writes benchmarks/results.json, so notes/benchmark_report.md's tables can be
regenerated with one command instead of manual copy-paste.

Usage: python benchmarks/collect_results.py
Requires: the lyapax dev environment, plus `pip install -e .[benchmark]` for
jitcode/jitcdde, plus a working `julia` on PATH with ChaosTools.jl installed
(see notes/benchmark_report.md's Environment table) -- chaostools scripts are
skipped with a warning, not a hard failure, if julia isn't found, since they're
not needed to validate the Python-side tools.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_PATH = HERE / "results.json"
TIMEOUT_S = 600

PY_SCRIPTS = [
    HERE / "lyapax" / "linear_ode.py",
    HERE / "lyapax" / "maps.py",
    HERE / "lyapax" / "lorenz.py",
    HERE / "lyapax" / "rossler.py",
    HERE / "lyapax" / "network.py",
    HERE / "lyapax" / "linear_scalar_dde.py",
    HERE / "lyapax" / "mackey_glass.py",
    HERE / "jitcode" / "linear_ode.py",
    HERE / "jitcode" / "lorenz.py",
    HERE / "jitcode" / "rossler.py",
    HERE / "jitcode" / "network.py",
    HERE / "jitcdde" / "linear_scalar.py",
    HERE / "jitcdde" / "mackey_glass.py",
]

JL_SCRIPTS = [
    HERE / "chaostools" / "linear_ode.jl",
    HERE / "chaostools" / "lorenz.jl",
    HERE / "chaostools" / "rossler.jl",
    HERE / "chaostools" / "network.jl",
    HERE / "chaostools" / "maps.jl",
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


def run_python(script):
    print(f"Running {script.relative_to(HERE)}...", file=sys.stderr)
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=TIMEOUT_S
    )
    return parse_json_lines(proc.stdout, proc.stderr, script.relative_to(HERE))


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
    for script in JL_SCRIPTS:
        all_results.extend(run_julia(script))

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"Wrote {len(all_results)} results to {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
