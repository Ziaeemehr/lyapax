"""Regenerate the Results tables in docs/background/benchmarks.md from
benchmarks/results.json, so the doc never needs a manual copy-paste edit
after a re-run of collect_results.py.

Usage:
    python benchmarks/report_tables.py            # print all tables to stdout
    python benchmarks/report_tables.py --write     # also splice them into
                                                    # the doc, replacing the
                                                    # content between each
                                                    # pair of
                                                    # <!-- AUTO:... --> markers

Reference values (exact eigenvalues, Lambert W roots, published/qualitative
bands) are derived here from the same parameters the benchmark scripts use
(benchmarks/{lyapax,jitcode,jitcdde,chaostools}/*).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.special import lambertw

HERE = Path(__file__).parent
RESULTS_PATH = HERE / "results.json"
REPORT_PATHS = [
    HERE.parent / "docs" / "background" / "benchmarks.md",
]

TOOL_LABEL = {
    "lyapax": "lyapax",
    "lyapax-rk6": "lyapax (RK6)",
    "lyapax-gpu": "lyapax (GPU)",
    "lyapax-rk6-gpu": "lyapax (RK6, GPU)",
    "jitcode": "jitcode",
    "jitcdde": "jitcdde",
    "chaostools": "ChaosTools.jl",
    "chaostools-rk4": "ChaosTools.jl (RK4)",
    "chaostools-rk6": "ChaosTools.jl (Vern6)",
}

# ---- reference values ------------------------------------------------
# Linear ODE, Tier 0.1: A = diag(-1, -2, -5).
LINEAR_ODE_EXACT = np.array([-1.0, -2.0, -5.0])

# 4-node linear network, Tier 3.1: eigvals(gamma*I + G*W), 4-cycle graph.
_NET_W = np.array([
    [0., 1., 0., 1.],
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
    [1., 0., 1., 0.],
])
_NET_GAMMA, _NET_G = -2.0, 0.5
LINEAR_NETWORK_EXACT = np.sort(
    np.linalg.eigvals(_NET_GAMMA * np.eye(4) + _NET_G * _NET_W).real
)[::-1]

# Lorenz, Tier 1.1/2.
LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
LORENZ_SUM_EXACT = -(LORENZ_SIGMA + 1.0 + LORENZ_BETA)
LORENZ_LAMBDA1_PUBLISHED = 0.9056

# Roessler, Tier 1.2/2 -- only a qualitative published band, no fixed digit.
ROSSLER_LAMBDA1_PUBLISHED = 0.07

# Maps, Tier 0.2/0.3.
LOGISTIC_TENT_EXACT = np.log(2.0)
HENON_B = 0.3
HENON_SUM_EXACT = np.log(HENON_B)

# Linear scalar DDE, Tier 4.2: x' = -a*x(t-tau).
DDE_A, DDE_TAU = 0.5, 0.3
DDE_LAMBDA1_EXACT = float((lambertw(-DDE_A * DDE_TAU, k=0) / DDE_TAU).real)

# Mackey-Glass, Tier 4.1 -- qualitative literature bands only.
MG_LAMBDA1_RANGE = (1e-3, 1e-2)
MG_KY_RANGE = (2.0, 3.0)


def load_results() -> dict[str, dict[str, dict]]:
    """system -> tool -> result row."""
    rows = json.loads(RESULTS_PATH.read_text())
    by_system: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_system.setdefault(row["system"], {})[row["tool"]] = row
    return by_system


def _present(by_system: dict, systems: list[str], tool_order: list[str]) -> list[str]:
    present = []
    for tool in tool_order:
        if any(tool in by_system.get(s, {}) for s in systems):
            present.append(tool)
    return present


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _diff_note(label_vals: list[tuple[str, float]]) -> str:
    parts = [f"{label}: `{abs(v):.2e}`" for label, v in label_vals]
    return "max abs diff from reference -- " + ", ".join(parts)


def _spectrum_cell(row: dict | None) -> str:
    if row is None:
        return "--"
    return "`[" + ", ".join(f"{x:.4f}" for x in row["exponents"]) + "]`"


def _spectrum_diff(row: dict, exact: np.ndarray) -> float:
    got = np.sort(np.asarray(row["exponents"]))[::-1]
    return float(np.max(np.abs(got - exact)))


# ---- table builders ----------------------------------------------------

def build_ode_table(by_system: dict) -> str:
    order = ["lyapax", "lyapax-rk6", "jitcode", "chaostools",
              "chaostools-rk4", "chaostools-rk6"]
    systems = ["linear_ode_tier0.1", "lorenz_tier1.1", "rossler_tier1.2",
               "linear_network_tier3.1"]
    tools = _present(by_system, systems, order)
    headers = ["System"] + [TOOL_LABEL[t] for t in tools] + ["Reference", "Notes"]
    rows = []

    lin = by_system.get("linear_ode_tier0.1", {})
    cells = [_spectrum_cell(lin.get(t)) for t in tools]
    diffs = [(TOOL_LABEL[t], _spectrum_diff(lin[t], LINEAR_ODE_EXACT))
             for t in tools if t in lin]
    rows.append(["Linear ODE (Tier 0.1)", *cells, "exact `[-1, -2, -5]`",
                 _diff_note(diffs)])

    lor = by_system.get("lorenz_tier1.1", {})
    l1_cells = [f"`{lor[t]['exponents'][0]:.5f}`" if t in lor else "--" for t in tools]
    diffs = [(TOOL_LABEL[t], lor[t]["exponents"][0] - LORENZ_LAMBDA1_PUBLISHED)
             for t in tools if t in lor]
    rows.append(["Lorenz λ1 (Tier 1.1/2)", *l1_cells,
                 f"published `≈{LORENZ_LAMBDA1_PUBLISHED}`", _diff_note(diffs)])

    sum_cells = [f"`{sum(lor[t]['exponents']):.4f}`" if t in lor else "--" for t in tools]
    diffs = [(TOOL_LABEL[t], sum(lor[t]["exponents"]) - LORENZ_SUM_EXACT)
             for t in tools if t in lor]
    rows.append(["Lorenz sum(λ)", *sum_cells,
                 f"exact `{LORENZ_SUM_EXACT:.4f}` (`-(σ+1+β)`)",
                 _diff_note(diffs)])

    ros = by_system.get("rossler_tier1.2", {})
    l1_cells = [f"`{ros[t]['exponents'][0]:.5f}`" if t in ros else "--" for t in tools]
    diffs = [(TOOL_LABEL[t], ros[t]["exponents"][0] - ROSSLER_LAMBDA1_PUBLISHED)
              for t in tools if t in ros]
    rows.append(["Rössler λ1 (Tier 1.2/2)", *l1_cells,
                 f"qualitative `≈{ROSSLER_LAMBDA1_PUBLISHED}`", _diff_note(diffs)])

    net = by_system.get("linear_network_tier3.1", {})
    cells = [_spectrum_cell(net.get(t)) for t in tools]
    diffs = [(TOOL_LABEL[t], _spectrum_diff(net[t], LINEAR_NETWORK_EXACT))
             for t in tools if t in net]
    rows.append(["4-node linear network (Tier 3.1)", *cells,
                 "exact `[-1, -2, -2, -3]`", _diff_note(diffs)])

    return _md_table(headers, rows)


def build_maps_table(by_system: dict) -> str:
    order = ["lyapax", "chaostools"]
    systems = ["logistic_map_tier0.2", "tent_map_tier0.2", "henon_map_tier0.3"]
    tools = _present(by_system, systems, order)
    headers = ["System"] + [TOOL_LABEL[t] for t in tools] + ["Exact", "Notes"]
    rows = []

    for key, label in [("logistic_map_tier0.2", "Logistic map `r=4` (Tier 0.2)"),
                        ("tent_map_tier0.2", "Tent map (Tier 0.2)")]:
        sys_rows = by_system.get(key, {})
        cells = [f"`{sys_rows[t]['exponents'][0]:.7f}`" if t in sys_rows else "--"
                 for t in tools]
        diffs = [(TOOL_LABEL[t], sys_rows[t]["exponents"][0] - LOGISTIC_TENT_EXACT)
                 for t in tools if t in sys_rows]
        rows.append([label, *cells, f"`ln 2 = {LOGISTIC_TENT_EXACT:.7f}`",
                     _diff_note(diffs)])

    hen = by_system.get("henon_map_tier0.3", {})
    sum_cells = [f"`{sum(hen[t]['exponents']):.6f}`" if t in hen else "--" for t in tools]
    diffs = [(TOOL_LABEL[t], sum(hen[t]["exponents"]) - HENON_SUM_EXACT)
             for t in tools if t in hen]
    indiv = "; ".join(
        f"{TOOL_LABEL[t]} {_spectrum_cell(hen[t])}" for t in tools if t in hen
    )
    rows.append(["Hénon map, sum(λ) (Tier 0.3)", *sum_cells,
                 f"`ln {HENON_B} = {HENON_SUM_EXACT:.6f}`",
                 _diff_note(diffs) + "; individual exponents: " + indiv])

    return _md_table(headers, rows)


def build_dde_table(by_system: dict) -> str:
    order = ["lyapax", "lyapax-rk6", "jitcdde"]
    systems = ["linear_scalar_dde_tier4.2", "mackey_glass_tier4.1"]
    tools = _present(by_system, systems, order)
    headers = ["System"] + [TOOL_LABEL[t] for t in tools] + ["Reference", "Notes"]
    rows = []

    dde = by_system.get("linear_scalar_dde_tier4.2", {})
    cells = [f"`{dde[t]['exponents'][0]:.5f}`" if t in dde else "--" for t in tools]
    diffs = [(TOOL_LABEL[t], dde[t]["exponents"][0] - DDE_LAMBDA1_EXACT)
             for t in tools if t in dde]
    rows.append(["Linear scalar DDE (Tier 4.2)", *cells,
                 f"Lambert W root `{DDE_LAMBDA1_EXACT:.6f}`", _diff_note(diffs)])

    mg = by_system.get("mackey_glass_tier4.1", {})
    cells = [f"`{mg[t]['exponents'][0]:.5f}`" if t in mg else "--" for t in tools]
    lo, hi = MG_LAMBDA1_RANGE
    inside = "; ".join(
        f"{TOOL_LABEL[t]} {'inside' if lo <= mg[t]['exponents'][0] <= hi else 'OUTSIDE'} band"
        for t in tools if t in mg
    )
    rows.append(["Mackey-Glass λ1 (Tier 4.1)", *cells,
                 f"qualitative `{lo:.0e}`-`{hi:.0e}`", inside])

    ky_cells = [f"`{mg[t]['kaplan_yorke_dim']:.3f}`" if t in mg and "kaplan_yorke_dim" in mg[t]
                else "--" for t in tools]
    lo, hi = MG_KY_RANGE
    inside = "; ".join(
        f"{TOOL_LABEL[t]} {'inside' if lo <= mg[t]['kaplan_yorke_dim'] <= hi else 'OUTSIDE'} band"
        for t in tools if t in mg and "kaplan_yorke_dim" in mg[t]
    )
    rows.append(["Mackey-Glass KY dimension", *ky_cells, f"`{lo}`-`{hi}`", inside])

    rows.append(["2-node delayed linear network (Tier 4.3)",
                 *(["not yet run"] * len(tools)), "Lambert W root (2x2)",
                 "deferred -- not yet run against jitcdde"])

    return _md_table(headers, rows)


def build_performance_table(by_system: dict) -> str:
    order = ["lyapax", "lyapax-rk6", "lyapax-gpu", "lyapax-rk6-gpu",
              "jitcode", "jitcdde", "chaostools", "chaostools-rk4",
              "chaostools-rk6"]
    systems = ["linear_ode_tier0.1", "lorenz_tier1.1", "rossler_tier1.2",
               "linear_network_tier3.1", "linear_scalar_dde_tier4.2",
               "mackey_glass_tier4.1"]
    labels = {
        "linear_ode_tier0.1": "Linear ODE (Tier 0.1)",
        "lorenz_tier1.1": "Lorenz",
        "rossler_tier1.2": "Rössler",
        "linear_network_tier3.1": "4-node network (Tier 3.1)",
        "linear_scalar_dde_tier4.2": "Linear scalar DDE (Tier 4.2)",
        "mackey_glass_tier4.1": "Mackey-Glass",
    }
    tools = _present(by_system, systems, order)
    headers = ["System"] + [TOOL_LABEL[t] for t in tools]
    rows = []
    for sys_key in systems:
        sys_rows = by_system.get(sys_key, {})
        if not sys_rows:
            continue
        warm = [f"`{sys_rows[t]['warm_s']:.3f}s`" if t in sys_rows else "--" for t in tools]
        rows.append([labels[sys_key], *warm])
    return _md_table(headers, rows)


def build_scaling_table(by_system: dict) -> str:
    order = ["lyapax", "lyapax-gpu", "chaostools-rk4", "jitcode"]
    sizes = [50, 200, 1000, 2000]
    systems = [f"kuramoto_scaling_d{d}" for d in sizes]
    tools = _present(by_system, systems, order)
    headers = ["Network size (d)"] + [TOOL_LABEL[t] for t in tools]
    rows = []
    for d, sys_key in zip(sizes, systems):
        sys_rows = by_system.get(sys_key, {})
        if not sys_rows:
            continue
        warm = [f"`{sys_rows[t]['warm_s']:.3f}s`" if t in sys_rows else "not attempted"
                for t in tools]
        rows.append([f"`{d}`", *warm])
    return _md_table(headers, rows)


MARKERS = {
    "ode-accuracy": build_ode_table,
    "maps-accuracy": build_maps_table,
    "dde-accuracy": build_dde_table,
    "performance": build_performance_table,
    "scaling": build_scaling_table,
}


def splice(report_text: str, by_system: dict, report_path: Path) -> str:
    for name, builder in MARKERS.items():
        begin, end = f"<!-- AUTO:{name} -->", f"<!-- END AUTO:{name} -->"
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        if not pattern.search(report_text):
            raise SystemExit(
                f"Marker pair {begin} ... {end} not found in {report_path} -- "
                "add it around the table to auto-generate before running --write."
            )
        replacement = f"{begin}\n{builder(by_system)}\n{end}"
        report_text = pattern.sub(lambda _m, r=replacement: r, report_text)
    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                         help="splice tables into docs/background/benchmarks.md")
    args = parser.parse_args()

    by_system = load_results()

    print("### Accuracy -- ODE systems\n")
    print(build_ode_table(by_system))
    print("\n### Accuracy -- Map systems\n")
    print(build_maps_table(by_system))
    print("\n### Accuracy -- DDE systems\n")
    print(build_dde_table(by_system))
    print("\n### Performance\n")
    print(build_performance_table(by_system))
    print("\n### Network-size scaling\n")
    print(build_scaling_table(by_system))

    if args.write:
        for report_path in REPORT_PATHS:
            text = report_path.read_text()
            report_path.write_text(splice(text, by_system, report_path))
            print(f"\nWrote tables into {report_path}", flush=True)


if __name__ == "__main__":
    main()
