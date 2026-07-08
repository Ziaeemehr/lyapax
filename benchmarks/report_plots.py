"""Regenerate the comparison plot embedded in docs/background/benchmarks.md
from benchmarks/results.json -- companion to report_tables.py's tables,
same source data, same "don't hand-edit, rerun after collect_results.py"
contract.

Usage:
    python benchmarks/report_plots.py

Writes docs/_static/benchmarks_performance.png (every tool's warm
wall-clock time per system).
"""
from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS_PATH = HERE / "results.json"
STATIC_DIR = HERE.parent / "docs" / "_static"

# ---- palette (see the dataviz skill's reference/palette.md) -----------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

TOOL_COLOR = {
    "lyapax": "#2a78d6",           # categorical slot 1, blue
    "lyapax-rk6": "#1baf7a",       # slot 2, aqua
    "jitcode": "#eda100",          # slot 3, yellow
    "jitcdde": "#008300",          # slot 4, green
    "chaostools": "#4a3aa7",       # slot 5, violet
    "chaostools-rk4": "#e34948",   # slot 6, red
    "chaostools-rk6": "#e87ba4",   # slot 7, magenta
}
TOOL_LABEL = {
    "lyapax": "lyapax",
    "lyapax-rk6": "lyapax (RK6)",
    "jitcode": "jitcode",
    "jitcdde": "jitcdde",
    "chaostools": "ChaosTools.jl",
    "chaostools-rk4": "ChaosTools.jl (RK4)",
    "chaostools-rk6": "ChaosTools.jl (Vern6)",
}

SYSTEM_LABEL = {
    "linear_ode_tier0.1": "Linear ODE",
    "lorenz_tier1.1": "Lorenz",
    "rossler_tier1.2": "Rössler",
    "linear_network_tier3.1": "4-node network",
    "logistic_map_tier0.2": "Logistic map",
    "tent_map_tier0.2": "Tent map",
    "henon_map_tier0.3": "Hénon map",
    "linear_scalar_dde_tier4.2": "Linear scalar DDE",
    "mackey_glass_tier4.1": "Mackey-Glass",
}
SYSTEM_ORDER = list(SYSTEM_LABEL)


def load_results() -> dict[str, dict[str, dict]]:
    rows = json.loads(RESULTS_PATH.read_text())
    by_system: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_system.setdefault(row["system"], {})[row["tool"]] = row
    return by_system


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_performance(by_system: dict, out_path: Path) -> None:
    """Warm wall-clock time per system, all CPU tools, log-scaled x-axis.

    GPU rows are intentionally excluded here (they're in the table above):
    these toy-sized systems don't amortize GPU dispatch overhead, so a GPU
    bar would only distract from the CPU-vs-CPU comparison this plot is
    for -- see docs/background/jax_performance.md for where GPU does pay
    off.
    """
    tool_order = ["lyapax", "lyapax-rk6", "jitcode", "jitcdde", "chaostools",
                  "chaostools-rk4", "chaostools-rk6"]
    systems = [s for s in SYSTEM_ORDER if s in by_system]
    counts = [sum(1 for t in tool_order if t in by_system[s]) for s in systems]

    fig, axes = plt.subplots(
        len(systems), 1, figsize=(6.4, 0.34 * sum(counts) + 0.9),
        facecolor=SURFACE,
        gridspec_kw={"height_ratios": counts},
    )
    fig.suptitle("Warm wall-clock time per system (lower is faster)",
                  color=INK_PRIMARY, fontsize=11, y=0.995)

    for ax, sys_key in zip(axes, systems):
        rows = by_system[sys_key]
        present = [t for t in tool_order if t in rows]
        times = [rows[t]["warm_s"] for t in present]
        y = np.arange(len(present))[::-1]
        colors = [TOOL_COLOR[t] for t in present]
        ax.barh(y, times, color=colors, height=0.6, zorder=2)
        for yi, t, val in zip(y, present, times):
            ax.text(val * 1.15, yi, f"{val:.3g}s", va="center",
                     ha="left", fontsize=8, color=INK_SECONDARY)
        ax.set_yticks(y)
        ax.set_yticklabels([TOOL_LABEL[t] for t in present], fontsize=8.5,
                            color=INK_PRIMARY)
        ax.set_xscale("log")
        ax.set_xlim(min(times) / 3, max(max(times) * 6, min(times) * 20))
        ax.set_ylabel(SYSTEM_LABEL[sys_key], rotation=0, ha="right",
                       va="center", fontsize=9.5, color=INK_PRIMARY,
                       labelpad=8)
        _style_axes(ax)
        if ax is not axes[-1]:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("warm wall-clock time, seconds (log scale)",
                         fontsize=9, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    by_system = load_results()
    plot_performance(by_system, STATIC_DIR / "benchmarks_performance.png")
    print(f"Wrote {STATIC_DIR / 'benchmarks_performance.png'}")


if __name__ == "__main__":
    main()
