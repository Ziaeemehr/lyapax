"""Tier 4.1: Mackey-Glass. Same params as benchmarks/lyapax/mackey_glass.py."""
import numpy as np
from jitcdde import jitcdde_lyap, y, t

from _common import time_build_and_run, run_lyap_spectrum, emit


def _kaplan_yorke_dimension(exponents: np.ndarray) -> float:
    cumsum = np.cumsum(exponents)
    j = 0
    for i in range(1, len(exponents) + 1):
        if cumsum[i - 1] >= 0:
            j = i
        else:
            break
    if j == 0 or j >= len(exponents):
        return float(j)
    return j + cumsum[j - 1] / abs(exponents[j])


def build():
    beta, gamma, n, tau = 0.2, 0.1, 10.0, 17.0
    f = [beta * y(0, t - tau) / (1 + y(0, t - tau) ** n) - gamma * y(0)]
    DDE = jitcdde_lyap(f, n_lyap=8)
    return DDE


def run(DDE):
    return run_lyap_spectrum(
        DDE, past_value=[1.2],
        dt=1.0, n_steps=30_000, renorm_every=10, t_transient=3_000.0,
    )


if __name__ == "__main__":
    first_s, warm_s, exponents = time_build_and_run(build, run)
    ky = _kaplan_yorke_dimension(exponents)
    emit("mackey_glass_tier4.1", exponents, first_s, warm_s, kaplan_yorke_dim=ky)
