"""Tier 1.2/2: Rossler system. Same params as benchmarks/lyapax/rossler.py."""
from jitcode import jitcode_lyap, y

from _common import time_build_and_run, run_lyap_spectrum, emit


def build():
    a, b, c = 0.2, 0.2, 5.7
    f = [
        -y(1) - y(2),
        y(0) + a * y(1),
        b + y(2) * (y(0) - c),
    ]
    ODE = jitcode_lyap(f, n_lyap=3)
    ODE.set_integrator("dopri5")
    return ODE


def run(ODE):
    return run_lyap_spectrum(
        ODE, state0=[1.0, 1.0, 1.0],
        dt=1e-2, n_steps=200_000, renorm_every=10, t_transient=200.0,
    )


if __name__ == "__main__":
    first_s, warm_s, exponents = time_build_and_run(build, run)
    emit("rossler_tier1.2", exponents, first_s, warm_s)
