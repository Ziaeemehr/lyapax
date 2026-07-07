"""Tier 0.1: linear ODE, 3 distinct real eigenvalues. Same system/params as
benchmarks/lyapax/linear_ode.py / tests/test_lyapunov_core.py's version.
"""
from _common import emit, run_lyap_spectrum, time_build_and_run
from jitcode import jitcode_lyap, y


def build():
    f = [-1.0 * y(0), -2.0 * y(1), -5.0 * y(2)]
    ODE = jitcode_lyap(f, n_lyap=3)
    ODE.set_integrator("dopri5")
    return ODE


def run(ODE):
    return run_lyap_spectrum(
        ODE, state0=[0.3, -0.2, 0.5],
        dt=1e-3, n_steps=20_000, renorm_every=10, t_transient=5.0,
    )


if __name__ == "__main__":
    first_s, warm_s, exponents = time_build_and_run(build, run)
    emit("linear_ode_tier0.1", exponents, first_s, warm_s)
