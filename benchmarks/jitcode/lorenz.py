"""Tier 1.1/2: Lorenz system. Same params as benchmarks/lyapax/lorenz.py."""
from _common import emit, run_lyap_spectrum, time_build_and_run
from jitcode import jitcode_lyap, y


def build():
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    f = [
        sigma * (y(1) - y(0)),
        y(0) * (rho - y(2)) - y(1),
        y(0) * y(1) - beta * y(2),
    ]
    ODE = jitcode_lyap(f, n_lyap=3)
    ODE.set_integrator("dopri5")
    return ODE


def run(ODE):
    return run_lyap_spectrum(
        ODE, state0=[1.0, 1.0, 1.0],
        dt=1e-2, n_steps=50_000, renorm_every=10, t_transient=100.0,
    )


if __name__ == "__main__":
    first_s, warm_s, exponents = time_build_and_run(build, run)
    emit("lorenz_tier1.1", exponents, first_s, warm_s)
