"""Tier 3.1: 4-node linear network, 4-cycle graph. Same params as
benchmarks/lyapax/network.py. Expressed as a flat ODE, dx_i/dt = gamma*x_i +
G*sum_j W_ij*x_j, since jitcode has no network/coupling abstraction of its
own.
"""
from jitcode import jitcode_lyap, y

from _common import time_build_and_run, run_lyap_spectrum, emit


def build():
    gamma, G = -2.0, 0.5
    # 4-cycle adjacency: node i coupled to i-1 and i+1 (mod 4).
    f = [
        gamma * y(0) + G * (y(1) + y(3)),
        gamma * y(1) + G * (y(0) + y(2)),
        gamma * y(2) + G * (y(1) + y(3)),
        gamma * y(3) + G * (y(0) + y(2)),
    ]
    ODE = jitcode_lyap(f, n_lyap=4)
    ODE.set_integrator("dopri5")
    return ODE


def run(ODE):
    return run_lyap_spectrum(
        ODE, state0=[0.3, -0.1, 0.2, -0.4],
        dt=1e-3, n_steps=20_000, renorm_every=10, t_transient=5.0,
    )


if __name__ == "__main__":
    first_s, warm_s, exponents = time_build_and_run(build, run)
    emit("linear_network_tier3.1", exponents, first_s, warm_s)
