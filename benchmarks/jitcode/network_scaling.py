"""Tier 5: dense (all-to-all) Kuramoto network, d=50 only -- see
benchmarks/lyapax/network_scaling.py for the full d=50/200/1000/2000 sweep
and benchmarks/chaostools/network_scaling.jl for why ChaosTools.jl stops
at d=200.

d=50 is the largest size attempted here at all: building this network's
symbolic derivative/Jacobian (jitcode differentiates the whole
right-hand side symbolically before compiling to C) took 54 seconds by
itself in a direct test, before a single integration step -- and that
cost is driven by the ~2,450 nonzero coupling terms in a dense d=50
network, so it only gets worse at the sizes (d=200, 1000, 2000) the rest
of this tier actually cares about. Not attempted here.
"""
from _common import emit, run_lyap_spectrum, time_build_and_run
from jitcode import jitcode_lyap, y
from symengine import sin

N = 50


def build():
    omega = [-1.0 + 2.0 * i / (N - 1) for i in range(N)]
    f = [omega[i] + sum(sin(y(j) - y(i)) for j in range(N) if j != i) for i in range(N)]
    ODE = jitcode_lyap(f, n_lyap=5)
    ODE.set_integrator("dopri5")
    return ODE


def run(ODE):
    import numpy as np
    state0 = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    return run_lyap_spectrum(
        ODE, state0=state0, dt=1e-2, n_steps=200, renorm_every=10, t_transient=0.0,
    )


if __name__ == "__main__":
    first_s, warm_s, exponents = time_build_and_run(build, run)
    emit(f"kuramoto_scaling_d{N}", exponents, first_s, warm_s)
