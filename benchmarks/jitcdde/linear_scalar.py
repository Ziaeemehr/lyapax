"""Tier 4.2: linear scalar DDE, x'(t) = -a*x(t-tau). Same params as
benchmarks/lyapax/linear_scalar_dde.py.
"""
from _common import emit, run_lyap_spectrum, time_build_and_run
from jitcdde import jitcdde_lyap, t, y


def build():
    a, tau = 0.5, 0.3
    f = [-a * y(0, t - tau)]
    DDE = jitcdde_lyap(f, n_lyap=1)
    return DDE


def run(DDE):
    return run_lyap_spectrum(
        DDE, past_value=[1.0],
        dt=1e-2, n_steps=20_000, renorm_every=5, t_transient=10.0,
    )


if __name__ == "__main__":
    first_s, warm_s, exponents = time_build_and_run(build, run)
    emit("linear_scalar_dde_tier4.2", exponents, first_s, warm_s)
