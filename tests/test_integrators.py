"""Empirical convergence-order checks for lyapax.integrators.

Each fixed-step integrator's nominal order p should show local/global error
shrinking by roughly 2^p when dt is halved. This is checked against a
genuinely nonlinear autonomous ODE with a known closed-form solution
(y' = -y*log(y), y(t) = y0**exp(-t)) rather than a linear or polynomial
system: a linear scalar test only exercises the "single chain" elementary
differentials (RK order conditions collapse for it), and a polynomial rhs
has vanishing higher derivatives that can mask a wrong high-order tableau
-- see the design note on rk6_combine in lyapax/integrators.py for the
concrete case (a red-herring 9-stage weight set that passed those weaker
checks but was empirically order 5, not 6) this test class is meant to
catch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lyapax.integrators import ode_step

jax.config.update("jax_enable_x64", True)

Y0 = 1.5
T = 0.6


def _rhs(y):
    return -y * jnp.log(y)


def _exact(t):
    return Y0 ** np.exp(-t)


def _empirical_order(integrator: str, ns: tuple[int, int]) -> float:
    errs = []
    for n in ns:
        dt = T / n
        step = ode_step(_rhs, dt=dt, integrator=integrator)
        y = jnp.array([Y0])
        for _ in range(n):
            y = step(y)
        errs.append(abs(float(y[0]) - _exact(T)))
    return float(np.log(errs[0] / errs[1]) / np.log(ns[1] / ns[0]))


@pytest.mark.parametrize(
    "integrator,ns,expected_order",
    [
        ("euler", (16, 32), 1),
        ("heun", (16, 32), 2),
        ("rk4", (4, 8), 4),
        ("rk6", (4, 8), 6),
    ],
)
def test_integrator_matches_nominal_convergence_order(integrator, ns, expected_order):
    order = _empirical_order(integrator, ns)
    # Loose tolerance: dt isn't asymptotically tiny (avoiding float64
    # roundoff swamping the true truncation error for rk6, whose errors are
    # already ~1e-11 at these dt), so the measured order is a bit noisy --
    # this only needs to distinguish order p from p-1 or p+1, not pin it
    # down precisely.
    assert order == pytest.approx(expected_order, abs=0.75)
