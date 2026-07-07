"""lyapax — JAX-native Lyapunov exponent computation for ODEs and DDEs.

Benettin/QR tangent-propagation engine for differentiable JAX maps
(``lyapunov_spectrum``) and fixed-delay ring-buffer DDEs
(``lyapunov_spectrum_dde``). See :doc:`/background/lyapax_implementation`
for how the engine works and :doc:`/background/validation` for the
correctness checks this package is held to.
"""
from __future__ import annotations

from . import coupling
from .__version__ import __version__
from .core import LyapunovResult, ODEProblem, lyapunov_spectrum, ode_problem
from .dde import DDEProblem, dde_problem, lyapunov_spectrum_dde, network_dde_problem
from .integrators import euler_step, get_integrator, heun_step, ode_step, rk4_step, rk6_step
from .network import Network, network_problem, network_step, network_step_parametrized

__all__ = [
    "__version__",
    "LyapunovResult",
    "lyapunov_spectrum",
    "lyapunov_spectrum_dde",
    "ODEProblem",
    "ode_problem",
    "network_problem",
    "DDEProblem",
    "dde_problem",
    "network_dde_problem",
    "Network",
    "network_step",
    "network_step_parametrized",
    "coupling",
    "euler_step",
    "heun_step",
    "rk4_step",
    "rk6_step",
    "ode_step",
    "get_integrator",
]
