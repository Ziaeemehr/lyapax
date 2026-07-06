"""lyapax — JAX-native Lyapunov exponent computation for ODEs and DDEs.

Benettin/QR tangent-propagation engine for differentiable JAX maps
(``lyapunov_spectrum``) and fixed-delay ring-buffer DDEs
(``lyapunov_spectrum_dde``). See notes/milestones.md for the design
history and notes/validation_systems.md for the correctness tests this
package is held to.
"""
from __future__ import annotations

from . import coupling
from .core import ODEProblem, LyapunovResult, lyapunov_spectrum, ode_problem
from .dde import DDEProblem, dde_problem, lyapunov_spectrum_dde, network_dde_problem
from .integrators import euler_step, get_integrator, heun_step, ode_step, rk4_step
from .network import Network, network_problem, network_step, network_step_parametrized

__version__ = "0.0.1"

__all__ = [
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
    "ode_step",
    "get_integrator",
]
