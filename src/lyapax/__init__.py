"""lyapax — JAX-native Lyapunov exponent computation for ODEs and DDEs.

Benettin/QR tangent-propagation engine for differentiable JAX maps
(``lyapunov_spectrum``) and fixed-delay ring-buffer DDEs
(``lyapunov_spectrum_dde``). See notes/milestones.md for the design
history and notes/validation_systems.md for the correctness tests this
package is held to.
"""
from __future__ import annotations

from .core import LyapunovResult, lyapunov_spectrum
from .dde import lyapunov_spectrum_dde
from .integrators import euler_step, rk4_step

__version__ = "0.0.1"

__all__ = [
    "LyapunovResult",
    "lyapunov_spectrum",
    "lyapunov_spectrum_dde",
    "euler_step",
    "rk4_step",
]
