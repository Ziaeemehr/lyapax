"""Minimal, trimmed copy of vbi's model/coupling/step machinery.

See NOTICE.md for provenance. Only what M1+ needs to build the Lyapunov
tangent-propagation layer on top of; not a general-purpose simulation
library.
"""
from .coupling import Connectivity
from .model_spec import ModelSpec, Parameter, StateVar
from .step import build_jax_dfun, make_step_fn

__all__ = [
    "ModelSpec", "StateVar", "Parameter",
    "Connectivity",
    "build_jax_dfun", "make_step_fn",
]
