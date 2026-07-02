"""Trimmed model-specification dataclasses.

Adapted from vbi/simulator/spec/model.py — see NOTICE.md in this directory
for provenance and what was dropped.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateVar:
    name: str
    default_init: float = 0.0


@dataclass(frozen=True)
class Parameter:
    name: str
    default: float


@dataclass(frozen=True)
class ModelSpec:
    """
    Backend-agnostic description of a dynamical system's right-hand side.

    dfun_str maps each state variable name to a bare math expression string.
    Only these symbols may appear: state variable names, parameter names,
    'c' (coupling input, scalar per node — zero for uncoupled/single-node
    systems), and the math functions: exp, log, sin, cos, tanh, sqrt, abs, pi.
    No 'np.' / 'jnp.' prefix — the code generator injects the namespace.
    """
    name: str
    state_variables: tuple[StateVar, ...]
    parameters: tuple[Parameter, ...]
    cvar: tuple[str, ...]
    dfun_str: dict[str, str]

    @property
    def sv_names(self) -> tuple[str, ...]:
        return tuple(sv.name for sv in self.state_variables)

    @property
    def n_sv(self) -> int:
        return len(self.state_variables)

    @property
    def cvar_indices(self) -> tuple[int, ...]:
        names = self.sv_names
        return tuple(names.index(c) for c in self.cvar)

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def default_params(self) -> dict[str, float]:
        return {p.name: p.default for p in self.parameters}
