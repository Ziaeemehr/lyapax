"""Trimmed connectivity dataclass.

Adapted from vbi/simulator/spec/connectivity.py - see NOTICE.md in this
directory for provenance and what was dropped. ``vbi``'s ``CouplingSpec``
(a fixed-``kind`` enum) is *not* vendored here - lyapax's coupling is a
plain callable instead (see ``lyapax.coupling``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Connectivity:
    """
    Structural connectivity: weights + optional per-edge delay (tract length
    / speed). ``tract_lengths is None`` (or all-zero) means a pure ODE
    (zero-delay) network; any non-zero entry makes it a DDE network.
    """
    weights: np.ndarray
    tract_lengths: np.ndarray | None = None
    speed: float = 1.0

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64)
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError(f"weights must be square 2-D; got shape {w.shape}.")
        self.weights = w

        if self.tract_lengths is None:
            self.tract_lengths = np.zeros_like(w)
        else:
            tl = np.asarray(self.tract_lengths, dtype=np.float64)
            if tl.shape != w.shape:
                raise ValueError(
                    f"tract_lengths shape {tl.shape} must match weights shape {w.shape}."
                )
            if np.any(tl < 0):
                raise ValueError("tract_lengths must be non-negative.")
            self.tract_lengths = tl

        if self.speed <= 0:
            raise ValueError(f"speed must be > 0; got {self.speed!r}.")

    @property
    def n_nodes(self) -> int:
        return self.weights.shape[0]

    @property
    def has_delays(self) -> bool:
        return bool(np.any(self.tract_lengths > 0))

    def delay_steps(self, dt: float) -> np.ndarray:
        """(n, n) int32 array of delay expressed in integration steps.

        Integer-step only (no sub-step interpolation) - see
        :ref:`dde-history-interpolation` for the accuracy tradeoff this
        implies.
        """
        raw = self.tract_lengths / (self.speed * dt)
        return np.round(raw).astype(np.int32)

    def horizon(self, dt: float) -> int:
        """Ring-buffer depth = max delay (in steps) + 1."""
        d = self.delay_steps(dt)
        return int(d.max()) + 1 if d.size > 0 else 1
