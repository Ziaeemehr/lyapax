"""Network coupling as plain callables.

Design note: coupling here is deliberately *not*
a closed enum dispatched through hardcoded if/elif branches, unlike vbi's
``CouplingSpec.kind`` + ``build_coupling`` (duplicated across its numpy and
JAX backends, with ``jr_sigmoidal`` even special-cased to skip the ``G``
multiply). Each builder below just returns a plain function::

    coupling_fn(cvar_state, weights, params) -> coupling

that ``lyapax.network.make_network_step_fn`` calls directly. A user's own
function with that exact signature is a first-class coupling - no
registry, no "kind" string to add to a dispatch table, no library changes.

:param cvar_state: ``(n_cvar, n_nodes)`` coupling-variable state -- the
    instantaneous state for zero-delay networks, a delayed array of the
    same shape for DDE networks.
:param weights: ``(n_nodes, n_nodes)`` weights[tgt, src].
:param params: dict - the global coupling strength ``G`` is read from here
    (``params.get("G", G_default)``), the same convention the vendored
    step function uses, so ``G`` can be swept or differentiated
    without being baked into a closure constant.
:returns: ``(n_cvar, n_nodes)``
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

CouplingFn = Callable[[jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]


def linear_coupling(a: float = 1.0, b: float = 0.0, G_default: float = 1.0) -> CouplingFn:
    """c[cvar, tgt] = G * a * sum_src(w[tgt, src] * x[cvar, src]) + b."""
    def coupling_fn(cvar_state: jnp.ndarray, weights: jnp.ndarray, params: dict) -> jnp.ndarray:
        G = params.get("G", G_default)
        return G * a * jnp.einsum("ts,cs->ct", weights, cvar_state) + b
    return coupling_fn


def sigmoidal_coupling(
        a: float = 1.0, b: float = 0.0,
        midpoint: float = 0.0, sigma: float = 1.0,
        G_default: float = 1.0,
) -> CouplingFn:
    """c[cvar, tgt] = G * a * sum_src(w[tgt, src] * sigm(x[cvar, src])) + b,
    sigm(x) = 1 / (1 + exp(-(x - midpoint) / sigma))."""
    def coupling_fn(cvar_state: jnp.ndarray, weights: jnp.ndarray, params: dict) -> jnp.ndarray:
        G = params.get("G", G_default)
        sigm = 1.0 / (1.0 + jnp.exp(-(cvar_state - midpoint) / sigma))
        return G * a * jnp.einsum("ts,cs->ct", weights, sigm) + b
    return coupling_fn


def kuramoto_coupling(alpha: float = 0.0, G_default: float = 1.0) -> CouplingFn:
    """c[0, tgt] = (G/N) * sum_src w[tgt, src] * sin(theta_src - theta_tgt + alpha).

    Requires exactly one coupling variable (the phase); ``cvar_state[0]``
    is taken as ``theta``.
    """
    def coupling_fn(cvar_state: jnp.ndarray, weights: jnp.ndarray, params: dict) -> jnp.ndarray:
        G = params.get("G", G_default)
        n_nodes = weights.shape[0]
        theta = cvar_state[0]                                  # (n_nodes,)
        diff = theta[None, :] - theta[:, None] + alpha          # diff[tgt, src]
        c = (G / n_nodes) * jnp.sum(weights * jnp.sin(diff), axis=1)
        return c[None, :]
    return coupling_fn
