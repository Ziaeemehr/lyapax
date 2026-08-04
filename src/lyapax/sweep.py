"""Batched parameter/initial-condition sweeps via ``jax.vmap``.

Mirrors vbi's ``JaxSweeper`` pattern (batch a simulation over a grid of
parameter values) without reusing its code, per the vendoring decision.
The surface needed here is much smaller than a
general sweeper: ``lyapax.simulator.make_step_fn``'s carry already threads
``params`` through as data rather than closing over it (see that function's
docstring), so a parameter
sweep is just ``jax.vmap`` applied to a function that calls
``lyapax.core.lyapunov_spectrum`` once, with ``params`` (and optionally
``state0``) as the vmapped argument. No new tangent-propagation or
QR machinery -- this is a batching wrapper around the existing engine, not
a third one.

The one piece of the network path that still closed over ``params`` at
construction time was ``lyapax.network.make_network_step_fn`` (a thin
adapter, not the tangent engine); ``make_parametrized_network_step_fn``
(in that module) is the call-time-``params`` sibling this needs.
"""
from __future__ import annotations

import warnings
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .core import LyapunovResult, _n_transient_steps, lyapunov_spectrum
from .utils import simulate_trajectory

StepFlatParametrized = Callable[[jnp.ndarray, dict], jnp.ndarray]

_DEFAULT_TRAJECTORY_WARN_THRESHOLD_BYTES = 1 << 30  # 1 GiB


def _warn_if_trajectory_too_large(
        n_sweep: int, n_saves: int, d: int, itemsize: int,
        batch_size: int | None,
        threshold_bytes: int = _DEFAULT_TRAJECTORY_WARN_THRESHOLD_BYTES,
) -> None:
    """Warn (not raise -- the sweep still runs) when an unchunked
    ``save_states=True`` call would materialize an estimated ``nbytes``
    trajectory on-device all at once. Only fires when ``batch_size is
    None``: chunking already bounds peak memory to one chunk, so there is
    nothing to warn about once it's given."""
    if batch_size is not None:
        return
    nbytes = n_sweep * n_saves * d * itemsize
    if nbytes > threshold_bytes:
        warnings.warn(
            f"sweep_lyapunov_spectrum: save_states=True with batch_size=None "
            f"would materialize an estimated {nbytes / 1e9:.2f} GB "
            f"trajectory ({n_sweep} x {n_saves} x {d} x {itemsize} bytes) "
            "on-device at once. Pass batch_size to run the sweep in "
            "memory-bounded chunks (optionally with trajectory_to_host=True "
            "to park finished chunks on host), and/or increase decimate to "
            "thin the saved time series.",
            stacklevel=2,
        )


def _iter_chunks(n_sweep: int, batch_size: int | None):
    """Yield ``(start, stop)`` index pairs covering ``[0, n_sweep)``: one
    pair for the whole grid when ``batch_size is None``, otherwise
    ``ceil(n_sweep / batch_size)`` pairs (the last possibly ragged)."""
    if batch_size is None:
        yield 0, n_sweep
        return
    for start in range(0, n_sweep, batch_size):
        yield start, min(start + batch_size, n_sweep)


def _concat_chunks(chunks: list):
    """Concatenate a list of same-structure pytrees along their leading
    (batch) axis, per leaf. A ``numpy`` leaf (parked on host via
    ``trajectory_to_host``) concatenates with ``numpy``, keeping it off
    the device; any other leaf concatenates with ``jnp.concatenate``.
    Single-chunk input (the unchunked, ``batch_size=None`` case) is
    returned as-is, with no concatenation call at all."""
    if len(chunks) == 1:
        return chunks[0]

    def _cat(*xs):
        if isinstance(xs[0], np.ndarray):
            return np.concatenate(xs, axis=0)
        return jnp.concatenate(xs, axis=0)

    return jax.tree.map(_cat, *chunks)


def sweep_lyapunov_spectrum(
        step_fn: StepFlatParametrized,
        state0: jnp.ndarray,
        params_batch: dict,
        dt: float,
        n_steps: int,
        k: int | None = None,
        renorm_every: int = 1,
        t_transient: float = 0.0,
        seed: int = 0,
        state0_batch: jnp.ndarray | None = None,
        save_states: bool = False,
        decimate: int = 1,
        batch_size: int | None = None,
        trajectory_to_host: bool = False,
        _trajectory_warn_threshold_bytes: int = _DEFAULT_TRAJECTORY_WARN_THRESHOLD_BYTES,
) -> LyapunovResult:
    """
    Batched ``lyapunov_spectrum`` over a grid of parameter values (and,
    optionally, a matching grid of initial conditions) via one ``jax.vmap``
    call, rather than a Python loop making one call per grid point (as in
    e.g. ``examples/05_kuramoto_sync.py``/``09_kuramoto_delayed_network.py``).

    step_fn : ``(state, params) -> new_state`` -- ``params`` must be a
        call-time argument, *not* closed over. See
        ``lyapax.network.make_parametrized_network_step_fn`` for the
        network-coupling case; a hand-written function with this signature
        works too (same "plain callable, no registry" extension point as
        ``lyapax.coupling``).
    state0 : (d,) initial state, shared across the whole sweep, unless
        ``state0_batch`` is given.
    params_batch : pytree matching one ``params`` dict's structure, but
        every leaf carries an extra leading batch axis of size
        ``n_sweep`` -- e.g. ``{"G": jnp.linspace(0, 4, 13), "omega":
        jnp.broadcast_to(omega, (13, n_nodes))}`` to sweep ``G`` at fixed
        ``omega``.
    dt, n_steps, k, renorm_every, t_transient, seed : see
        ``lyapax.core.lyapunov_spectrum`` - shared (not swept) across the
        whole grid.
    state0_batch : optional ``(n_sweep, d)`` initial conditions, one row
        per grid point, if the initial condition should vary alongside
        ``params_batch`` rather than staying fixed at ``state0``.
    save_states, decimate : see ``lyapax.core.lyapunov_spectrum`` (same
        meaning and validation, shared across the whole grid). When
        ``save_states=True``, ``result.trajectory``/``result.save_times``
        gain the batch axis like every other field:
        ``result.trajectory.shape == (n_sweep, n_saves, d)``. **Mind
        memory** -- this is ``n_sweep * n_saves * d`` floats held at once;
        use ``decimate``, and/or ``batch_size``/``trajectory_to_host``
        below, before sweeping a wide grid at fine time resolution. An
        unchunked (``batch_size=None``) call estimated above ~1 GiB warns,
        naming ``batch_size``/``decimate`` as the remedies -- it still
        runs, this is advisory only.
    batch_size : if given, run the grid in chunks of this many points
        (the last chunk ragged if ``batch_size`` doesn't divide
        ``n_sweep``) instead of one ``jax.vmap`` call over the whole
        grid. Every field is bitwise identical to the unchunked call --
        chunking only changes how much is materialized on-device at
        once, peak ``batch_size * n_saves * d`` instead of
        ``n_sweep * n_saves * d`` for ``trajectory``. ``None`` (the
        default): one unchunked ``jax.vmap`` call, as before this
        parameter existed.
    trajectory_to_host : if True, pull each chunk's ``trajectory`` to host
        (``numpy``, via ``jax.device_get``) before the next chunk starts,
        so the full-sweep trajectory never needs to fit in device memory
        at once. Only meaningful together with ``batch_size`` (with
        ``batch_size=None`` there is only one chunk, already fully
        materialized on-device before this could help) and
        ``save_states=True``. Every *other* field -- ``exponents``,
        ``history``, ``times``, ``checkpoint`` -- stays a normal JAX
        (device) array regardless; only ``trajectory`` moves.

    Returns
    -------
    LyapunovResult, every field with an extra leading batch axis of size
    ``n_sweep`` -- e.g. ``result.exponents.shape == (n_sweep, k)``.

    Examples
    --------
    Sweep the coupling strength ``G`` of a Kuramoto network in one
    ``jax.vmap`` call instead of a Python loop over ``lyapunov_spectrum``:

    >>> import jax
    >>> jax.config.update("jax_enable_x64", True)
    >>> import jax.numpy as jnp
    >>> from lyapax.coupling import kuramoto_coupling
    >>> from lyapax.network import Network, network_step_parametrized
    >>> from lyapax.simulator import ModelSpec, StateVar, Parameter, build_jax_dfun
    >>> from lyapax.sweep import sweep_lyapunov_spectrum
    >>> n_nodes = 6
    >>> omega = jnp.linspace(-1.0, 1.0, n_nodes)
    >>> weights = jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)
    >>> model = ModelSpec(
    ...     name="kuramoto",
    ...     state_variables=(StateVar("theta", default_init=0.0),),
    ...     parameters=(Parameter("omega", 0.0),),
    ...     cvar=("theta",),
    ...     dfun_str={"theta": "omega + c"},
    ... )
    >>> dfun = build_jax_dfun(model)
    >>> network = Network(weights=weights, cvar_indices=model.cvar_indices)
    >>> dt = 1e-2
    >>> state0 = jnp.linspace(0.0, 2 * jnp.pi, n_nodes, endpoint=False)
    >>> G_values = jnp.linspace(0.0, 4.0, 13)
    >>> step_p = network_step_parametrized(dfun, network, kuramoto_coupling(alpha=0.0), dt)
    >>> params_batch = {"omega": jnp.broadcast_to(omega, (13, n_nodes)), "G": G_values}
    >>> result = sweep_lyapunov_spectrum(
    ...     step_p, state0, params_batch, dt=dt, n_steps=5_000,
    ...     renorm_every=10, k=2, t_transient=50.0,
    ... )
    >>> result.exponents.shape  # doctest: +SKIP
    (13, 2)
    """
    def _run_one(params, state0_i):
        return lyapunov_spectrum(
            lambda state: step_fn(state, params), state0_i, dt, n_steps,
            k=k, renorm_every=renorm_every, t_transient=t_transient, seed=seed,
            save_states=save_states, decimate=decimate,
        )

    in_axes = (0, None) if state0_batch is None else (0, 0)
    n_sweep = jax.tree.leaves(params_batch)[0].shape[0]

    if save_states:
        state0_arr = jnp.asarray(state0)
        _warn_if_trajectory_too_large(
            n_sweep=n_sweep, n_saves=n_steps // decimate,
            d=int(np.prod(state0_arr.shape)), itemsize=state0_arr.dtype.itemsize,
            batch_size=batch_size,
            threshold_bytes=_trajectory_warn_threshold_bytes,
        )

    chunks = []
    for start, stop in _iter_chunks(n_sweep, batch_size):
        chunk_params = jax.tree.map(lambda x: x[start:stop], params_batch)
        chunk_state0 = state0 if state0_batch is None else state0_batch[start:stop]
        chunk_result = jax.vmap(_run_one, in_axes=in_axes)(chunk_params, chunk_state0)
        if trajectory_to_host and chunk_result.trajectory is not None:
            chunk_result = chunk_result._replace(
                trajectory=jax.device_get(chunk_result.trajectory))
        chunks.append(chunk_result)
    return _concat_chunks(chunks)


def sweep_trajectory(
        step_fn: StepFlatParametrized,
        state0: jnp.ndarray,
        params_batch: dict,
        dt: float,
        n_steps: int,
        t_transient: float = 0.0,
        decimate: int = 1,
        renorm_every: int = 1,
        state0_batch: jnp.ndarray | None = None,
        batch_size: int | None = None,
        trajectory_to_host: bool = False,
):
    """
    Batched state time series over a grid of parameter values, via
    ``lyapax.utils.simulate_trajectory`` and ``jax.vmap`` -- the
    ``sweep_lyapunov_spectrum(..., save_states=True)`` trajectory, without
    the O(k) tangent propagation/QR cost of actually computing exponents.
    Use this when only the dynamics are wanted (e.g. to derive features
    from the time series downstream), and
    ``sweep_lyapunov_spectrum(..., save_states=True)`` when the trajectory
    is wanted *alongside* the exponents from the same run.

    Lands on exactly the same post-transient window and decimation stride
    as ``sweep_lyapunov_spectrum(..., save_states=True, decimate=decimate)``
    would for the same ``t_transient``/``renorm_every``/``decimate`` (see
    the cross-check in ``tests/test_sweep.py``) -- the transient-length
    rounding is shared with the LE engine via
    ``lyapax.core._n_transient_steps`` rather than re-derived here.

    step_fn, state0, params_batch, state0_batch : see
        ``sweep_lyapunov_spectrum`` (same meaning; ``step_fn`` need not be
        differentiable here, since no tangent propagation happens).
    dt, n_steps, t_transient, renorm_every : see
        ``lyapax.core.lyapunov_spectrum`` -- ``n_steps`` is the
        post-transient step count, matching ``sweep_lyapunov_spectrum``'s
        meaning; ``renorm_every`` only affects the transient-length
        rounding (there is no renormalization here), so it must match
        the LE call's value for the two to align.
    decimate : keep every this many steps (must divide ``n_steps``).
        ``decimate=1`` (the default) keeps every step.
    batch_size, trajectory_to_host : see ``sweep_lyapunov_spectrum`` (same
        meaning and the same memory tradeoff).

    Returns
    -------
    ``(t, trajectory)`` -- ``t``: ``(n_sweep, n_saves)``; ``trajectory``:
    ``(n_sweep, n_saves, d)``, or ``d``'s ``sweep_lyapunov_spectrum``
    network-state shape if ``step_fn`` produces one. ``n_saves = n_steps
    // decimate``. Row 0 is the state ``decimate`` steps after the
    transient (not the post-transient state itself), same convention as
    ``lyapax.core.LyapunovResult.trajectory``.
    """
    n_transient = _n_transient_steps(t_transient, dt, renorm_every) if t_transient > 0.0 else 0

    def _run_one(params, state0_i):
        def step(state):
            return step_fn(state, params)

        return simulate_trajectory(
            step, state0_i, n_steps, dt=dt, decimate=decimate,
            n_transient=n_transient,
        )

    in_axes = (0, None) if state0_batch is None else (0, 0)
    n_sweep = jax.tree.leaves(params_batch)[0].shape[0]

    chunks = []
    for start, stop in _iter_chunks(n_sweep, batch_size):
        chunk_params = jax.tree.map(lambda x: x[start:stop], params_batch)
        chunk_state0 = state0 if state0_batch is None else state0_batch[start:stop]
        t_chunk, traj_chunk = jax.vmap(_run_one, in_axes=in_axes)(chunk_params, chunk_state0)
        if trajectory_to_host:
            traj_chunk = jax.device_get(traj_chunk)
        chunks.append((t_chunk, traj_chunk))
    t, trajectory = _concat_chunks(chunks)
    # Drop the leading (post-transient-state-itself) row that
    # simulate_trajectory always includes -- sweep_lyapunov_spectrum's
    # trajectory does not have this row (its first row is one decimate
    # stride *past* the transient), so this alignment is what makes the
    # two paths comparable/cross-checkable.
    return t[:, 1:], trajectory[:, 1:]
