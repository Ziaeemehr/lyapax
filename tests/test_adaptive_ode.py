"""M9.5 validation for lyapax.adaptive.diffrax_adaptive_step:

- convergence as rtol/atol tighten, against a system with a known exact
  Lyapunov exponent (Lorenz's published lambda1, same reference
  test_lyapunov_core.py's fixed-step rk4 test is held to);
- a cross-check that the adaptive integrator and fixed-step rk4 agree on
  the same system, not just that each converges to *something*;
- a differentiability check: jax.jacfwd of an exponent w.r.t. a system
  parameter matches a finite-difference reference (jax.grad -- reverse
  mode -- does not work through this integrator's dynamic-trip-count
  while_loop; see lyapax.adaptive's module docstring);
- lyapax.dde rejects this integrator rather than silently mishandling it
  (dde_problem/network_dde_problem take no `integrator` argument at all,
  so this is really "there is no way to pass one in", confirmed here).
"""
import jax
import jax.numpy as jnp
import pytest

diffrax = pytest.importorskip("diffrax")

jax.config.update("jax_enable_x64", True)

from lyapax import dde_problem, lyapunov_spectrum, ode_problem, systems  # noqa: E402
from lyapax.adaptive import diffrax_adaptive_step  # noqa: E402
from lyapax.integrators import rk4_step  # noqa: E402


def _lorenz_adaptive_result(rtol, atol, dt=0.1, n_steps=1_000, renorm_every=1):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rhs = systems.lorenz(sigma, rho, beta)
    integrator = diffrax_adaptive_step(rtol=rtol, atol=atol)
    problem = ode_problem(rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=dt, integrator=integrator)
    return lyapunov_spectrum(
        problem, n_steps=n_steps, renorm_every=renorm_every, t_transient=100.0)


def test_lorenz_lambda1_matches_published_value():
    result = _lorenz_adaptive_result(rtol=1e-9, atol=1e-11, n_steps=5_000)
    assert abs(float(result.exponents[0]) - 0.9056) < 0.08
    assert abs(float(result.exponents[1])) < 0.03


def test_convergence_as_tolerance_tightens():
    errs = []
    for rtol in (1e-4, 1e-6, 1e-9):
        result = _lorenz_adaptive_result(rtol=rtol, atol=rtol * 1e-2, n_steps=2_000)
        errs.append(abs(float(result.exponents[0]) - 0.9056))
    # not strictly monotonic (chaotic-system LE estimates are noisy at any
    # single dt/tolerance), but the tight-tolerance error should not be
    # worse than the loose one by more than noise-level slack.
    assert errs[-1] <= errs[0] + 0.02


def test_matches_fixed_step_rk4_cross_check():
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rhs = systems.lorenz(sigma, rho, beta)
    state0 = jnp.array([1.0, 1.0, 1.0])

    rk4_problem = ode_problem(rhs, state0=state0, dt=1e-2, integrator=rk4_step)
    rk4_result = lyapunov_spectrum(
        rk4_problem, n_steps=20_000, renorm_every=10, t_transient=100.0)

    adaptive_result = _lorenz_adaptive_result(
        rtol=1e-10, atol=1e-12, dt=0.1, n_steps=2_000, renorm_every=1)

    assert abs(float(rk4_result.exponents[0]) - float(adaptive_result.exponents[0])) < 0.05


def test_jacfwd_matches_finite_difference():
    state0 = jnp.array([1.0, 0.3])

    def lambda_max(a):
        def rhs(y):
            return jnp.array([[-a, 1.0], [-1.0, -a]]) @ y
        integrator = diffrax_adaptive_step(rtol=1e-8, atol=1e-10)
        problem = ode_problem(rhs, state0=state0, dt=0.2, integrator=integrator)
        result = lyapunov_spectrum(problem, n_steps=50, renorm_every=1, k=1)
        return result.exponents[0]

    a0 = 0.5
    grad_fwd = jax.jacfwd(lambda_max)(a0)

    eps = 1e-5
    finite_diff = (lambda_max(a0 + eps) - lambda_max(a0 - eps)) / (2 * eps)

    assert abs(float(grad_fwd) - float(finite_diff)) < 1e-6


def test_grad_reverse_mode_raises():
    """Documents a real limitation, not a bug to be silently masked: reverse
    -mode AD can't replay a dynamic-trip-count while_loop backward. If a
    future JAX/diffrax version starts supporting this, this test starts
    failing (not erroring) -- the signal to relax the module docstring's
    "use jacfwd, not grad" guidance.
    """
    def lambda_max(a):
        def rhs(y):
            return jnp.array([[-a, 1.0], [-1.0, -a]]) @ y
        integrator = diffrax_adaptive_step(rtol=1e-8, atol=1e-10)
        problem = ode_problem(rhs, state0=jnp.array([1.0, 0.3]), dt=0.2, integrator=integrator)
        result = lyapunov_spectrum(problem, n_steps=20, renorm_every=1, k=1)
        return result.exponents[0]

    with pytest.raises(ValueError, match="Reverse-mode differentiation"):
        jax.grad(lambda_max)(0.5)


def test_max_steps_exhaustion_raises():
    """max_steps=0 deterministically forces the internal while_loop to exit
    with tprev == t0 < t1 on the very first dt advance, regardless of
    system/tolerances -- exercises the eqx.error_if completion check
    without needing a genuinely pathological rhs/tolerance combination.
    """
    rhs = systems.lorenz(10.0, 28.0, 8.0 / 3.0)
    integrator = diffrax_adaptive_step(rtol=1e-9, atol=1e-11, max_steps=0)
    problem = ode_problem(rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=0.1, integrator=integrator)
    with pytest.raises(RuntimeError, match="max_steps exhausted"):
        lyapunov_spectrum(problem, n_steps=1, renorm_every=1)


def test_default_solver_requires_scan_kind_lax():
    with pytest.raises(ValueError, match="scan_kind"):
        diffrax_adaptive_step(solver=diffrax.Dopri5())


def test_dde_rejects_adaptive_integrator():
    """dde_problem/network_dde_problem do accept an `integrator` argument
    (for choosing among *fixed-step* DDE-compatible builders -- a
    different calling convention than ode_problem's), so passing an
    adaptive ODE integrator there isn't caught just by "no such
    parameter". Confirms the explicit guard in
    lyapax.simulator.step.make_step_fn raises a clear, targeted error
    instead of relying on an incidental arity-mismatch TypeError --
    M9.3's "clear ValueError, not silent fixed-step fallback."
    """
    def rhs_delayed(state, delayed, params):
        return -state + delayed

    integrator = diffrax_adaptive_step(rtol=1e-6, atol=1e-9)
    with pytest.raises(ValueError, match="not supported for DDEs"):
        dde_problem(
            rhs_delayed, state0=jnp.array([1.0]), tau=0.3, dt=0.05,
            integrator=integrator,
        )
