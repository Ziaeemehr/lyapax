"""Audit of ``notes/open_issues.md`` item 6.1: differentiating a Lyapunov
exponent w.r.t. a system parameter through ``lyapax.core.lyapunov_spectrum``.

Unlike ``lyapax.adaptive``'s diffrax-backed integrator (which raises
``ValueError`` under reverse-mode AD, see ``tests/test_adaptive_ode.py``),
the fixed-step engine is built entirely from ``jax.lax.scan`` (static trip
count) and ``jnp.linalg.qr`` -- both support reverse-mode, so ``jax.grad``
does not raise here. But raising isn't the only way a gradient can be
useless: for a genuinely chaotic trajectory, ``jax.grad``/``jax.jacfwd``
differentiate through the *entire* unrolled state trajectory (``step_fn``
closes over the parameter, and ``_advance`` computes the primal state via
the same ``jax.jvp`` call used for tangent propagation), so the returned
"gradient" inherits the trajectory's own exponential sensitivity to its
initial condition -- it grows roughly like ``exp(lambda_max * horizon)``
and is numerically meaningless (not just noisy) well before it overflows.
This is a known phenomenon in chaotic sensitivity analysis (the reason
shadowing-based methods, e.g. least-squares shadowing, exist), not a
lyapax-specific bug -- these tests characterize it so it's a documented,
tested caveat rather than a silent trap.
"""
import jax
import jax.numpy as jnp
import numpy as np

from lyapax import lyapunov_spectrum, ode_problem, systems


def _linear_lambda_max(gamma, n_steps=2_000):
    """Top exponent of a 2D diagonal linear system is exactly ``gamma``
    (the less-negative eigenvalue) -- ``d(lambda_max)/d(gamma) == 1``
    exactly, a clean non-chaotic reference case."""
    def rhs(y):
        return jnp.array([[gamma, 0.0], [0.0, gamma - 1.0]]) @ y
    problem = ode_problem(rhs, state0=jnp.array([1.0, 0.3]), dt=1e-2, integrator="rk4")
    result = lyapunov_spectrum(problem, n_steps=n_steps, renorm_every=1, k=1, t_transient=5.0)
    return result.exponents[0]


def test_grad_matches_analytic_derivative_for_linear_system():
    gamma0 = 0.5
    grad = jax.grad(_linear_lambda_max)(gamma0)
    np.testing.assert_allclose(float(grad), 1.0, atol=1e-6)


def test_grad_matches_jacfwd_and_finite_difference_for_linear_system():
    gamma0 = 0.5
    grad_rev = jax.grad(_linear_lambda_max)(gamma0)
    grad_fwd = jax.jacfwd(_linear_lambda_max)(gamma0)

    eps = 1e-5
    finite_diff = (
        _linear_lambda_max(gamma0 + eps) - _linear_lambda_max(gamma0 - eps)
    ) / (2 * eps)

    assert abs(float(grad_rev) - float(finite_diff)) < 1e-6
    assert abs(float(grad_fwd) - float(finite_diff)) < 1e-6


def _lorenz_lambda_max(sigma, n_steps, t_transient=20.0):
    rhs = systems.lorenz(sigma=sigma, rho=28.0, beta=8.0 / 3.0)
    problem = ode_problem(rhs, state0=jnp.array([1.0, 1.0, 1.0]), dt=1e-2, integrator="rk4")
    result = lyapunov_spectrum(
        problem, n_steps=n_steps, renorm_every=5, k=1, t_transient=t_transient)
    return result.exponents[0]


def test_grad_diverges_exponentially_with_horizon_for_chaotic_system():
    """Not a bug: characterizes the blow-up. The naive gradient's magnitude
    grows by many orders of magnitude between a short and a 10x-longer
    Lorenz run, consistent with the exp(lambda_max * horizon) scaling
    chaotic sensitivity analysis predicts -- confirming this is a real
    property of the estimator, not test noise."""
    sigma0 = 10.0
    grad_short = jax.grad(lambda s: _lorenz_lambda_max(s, n_steps=300))(sigma0)
    grad_long = jax.grad(lambda s: _lorenz_lambda_max(s, n_steps=3_000))(sigma0)

    assert abs(float(grad_short)) > 1e2
    assert abs(float(grad_long)) > 1e8
    assert abs(float(grad_long)) > 1e4 * abs(float(grad_short))


def test_grad_and_jacfwd_agree_even_when_diverging():
    """Reverse- and forward-mode AD compute the same value here -- unlike
    the diffrax adaptive integrator, there is no reverse-mode-specific
    failure. The blow-up in the test above is the actual mathematical
    gradient of this finite-trajectory estimator, not an artifact of one
    AD mode."""
    sigma0 = 10.0
    n_steps = 1_000
    grad_rev = jax.grad(lambda s: _lorenz_lambda_max(s, n_steps=n_steps))(sigma0)
    grad_fwd = jax.jacfwd(lambda s: _lorenz_lambda_max(s, n_steps=n_steps))(sigma0)
    np.testing.assert_allclose(float(grad_rev), float(grad_fwd), rtol=1e-6)
