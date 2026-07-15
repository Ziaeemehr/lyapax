"""Audit of notes/open_issues.md item 6.2: Kaplan-Yorke (Lyapunov) dimension
as a public post-processing helper (Kaplan & Yorke, 1979). Replaces two
byte-for-byte identical private ``_kaplan_yorke_dimension`` helpers that
previously lived in tests/test_dde.py and benchmarks/lyapax/mackey_glass.py.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from lyapax.core import kaplan_yorke_dimension


def test_hand_checked_crossing_within_spectrum():
    # cumsum = [1.0, -2.0] -- crosses zero after index 1.
    # D_KY = 1 + 1.0 / |-3.0| = 1.3333...
    ky = kaplan_yorke_dimension(jnp.array([1.0, -3.0]))
    np.testing.assert_allclose(float(ky), 1.0 + 1.0 / 3.0, atol=1e-10)


def test_all_negative_gives_dimension_zero():
    ky = kaplan_yorke_dimension(jnp.array([-1.0, -2.0, -3.0]))
    assert float(ky) == 0.0


def test_full_spectrum_never_negative_gives_full_dimension():
    # cumsum = [1.0, 0.0] -- never goes negative, exactly the full (2D)
    # spectrum was given, so the attractor fills the whole tracked space.
    ky = kaplan_yorke_dimension(jnp.array([1.0, -1.0]))
    assert float(ky) == 2.0


def test_partial_spectrum_never_negative_raises_with_d_total():
    # k=2 tracked exponents both non-negative, but the true system has
    # d_total=5 dimensions -- the crossing point is beyond what's tracked,
    # so silently returning k=2 would understate the true dimension.
    with pytest.raises(ValueError, match="lies beyond the tracked"):
        kaplan_yorke_dimension(jnp.array([0.5, 0.3]), d_total=5)


def test_partial_spectrum_never_negative_without_d_total_is_not_flagged():
    # Same input as above, but the caller didn't say this was a partial
    # spectrum -- the function trusts it's the full spectrum, per its
    # documented default.
    ky = kaplan_yorke_dimension(jnp.array([0.5, 0.3]))
    assert float(ky) == 2.0


def test_lorenz_published_exponents_match_known_ky_dimension():
    # Published Lorenz exponents (sigma=10, rho=28, beta=8/3):
    # ~0.9056, ~0.0, ~-14.5723 -- literature Kaplan-Yorke dimension ~2.06.
    ky = kaplan_yorke_dimension(jnp.array([0.9056, 0.0, -14.5723]))
    np.testing.assert_allclose(float(ky), 2.062, atol=5e-3)
