"""Pytest configuration for lyapax.

Forces CPU execution: this dev machine's GPU has a broken cudnn/driver
combination (RET_CHECK dnn_support != nullptr), so tests must not silently
dispatch to the GPU backend. Also enables float64, which the Lyapunov
engine requires (float32 silently corrupts long-horizon log-growth-rate
averages). Both must happen before jax picks a backend or traces anything,
hence set at conftest import time, not inside a fixture.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
