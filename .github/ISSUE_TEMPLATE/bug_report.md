---
name: Bug report
about: Report incorrect results, a crash, or unexpected behavior
title: ''
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of what went wrong.

**Minimal reproducing example**

```python
# a minimal script that reproduces the issue
```

**Expected behavior**
What you expected to happen instead.

**Environment**

- lyapax version: `python -c "import lyapax; print(lyapax.__version__)"`
- JAX version and backend (CPU/GPU): `python -c "import jax; print(jax.__version__, jax.default_backend())"`
- `jax_enable_x64` set? (float32 is a common source of Lyapunov-exponent
  results that look like bugs but aren't)
- OS / Python version

**Additional context**
Any other context, tracebacks, or plots.
