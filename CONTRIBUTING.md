# Contributing to lyapax

Thanks for your interest in improving lyapax. Bug reports, documentation
fixes, new examples, and pull requests are all welcome.

## Reporting bugs and requesting features

Please open a [GitHub issue](https://github.com/Ziaeemehr/lyapax/issues).
For bug reports, include:

- The lyapax version (`python -c "import lyapax; print(lyapax.__version__)"`)
- A minimal reproducing script
- The full traceback or a description of the incorrect result
- Whether `jax_enable_x64` was set (float32 is a common source of
  Lyapunov-exponent bugs that are not actually bugs)

## Getting help

For usage questions that are not bugs, open a
[GitHub Discussion](https://github.com/Ziaeemehr/lyapax/discussions) or an
issue tagged `question`.

## Development setup

```bash
git clone https://github.com/Ziaeemehr/lyapax.git
cd lyapax
pip install -e ".[dev,examples,docs]"
```

Run the test suite:

```bash
pytest -q
```

Lint:

```bash
ruff check .
```

Build the docs:

```bash
sphinx-build -b html docs docs/_build/html
```

## Making a pull request

1. Fork the repository and create a branch off `main`.
2. Add or update tests for any behavior change.
3. Make sure `pytest -q` and `ruff check .` pass locally.
4. If you change public API behavior, update the relevant docstring and,
   if applicable, `docs/background/capabilities.md`.
5. Open a pull request describing the change and why it's needed.

## Code style

- Keep step functions and tangent propagation JAX-native: no Python-side
  loops in traced code, prefer `jax.lax.scan`/`jax.vmap` over manual
  batching.
- New public functions should have a docstring describing shapes and
  units of the returned Lyapunov exponents.
- Follow the existing `ruff` configuration (`pyproject.toml`).

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
