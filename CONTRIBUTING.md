# Contributing

Thank you for helping improve SPERM. Bug fixes, tests, documentation, and
focused feature proposals are welcome.

## Development setup

The project uses Python 3.12 for local development and supports Python 3.10+
as declared in `pyproject.toml`. Install the development environment with
[uv](https://docs.astral.sh/uv/):

```shell
uv sync --extra dev
```

Run commands through `uv run` so they use the project environment.

## Making changes

Create a short-lived branch and keep each change focused. Source code belongs
in `src/sperm`, tests in `tests`, examples in `examples`, and documentation in
`README.md` or `docs`.

- Use public scikit-learn APIs and follow its estimator conventions.
- Preserve hard, full-domain shape guarantees unless a limitation is clearly
  documented.
- Add regression tests for bug fixes and tests for new behavior.
- Update documentation and examples when changing the public API.
- Do not include generated figures, caches, virtual environments, or build
  artifacts.

## Checks

Before submitting a change, run:

```shell
uv run --extra dev pytest tests
uv run --extra dev ruff check src tests examples
```

When changing documentation, also build it locally:

```shell
uv run --extra dev sphinx-build -W -b html docs docs/_build/html
```

## Pull requests

A pull request should explain what changed and why, note any API or numerical
trade-offs, and list the checks that were run. Keep unrelated refactoring out
of the same pull request and make sure all tests pass before requesting review.

By contributing, you agree that your contribution is licensed under the
project's Apache License 2.0.
