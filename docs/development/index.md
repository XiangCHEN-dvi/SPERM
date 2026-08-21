# Development

SPERM uses a `src` layout, uv for reproducible environments, pytest for tests,
Ruff for static checks, and Sphinx for documentation.

```shell
git clone https://github.com/XiangCHEN-dvi/sperm.git
cd sperm
uv sync --extra dev
uv run pytest tests
uv run ruff check src tests examples
uv run sphinx-build -W -b html docs docs/_build/html
```

Read the repository's
[`CONTRIBUTING.md`](https://github.com/XiangCHEN-dvi/sperm/blob/main/CONTRIBUTING.md)
before opening a pull request.

```{toctree}
:maxdepth: 1

architecture
known-limitations
```
