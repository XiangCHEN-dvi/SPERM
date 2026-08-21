# Examples

Examples are executable scripts that compare constrained and unconstrained
models under controlled synthetic data.

```{toctree}
:maxdepth: 1

scalar-to-scalar
```

The repository's [`examples`](https://github.com/XiangCHEN-dvi/sperm/tree/main/examples)
directory contains the complete source.

## Prior Combinations

The model-specific scripts arrange plots by combination size: the first row
contains individual priors, the second contains pairs, and later rows contain
three- and four-prior combinations. Every panel compares the same estimator
with and without the displayed priors.

- [`linear_model.py`](https://github.com/XiangCHEN-dvi/sperm/blob/main/examples/linear_model.py)
  demonstrates monotonicity and slope bounds on `LinearRegression`.
- [`tree_models.py`](https://github.com/XiangCHEN-dvi/sperm/blob/main/examples/tree_models.py)
  demonstrates the non-redundant value, monotonicity, and unimodality subsets
  on gradient boosting.
- [`mlp.py`](https://github.com/XiangCHEN-dvi/sperm/blob/main/examples/mlp.py)
  demonstrates all subsets of a lower value bound, monotonicity, and
  convexity.
- [`gaussian_process.py`](https://github.com/XiangCHEN-dvi/sperm/blob/main/examples/gaussian_process.py)
  demonstrates all subsets of four compatible priors, including posterior
  uncertainty bands.

Run any script from a source checkout; for example:

```shell
uv run --extra examples python examples/gaussian_process.py
```

Generated figures are written to the local `assets` directory and are not
tracked by Git.
