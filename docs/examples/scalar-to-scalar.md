# Scalar-to-Scalar Comparison

[`scalar2scalar.py`](https://github.com/XiangCHEN-dvi/sperm/blob/main/examples/scalar2scalar.py)
builds a matrix of one-input, one-output experiments. Rows are shape priors;
columns group linear, tree, MLP, and Gaussian-process regressors. Solid curves
are constrained fits, dashed curves are unconstrained fits, and GPR panels add
analytic uncertainty bands.

Run it from a source checkout:

```shell
uv run --extra examples python examples/scalar2scalar.py
```

The experiment covers value bounds, monotonicity, slope bounds, unimodality,
and convexity. A crossed panel identifies a combination that is trivial,
impossible, degrading, or not yet supported instead of drawing a misleading
fit.

## A Focused Experiment

The same comparison pattern can be used in a smaller script:

```python
import matplotlib.pyplot as plt
import numpy as np

from sperm.gaussian_process import GaussianProcessRegressor
from sperm.priors import Increasing

rng = np.random.default_rng(0)
X = np.linspace(-2, 2, 50).reshape(-1, 1)
y = X[:, 0] ** 3 + rng.normal(scale=1.0, size=X.shape[0])
grid = np.linspace(-2.2, 2.2, 300).reshape(-1, 1)

plain = GaussianProcessRegressor().fit(X, y)
shaped = GaussianProcessRegressor(priors={0: Increasing()}).fit(X, y)

plt.scatter(X[:, 0], y, s=18, color="0.35")
plt.plot(grid[:, 0], plain.predict(grid), "--", label="w/o")
plt.plot(grid[:, 0], shaped.predict(grid), "-", label="w/")
plt.legend()
plt.show()
```
