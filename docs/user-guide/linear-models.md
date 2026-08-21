# Linear Models

{class}`sperm.linear_model.LinearRegression` and
{class}`sperm.linear_model.Ridge` constrain coefficients directly. For an
affine function, a feature's coefficient is its global slope, so arbitrary
slope intervals and monotonicity are exact linear constraints.

```python
from sperm.linear_model import Ridge
from sperm.priors import SlopeBound

model = Ridge(
    alpha=1.0,
    priors={0: SlopeBound(lower=-0.25, upper=1.5)},
).fit(X, y)
```

The constrained least-squares problem is solved as a hard constraint, not a
regularization penalty. Value bounds, unimodality, and convexity are rejected:
over the full real input space they would make an affine base model trivial or
add no meaningful nonlinear capacity.
