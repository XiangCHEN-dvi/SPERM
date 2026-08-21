# Gaussian Processes

{class}`sperm.gaussian_process.GaussianProcessRegressor` is a finite-rank,
additive Gaussian process built on clamped B-spline bases and a Gaussian
P-spline smoothness prior.

```python
from sperm.gaussian_process import GaussianProcessRegressor
from sperm.priors import Convex, Increasing, Priors, ValueBound

model = GaussianProcessRegressor(
    n_basis=20,
    priors=Priors(
        value=ValueBound(lower=0),
        features={0: Increasing()},
        curvature=Convex(),
    ),
).fit(X, y)

mean, std = model.predict(X_test, return_std=True)
```

Gaussian observation noise gives a closed-form coefficient posterior. SPERM
projects only its mean into the requested constraint set in the posterior
precision metric and retains the analytic covariance. Constraints on spline
controls and divided differences scale linearly with the number of basis
functions.

The representation is additive across features, so its cross-feature
expressiveness is intentionally restricted. Linear extrapolation carries
derivative and curvature constraints to the complete real line. Value bounds
also constrain tail slopes and currently require one input feature.
