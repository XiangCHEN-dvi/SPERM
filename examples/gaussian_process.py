"""Analytic GPR with a globally constrained posterior mean."""

import numpy as np

from sperm.gaussian_process import GaussianProcessRegressor
from sperm.priors import Convex, Increasing, Priors, ValueBound

X = np.linspace(-2, 2, 50).reshape(-1, 1)
y = np.exp(0.5 * X[:, 0]) + 0.1 * np.sin(8 * X[:, 0])

model = GaussianProcessRegressor(
    n_basis=16,
    alpha=0.02,
    smoothness=0.1,
    priors=Priors(
        value=ValueBound(lower=0),
        features={0: Increasing()},
        curvature=Convex(),
    ),
).fit(X, y)

mean, standard_deviation = model.predict(X, return_std=True)
print(mean[:3])
print(standard_deviation[:3])
