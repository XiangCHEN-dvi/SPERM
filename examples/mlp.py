"""Fit a lower-bounded, monotone, convex ICNN."""

import numpy as np

from sperm.neural_network import MLPRegressor
from sperm.priors import Convex, Increasing, Priors, ValueBound

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(500, 2))
y = np.exp(X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.05, X.shape[0])

priors = Priors(
    value=ValueBound(lower=0),
    features={0: Increasing()},
    curvature=Convex(),
)
model = MLPRegressor(
    hidden_layer_sizes=(32, 32),
    learning_rate_init=0.01,
    max_iter=500,
    early_stopping=True,
    random_state=0,
    priors=priors,
).fit(X, y)

print(model.score(X, y))
