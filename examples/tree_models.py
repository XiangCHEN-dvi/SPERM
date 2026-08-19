"""Fit tree-based regressors with global value and monotonicity priors."""

import numpy as np

from sperm.priors import Increasing, Priors, ValueBound
from sperm.tree_model import (
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

X = np.linspace(-2, 2, 200).reshape(-1, 1)
y = X[:, 0] ** 3 + np.sin(5 * X[:, 0])
priors = Priors(
    value=ValueBound(lower=-5, upper=5),
    features={0: Increasing()},
)

models = [
    DecisionTreeRegressor(max_depth=4, priors=priors),
    RandomForestRegressor(n_estimators=100, random_state=0, priors=priors),
    GradientBoostingRegressor(max_iter=100, random_state=0, priors=priors),
]

for model in models:
    predictions = model.fit(X, y).predict(X)
    assert np.all(np.diff(predictions) >= -1e-12)
    assert np.all((-5 <= predictions) & (predictions <= 5))
    print(type(model).__name__, model.score(X, y))
