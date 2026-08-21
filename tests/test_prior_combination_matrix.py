"""Acceptance matrix for every subset of the canonical shape priors."""

from itertools import combinations

import numpy as np
import pytest

from sperm.gaussian_process import GaussianProcessRegressor
from sperm.linear_model import LinearRegression, Ridge
from sperm.neural_network import MLPRegressor
from sperm.priors import Convex, Increasing, Priors, SlopeBound, Unimodality, ValueBound
from sperm.tree_model import (
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Maximum iterations reached before convergence.:"
    "sklearn.exceptions.ConvergenceWarning"
)

PRIOR_NAMES = ("value", "monotonicity", "slope", "unimodality", "convexity")


def _all_prior_combinations():
    return tuple(
        combination
        for size in range(len(PRIOR_NAMES) + 1)
        for combination in combinations(PRIOR_NAMES, size)
    )


def _make_priors(names):
    feature_priors = []
    if "monotonicity" in names:
        feature_priors.append(Increasing())
    if "slope" in names:
        feature_priors.append(SlopeBound(lower=-1, upper=1))
    if "unimodality" in names:
        feature_priors.append(Unimodality("minimum"))
    return Priors(
        value=ValueBound(lower=0) if "value" in names else None,
        features={0: tuple(feature_priors)} if feature_priors else {},
        curvature=Convex() if "convexity" in names else None,
    )


MODEL_CASES = (
    (
        "linear",
        lambda priors: LinearRegression(priors=priors),
        frozenset({"monotonicity", "slope"}),
    ),
    (
        "ridge",
        lambda priors: Ridge(priors=priors),
        frozenset({"monotonicity", "slope"}),
    ),
    (
        "decision-tree",
        lambda priors: DecisionTreeRegressor(
            max_depth=3, random_state=0, priors=priors
        ),
        frozenset({"value", "monotonicity", "unimodality"}),
    ),
    (
        "random-forest",
        lambda priors: RandomForestRegressor(
            n_estimators=3, max_depth=3, random_state=0, priors=priors
        ),
        frozenset({"value", "monotonicity", "unimodality"}),
    ),
    (
        "gradient-boosting",
        lambda priors: GradientBoostingRegressor(
            max_iter=3,
            max_leaf_nodes=4,
            early_stopping=False,
            random_state=0,
            priors=priors,
        ),
        frozenset({"value", "monotonicity", "unimodality"}),
    ),
    (
        "mlp",
        lambda priors: MLPRegressor(
            hidden_layer_sizes=(4,),
            max_iter=1,
            n_iter_no_change=2,
            random_state=0,
            priors=priors,
        ),
        frozenset({"value", "monotonicity", "unimodality", "convexity"}),
    ),
    (
        "gpr",
        lambda priors: GaussianProcessRegressor(
            n_basis=6, max_iter=100, priors=priors
        ),
        frozenset(PRIOR_NAMES),
    ),
)


@pytest.mark.parametrize(
    ("model_name", "make_estimator", "supported"),
    MODEL_CASES,
    ids=[case[0] for case in MODEL_CASES],
)
@pytest.mark.parametrize(
    "combination",
    _all_prior_combinations(),
    ids=lambda names: "none" if not names else "+".join(names),
)
def test_every_prior_subset_is_accepted_or_rejected_as_designed(
    model_name,
    make_estimator,
    supported,
    combination,
):
    del model_name
    X = np.linspace(-2, 2, 16).reshape(-1, 1)
    y = np.logaddexp(0, X[:, 0])
    estimator = make_estimator(_make_priors(combination))
    should_accept = set(combination) <= supported

    if should_accept:
        fitted = estimator.fit(X, y)
        assert fitted.priors_ is not None
    else:
        with pytest.raises((TypeError, ValueError)):
            estimator.fit(X, y)
