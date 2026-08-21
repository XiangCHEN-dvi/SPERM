from sperm.gaussian_process import GaussianProcessRegressor
from sperm.linear_model import LinearRegression, Ridge
from sperm.neural_network import MLPRegressor
from sperm.priors import (
    Concave,
    Convex,
    Decreasing,
    Increasing,
    Lipschitz,
    Monotonicity,
    Prior,
    Priors,
    SlopeBound,
    Unimodality,
    ValueBound,
    parse_priors,
)
from sperm.tree_model import (
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)


def test_public_api_exports_estimators():
    assert Concave is not None
    assert Convex is not None
    assert Increasing is not None
    assert Decreasing is not None
    assert Lipschitz is not None
    assert Monotonicity is not None
    assert ValueBound is not None
    assert SlopeBound is not None
    assert Prior is not None
    assert Priors is not None
    assert Unimodality is not None
    assert parse_priors is not None
    assert LinearRegression is not None
    assert Ridge is not None
    assert MLPRegressor is not None
    assert GaussianProcessRegressor is not None
    assert DecisionTreeRegressor is not None
    assert RandomForestRegressor is not None
    assert GradientBoostingRegressor is not None
