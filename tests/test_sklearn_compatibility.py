from sklearn.utils.estimator_checks import parametrize_with_checks

from sperm.gaussian_process import GaussianProcessRegressor
from sperm.linear_model import LinearRegression, Ridge
from sperm.neural_network import MLPRegressor
from sperm.tree_model import (
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)


@parametrize_with_checks(
    [
        LinearRegression(),
        Ridge(),
        DecisionTreeRegressor(),
        # Bootstrap sampling is intentionally incompatible with sklearn's
        # repeated-sample equivalence check.
        RandomForestRegressor(bootstrap=False),
        GradientBoostingRegressor(),
        MLPRegressor(
            hidden_layer_sizes=(5,),
            learning_rate_init=0.01,
            max_iter=100,
            n_iter_no_change=10,
            random_state=0,
        ),
        GaussianProcessRegressor(n_basis=8),
    ]
)
def test_sklearn_estimator_compatibility(estimator, check):
    check(estimator)
