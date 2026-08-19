import numpy as np
import pytest
from sklearn.base import clone

from sperm.priors import (
    Decreasing,
    Increasing,
    Priors,
    SlopeBound,
    Unimodality,
    ValueBound,
)
from sperm.tree_model import (
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)


def tree_estimators():
    return [
        DecisionTreeRegressor(max_depth=4, random_state=0),
        RandomForestRegressor(n_estimators=20, max_depth=4, random_state=0),
        GradientBoostingRegressor(
            max_iter=30,
            max_depth=4,
            min_samples_leaf=2,
            early_stopping=False,
            random_state=0,
        ),
    ]


@pytest.mark.parametrize("estimator", tree_estimators())
def test_tree_estimators_follow_sklearn_clone_protocol(estimator):
    configured = estimator.set_params(priors={0: Increasing()})

    assert clone(configured).get_params() == configured.get_params()


@pytest.mark.parametrize("estimator", tree_estimators())
def test_monotonicity_is_enforced_globally(estimator):
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 1))
    y = np.sin(4 * X[:, 0]) + 0.1 * rng.normal(size=200)
    grid = np.linspace(-3, 3, 501).reshape(-1, 1)

    predictions = estimator.set_params(priors={0: Increasing()}).fit(X, y).predict(grid)

    assert np.all(np.diff(predictions) >= -1e-12)


@pytest.mark.parametrize("estimator", tree_estimators())
def test_value_bound_is_a_hard_prediction_bound(estimator):
    X = np.arange(20.0).reshape(-1, 1)
    y = np.linspace(-10, 10, 20)
    priors = Priors(value=ValueBound(lower=-1, upper=2))

    predictions = estimator.set_params(priors=priors).fit(X, y).predict(
        [[-100.0], [0.0], [100.0]]
    )

    assert np.all(predictions >= -1)
    assert np.all(predictions <= 2)


@pytest.mark.parametrize("estimator", tree_estimators())
def test_tree_models_reject_slope_bound(estimator):
    X = np.arange(10.0).reshape(-1, 1)
    y = np.arange(10.0)

    with pytest.raises(TypeError, match="SlopeBound is not supported"):
        estimator.set_params(priors={0: SlopeBound(lower=0)}).fit(X, y)


@pytest.mark.parametrize("estimator", tree_estimators())
@pytest.mark.parametrize(
    ("prior", "is_minimum"),
    [(Unimodality("minimum"), True), (Unimodality("maximum"), False)],
)
def test_tree_models_enforce_one_dimensional_unimodality(
    estimator,
    prior,
    is_minimum,
):
    rng = np.random.default_rng(42)
    X = np.linspace(-3, 3, 120).reshape(-1, 1)
    base = X[:, 0] ** 2 if is_minimum else -(X[:, 0] ** 2)
    y = base + 2 * np.sin(5 * X[:, 0]) + rng.normal(0, 0.2, X.shape[0])
    grid = np.linspace(-5, 5, 501).reshape(-1, 1)

    predictions = estimator.set_params(priors={0: prior}).fit(X, y).predict(grid)
    differences = np.diff(predictions)
    if is_minimum:
        assert not np.any((differences[:-1] > 1e-12) & (differences[1:] < -1e-12))
        turn = np.flatnonzero(differences > 1e-12)
        if turn.size:
            assert np.all(differences[turn[0] :] >= -1e-12)
    else:
        turn = np.flatnonzero(differences < -1e-12)
        if turn.size:
            assert np.all(differences[turn[0] :] <= 1e-12)


def test_unimodality_requires_one_input_feature():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)

    with pytest.raises(ValueError, match="exactly one input feature"):
        DecisionTreeRegressor(priors={0: Unimodality("minimum")}).fit(X, y)


def test_random_forest_reuses_first_tree_turning_point():
    X = np.linspace(-2, 2, 80).reshape(-1, 1)
    y = (X[:, 0] - 0.4) ** 2 + np.sin(8 * X[:, 0])
    model = RandomForestRegressor(
        n_estimators=8,
        max_depth=4,
        random_state=0,
        priors={0: Unimodality("minimum")},
    ).fit(X, y)

    assert all(
        tree.turning_point_ == model.turning_point_ for tree in model.estimators_
    )


def test_unimodality_combines_with_value_bound():
    X = np.linspace(-2, 2, 80).reshape(-1, 1)
    y = 10 * X[:, 0] ** 2
    priors = Priors(
        value=ValueBound(lower=0, upper=2),
        features={0: Unimodality("minimum")},
    )

    predictions = DecisionTreeRegressor(
        max_depth=5,
        priors=priors,
    ).fit(X, y).predict(np.linspace(-4, 4, 301).reshape(-1, 1))

    assert np.all((0 <= predictions) & (predictions <= 2))


@pytest.mark.parametrize("estimator", tree_estimators())
@pytest.mark.parametrize(
    "unimodality", [Unimodality("minimum"), Unimodality("maximum")]
)
@pytest.mark.parametrize(
    ("monotonic_prior", "sign"),
    [(Increasing(), 1), (Decreasing(), -1)],
)
def test_monotonicity_absorbs_redundant_unimodality(
    estimator,
    unimodality,
    monotonic_prior,
    sign,
):
    X = np.linspace(-2, 2, 100).reshape(-1, 1)
    y = np.sin(5 * X[:, 0])
    model = estimator.set_params(
        priors={0: (unimodality, monotonic_prior)},
    ).fit(X, y)

    predictions = model.predict(np.linspace(-4, 4, 301).reshape(-1, 1))

    assert np.all(sign * np.diff(predictions) >= -1e-12)
    assert model.unimodality_constraint_ is None
    assert model.priors_.features == {0: (monotonic_prior,)}


@pytest.mark.parametrize("estimator", tree_estimators())
@pytest.mark.parametrize("sign", [1, -1])
def test_both_unimodality_modes_select_a_hard_monotonic_direction(estimator, sign):
    X = np.linspace(-2, 2, 120).reshape(-1, 1)
    y = sign * X[:, 0] + 0.05 * np.sin(7 * X[:, 0])
    model = estimator.set_params(
        priors={
            0: (Unimodality("minimum"), Unimodality("maximum"))
        },
    ).fit(X, y)

    predictions = model.predict(np.linspace(-3, 3, 301).reshape(-1, 1))

    assert np.all(sign * np.diff(predictions) >= -1e-12)
    assert model.monotonic_direction_ == (
        "increasing" if sign == 1 else "decreasing"
    )
    assert model.priors_.features == {
        0: (Unimodality("minimum"), Unimodality("maximum"))
    }
