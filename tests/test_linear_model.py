import numpy as np
import pytest
from sklearn.base import clone

from sperm.linear_model import LinearRegression, Ridge
from sperm.priors import Increasing, Lipschitz, Priors, ValueBound


@pytest.mark.parametrize("estimator", [LinearRegression(), Ridge(alpha=0.5)])
def test_estimators_follow_the_sklearn_clone_protocol(estimator):
    cloned = clone(estimator)

    assert cloned.get_params() == estimator.get_params()


def test_priors_container_follows_the_sklearn_clone_protocol():
    estimator = Ridge(
        priors=Priors(
            value=ValueBound(lower=0),
            features={0: Increasing()},
        )
    )

    assert clone(estimator).get_params() == estimator.get_params()


@pytest.mark.parametrize("estimator", [LinearRegression(), Ridge(alpha=0.0)])
def test_increasing_constraint_bounds_the_coefficient(estimator):
    X = np.arange(5.0).reshape(-1, 1)
    y = -2.0 * X[:, 0] + 3.0

    fitted = estimator.set_params(priors={0: Increasing()}).fit(X, y)

    assert fitted.coef_[0] == pytest.approx(0.0, abs=1e-8)


def test_lipschitz_constraint_bounds_the_coefficient():
    X = np.arange(5.0).reshape(-1, 1)
    y = 4.0 * X[:, 0] + 1.0

    fitted = LinearRegression(priors={0: Lipschitz(1.5)}).fit(X, y)

    assert fitted.coef_[0] == pytest.approx(1.5, abs=1e-8)


@pytest.mark.parametrize("estimator", [LinearRegression(), Ridge(alpha=0.0)])
def test_sample_weight_is_applied(estimator):
    X = np.array([[0.0], [1.0], [2.0], [100.0]])
    y = np.array([1.0, 3.0, 5.0, -1000.0])
    sample_weight = np.array([1.0, 1.0, 1.0, 0.0])

    fitted = estimator.fit(X, y, sample_weight=sample_weight)

    assert fitted.coef_[0] == pytest.approx(2.0, abs=1e-8)
    assert fitted.intercept_ == pytest.approx(1.0, abs=1e-8)


def test_multioutput_target_is_rejected():
    X = np.arange(6.0).reshape(3, 2)
    y = np.ones((3, 2))

    with pytest.raises(ValueError):
        LinearRegression().fit(X, y)


@pytest.mark.parametrize("estimator", [LinearRegression(), Ridge(alpha=1.0)])
def test_linear_models_reject_value_bound(estimator):
    X = np.arange(6.0).reshape(3, 2)
    y = np.array([4.0, 5.0, 6.0])
    priors = Priors(value=ValueBound(lower=1, upper=2))

    with pytest.raises(
        ValueError,
        match="ValueBound is not supported by the linear base model",
    ):
        estimator.set_params(priors=priors).fit(X, y)
