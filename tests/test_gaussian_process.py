import numpy as np
import pytest
from sklearn.base import clone

from sperm.gaussian_process import GaussianProcessRegressor
from sperm.priors import (
    Concave,
    Convex,
    Decreasing,
    Increasing,
    Priors,
    SlopeBound,
    Unimodality,
    ValueBound,
)


def _data():
    X = np.linspace(-2, 2, 40).reshape(-1, 1)
    return X, np.sin(X[:, 0]) + 0.2 * X[:, 0]


def test_clone_and_analytic_posterior():
    estimator = GaussianProcessRegressor(n_basis=10, alpha=0.1)
    assert clone(estimator).get_params() == estimator.get_params()
    X, y = _data()
    fitted = estimator.fit(X, y)
    mean, covariance = fitted.predict(X[:5], return_cov=True)
    mean_std, std = fitted.predict(X[:5], return_std=True)
    assert mean.shape == (5,)
    assert covariance.shape == (5, 5)
    assert np.allclose(mean, mean_std)
    assert np.allclose(std**2, np.diag(covariance))
    assert fitted.posterior_covariance_.shape == (
        fitted.n_basis_features_,
        fitted.n_basis_features_,
    )


def test_basis_is_independent_of_priors():
    X, y = _data()
    unconstrained = GaussianProcessRegressor(n_basis=9).fit(X, y)
    constrained = GaussianProcessRegressor(
        n_basis=9, priors=Priors(value=ValueBound(lower=-1, upper=1))
    ).fit(X, y)
    grid = np.linspace(-4, 4, 101).reshape(-1, 1)
    assert np.array_equal(
        unconstrained.basis_.features[0].knots,
        constrained.basis_.features[0].knots,
    )
    assert np.allclose(
        unconstrained.basis_.transform(grid),
        constrained.basis_.transform(grid),
    )


def test_spline_gp_has_useful_unconstrained_capacity():
    X = np.linspace(-3, 3, 80).reshape(-1, 1)
    y = np.sin(2 * X[:, 0]) + 0.2 * np.cos(5 * X[:, 0])
    prediction = GaussianProcessRegressor(
        n_basis=20,
        alpha=1e-4,
        smoothness=0.01,
    ).fit(X, y).predict(X)
    assert np.sqrt(np.mean((prediction - y) ** 2)) < 0.05


def test_constraint_count_grows_linearly_with_basis_size():
    X, y = _data()
    counts = []
    for n_basis in (8, 16):
        model = GaussianProcessRegressor(
            n_basis=n_basis,
            priors=Priors(
                value=ValueBound(lower=-1, upper=1),
                features={0: Increasing()},
            ),
        ).fit(X, y)
        counts.append(model.constraint_matrix_.shape[0])
    assert counts == [27, 51]


def test_global_value_bound_with_monotonicity():
    X, y = _data()
    model = GaussianProcessRegressor(
        n_basis=12,
        priors=Priors(
            value=ValueBound(lower=-0.5, upper=1.5),
            features={0: Increasing()},
        ),
    ).fit(X, y)
    grid = np.linspace(-1e3, 1e3, 1001).reshape(-1, 1)
    prediction = model.predict(grid)
    assert prediction.min() >= -0.5 - 1e-6
    assert prediction.max() <= 1.5 + 1e-6
    assert np.diff(prediction).min() >= -1e-6


@pytest.mark.parametrize(
    ("curvature", "value", "feature", "sign"),
    [
        (Convex(), ValueBound(lower=-1), Increasing(), 1),
        (Concave(), ValueBound(upper=1), Decreasing(), -1),
    ],
)
def test_curvature_value_and_monotonicity_combine(
    curvature, value, feature, sign
):
    X, y = _data()
    model = GaussianProcessRegressor(
        n_basis=10,
        priors=Priors(value=value, features={0: feature}, curvature=curvature),
    ).fit(X, y)
    grid = np.linspace(-20, 20, 2001).reshape(-1, 1)
    prediction = model.predict(grid)
    first = sign * np.diff(prediction)
    second = sign * np.diff(prediction, n=2)
    assert first.min() >= -1e-6
    assert second.min() >= -1e-6
    if value.lower is not None:
        assert prediction.min() >= value.lower - 1e-6
    if value.upper is not None:
        assert prediction.max() <= value.upper + 1e-6


def test_slope_bound_is_global():
    X, y = _data()
    model = GaussianProcessRegressor(
        n_basis=12, priors={0: SlopeBound(lower=-0.3, upper=0.4)}
    ).fit(X, y)
    grid = np.linspace(-100, 100, 20001)
    step = grid[1] - grid[0]
    slope = np.diff(model.predict(grid.reshape(-1, 1))) / step
    assert slope.min() >= -0.3 - 2e-4
    assert slope.max() <= 0.4 + 2e-4


@pytest.mark.parametrize(
    ("prior", "sign"),
    [(Unimodality("minimum"), 1), (Unimodality("maximum"), -1)],
)
def test_unimodality_shape_is_global(prior, sign):
    X = np.linspace(-3, 3, 60).reshape(-1, 1)
    y = sign * (np.exp(-X[:, 0] ** 2) + 0.15 * np.sin(4 * X[:, 0]))
    model = GaussianProcessRegressor(
        n_basis=12,
        alpha=0.02,
        smoothness=0.1,
        priors={0: prior},
    ).fit(X, y)
    grid = np.linspace(-100, 100, 20001).reshape(-1, 1)
    shaped_prediction = sign * model.predict(grid)
    sampled_turn = np.argmin(shaped_prediction)
    assert np.diff(shaped_prediction[: sampled_turn + 1]).max(initial=0.0) <= 1e-8
    assert np.diff(shaped_prediction[sampled_turn:]).min(initial=0.0) >= -1e-8
    assert model.unimodality_candidate_[0] == 0
    turn = model.unimodality_candidate_[2]
    derivative = model.basis_.derivative_control_map(0, 1) @ model.posterior_mean_
    assert np.max(sign * derivative[:turn], initial=0.0) <= 1e-6
    assert np.min(sign * derivative[turn:], initial=0.0) >= -1e-6


def test_unimodality_convex_combines_with_value_and_slope_bounds():
    X = np.linspace(-3, 3, 60).reshape(-1, 1)
    y = (X[:, 0] - 0.4) ** 2 - 1
    model = GaussianProcessRegressor(
        n_basis=12,
        priors=Priors(
            value=ValueBound(lower=-0.8, upper=1.2),
            features={
                0: (Unimodality("minimum"), SlopeBound(lower=-0.7, upper=0.6))
            },
        ),
    ).fit(X, y)
    grid = np.linspace(-100, 100, 20001)
    prediction = model.predict(grid.reshape(-1, 1))
    slope = np.diff(prediction) / np.diff(grid)
    turn = np.argmin(prediction)
    assert prediction.min() >= -0.8 - 1e-6
    assert prediction.max() <= 1.2 + 1e-6
    assert slope.min() >= -0.7 - 2e-4
    assert slope.max() <= 0.6 + 2e-4
    assert np.diff(prediction[: turn + 1]).max(initial=0.0) <= 1e-6
    assert np.diff(prediction[turn:]).min(initial=0.0) >= -1e-6


def test_redundant_unimodality_priors_do_not_enumerate_candidates():
    X, y = _data()
    monotonic = GaussianProcessRegressor(
        priors={0: (Increasing(), Unimodality("minimum"), Unimodality("maximum"))}
    ).fit(X, y)
    convex = GaussianProcessRegressor(
        priors=Priors(features={0: Unimodality("minimum")}, curvature=Convex())
    ).fit(X, y)
    assert monotonic.unimodality_candidate_ is None
    assert convex.unimodality_candidate_ is None


def test_unimodality_convex_and_concave_reduce_to_a_monotonic_solution():
    X, y = _data()
    model = GaussianProcessRegressor(
        n_basis=10,
        priors={0: (Unimodality("minimum"), Unimodality("maximum"))},
    ).fit(X, y)
    prediction = model.predict(np.linspace(-20, 20, 2001).reshape(-1, 1))
    difference = np.diff(prediction)
    assert model.unimodality_candidate_ in {
        (0, "increasing"),
        (0, "decreasing"),
    }
    assert difference.min() >= -1e-6 or difference.max() <= 1e-6


def test_unimodality_candidate_count_grows_linearly():
    X, y = _data()
    model = GaussianProcessRegressor(
        n_basis=13, priors={0: Unimodality("minimum")}
    ).fit(X, y)
    assert model.n_unimodality_candidates_ == 13


def test_unimodality_applies_to_one_feature_of_a_multifeature_model():
    first = np.linspace(-2, 2, 12)
    second = np.linspace(-3, 3, 24)
    x0, x1 = np.meshgrid(first, second, indexing="ij")
    X = np.column_stack((x0.ravel(), x1.ravel()))
    y = np.sin(X[:, 0]) + (X[:, 1] - 0.4) ** 2
    model = GaussianProcessRegressor(
        n_basis=10,
        alpha=0.01,
        priors={1: Unimodality("minimum")},
    ).fit(X, y)

    grid = np.linspace(-20, 20, 2001)
    left_slice = np.column_stack((np.full_like(grid, -1.0), grid))
    right_slice = np.column_stack((np.full_like(grid, 1.0), grid))
    left_prediction = model.predict(left_slice)
    right_prediction = model.predict(right_slice)
    turn = model.unimodality_candidate_[2]
    derivative = model.basis_.derivative_control_map(1, 1) @ model.posterior_mean_

    assert model.unimodality_candidate_[:2] == (1, "minimum")
    assert np.max(derivative[:turn], initial=0.0) <= 1e-6
    assert np.min(derivative[turn:], initial=0.0) >= -1e-6
    assert np.ptp(right_prediction - left_prediction) <= 1e-8
    assert abs(np.mean(right_prediction - left_prediction)) > 0.1


def test_unimodality_currently_accepts_at_most_one_feature():
    X = np.column_stack(
        (np.linspace(-2, 2, 30), np.linspace(-3, 3, 30) ** 3)
    )
    y = X[:, 0] ** 2 + X[:, 1] ** 2

    with pytest.raises(ValueError, match="at most one feature"):
        GaussianProcessRegressor(
            priors={
                0: Unimodality("minimum"),
                1: Unimodality("minimum"),
            }
        ).fit(X, y)


@pytest.mark.parametrize(
    ("unimodality", "curvature"),
    [(Unimodality("minimum"), Concave()), (Unimodality("maximum"), Convex())],
)
def test_opposite_unimodality_and_curvature_combine_as_monotonic(unimodality, curvature):
    X, y = _data()
    model = GaussianProcessRegressor(
        n_basis=10,
        priors=Priors(features={0: unimodality}, curvature=curvature),
    ).fit(X, y)
    prediction = model.predict(np.linspace(-20, 20, 2001).reshape(-1, 1))
    difference = np.diff(prediction)
    assert difference.min() >= -1e-6 or difference.max() <= 1e-6


def test_rejects_unsupported_or_degrading_combinations():
    X, y = _data()
    with pytest.raises(ValueError, match="finite upper bound"):
        GaussianProcessRegressor(
            priors=Priors(value=ValueBound(upper=1), curvature=Convex())
        ).fit(X, y)
    with pytest.raises(ValueError, match="exactly one feature"):
        GaussianProcessRegressor(priors=Priors(value=ValueBound(upper=1))).fit(
            np.column_stack((X, X**2)), y
        )
