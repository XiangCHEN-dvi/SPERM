import numpy as np
import pytest
from sklearn.base import clone

from sperm.neural_network import MLPRegressor
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


def _model(priors=None):
    return MLPRegressor(
        hidden_layer_sizes=(8, 6),
        max_iter=8,
        n_iter_no_change=20,
        random_state=0,
        priors=priors,
    )


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] + rng.normal(0, 0.1, 80)
    return X, y


def test_mlp_follows_clone_protocol():
    model = _model({0: Increasing()})

    assert clone(model).get_params() == model.get_params()


def test_value_bound_is_global():
    X, y = _data()
    model = _model(Priors(value=ValueBound(lower=-1, upper=2))).fit(X, y)
    predictions = model.predict(np.array([[-1e6, 1e6], [1e6, -1e6]]))

    assert np.all((-1 <= predictions) & (predictions <= 2))


@pytest.mark.parametrize(
    ("prior", "sign"),
    [(Increasing(), 1), (Decreasing(), -1)],
)
def test_partial_monotonicity_is_global(prior, sign):
    X, y = _data()
    model = _model({0: prior}).fit(X, y)
    grid = np.column_stack([np.linspace(-10, 10, 501), np.full(501, 0.3)])

    assert np.all(sign * np.diff(model.predict(grid)) >= -1e-10)


@pytest.mark.parametrize(
    ("curvature", "sign"),
    [(Convex(), 1), (Concave(), -1)],
)
def test_icnn_curvature_is_global(curvature, sign):
    X, y = _data()
    model = _model(Priors(curvature=curvature)).fit(X, y)
    start = np.array([-3.0, 1.0])
    end = np.array([4.0, -2.0])
    fractions = np.linspace(0, 1, 301)[:, None]
    line = start + fractions * (end - start)
    predictions = model.predict(line)

    assert np.all(sign * np.diff(predictions, n=2) >= -1e-9)


@pytest.mark.parametrize(
    ("curvature", "bound", "monotonic", "sign"),
    [
        (Convex(), ValueBound(lower=0), Increasing(), 1),
        (Concave(), ValueBound(upper=3), Decreasing(), -1),
    ],
)
def test_curvature_value_and_monotonicity_combine(
    curvature,
    bound,
    monotonic,
    sign,
):
    X, y = _data()
    priors = Priors(value=bound, features={0: monotonic}, curvature=curvature)
    model = _model(priors).fit(X, y)
    grid = np.column_stack([np.linspace(-20, 20, 401), np.zeros(401)])
    predictions = model.predict(grid)

    assert np.all(sign * np.diff(predictions) >= -1e-10)
    if bound.lower is not None:
        assert np.all(predictions >= bound.lower)
    if bound.upper is not None:
        assert np.all(predictions <= bound.upper)
    assert np.all(sign * np.diff(predictions, n=2) >= -1e-8)


@pytest.mark.parametrize(
    "priors",
    [
        Priors(value=ValueBound(upper=1), curvature=Convex()),
        Priors(value=ValueBound(lower=-1), curvature=Concave()),
    ],
)
def test_rejects_degrading_curvature_and_value_combinations(priors):
    X, y = _data()

    with pytest.raises(ValueError, match="constant"):
        _model(priors).fit(X, y)


def test_warm_start_reuses_an_unchanged_network():
    X, y = _data()
    model = _model({0: Increasing()}).set_params(warm_start=True).fit(X, y)
    network = model._network_

    model.fit(X, y)

    assert model._network_ is network


def test_warm_start_rebuilds_when_curvature_architecture_changes():
    X, y = _data()
    model = _model().set_params(warm_start=True).fit(X, y)
    network = model._network_

    model.set_params(priors=Priors(curvature=Convex())).fit(X, y)

    assert model._network_ is not network
    line = np.column_stack((np.linspace(-5, 5, 501), np.zeros(501)))
    assert np.diff(model.predict(line), n=2).min() >= -1e-8


def test_warm_start_rebuilds_when_monotonic_direction_changes():
    X, y = _data()
    model = _model({0: Increasing()}).set_params(warm_start=True).fit(X, y)
    network = model._network_

    model.set_params(priors={0: Decreasing()}).fit(X, y)

    assert model._network_ is not network
    line = np.column_stack((np.linspace(-5, 5, 501), np.zeros(501)))
    assert np.diff(model.predict(line)).max() <= 1e-10


@pytest.mark.parametrize(
    ("prior", "sign"),
    [(Unimodality("minimum"), 1), (Unimodality("maximum"), -1)],
)
def test_unimodality_shape_is_global_and_hard_after_training(prior, sign):
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    y = sign * ((X[:, 0] - 0.4) ** 2 + 0.15 * np.sin(5 * X[:, 0]))
    model = MLPRegressor(
        hidden_layer_sizes=(8,),
        max_iter=40,
        n_iter_no_change=50,
        random_state=0,
        priors={0: prior},
    ).fit(X, y)
    grid = np.linspace(-100, 100, 20001).reshape(-1, 1)
    shaped = sign * model.predict(grid)
    turn = np.argmin(shaped)

    assert np.diff(shaped[: turn + 1]).max(initial=0.0) <= 1e-10
    assert np.diff(shaped[turn:]).min(initial=0.0) >= -1e-10
    assert model.unimodality_temperature_ == 0
    assert model._network_.temperature == 0
    assert model._network_.input_scale == pytest.approx(np.std(X[:, 0]))


def test_unimodality_turning_point_is_not_restricted_to_the_training_interval():
    X = np.linspace(-1, 1, 20).reshape(-1, 1)
    y = (X[:, 0] - 3) ** 2
    model = MLPRegressor(
        hidden_layer_sizes=(4,),
        max_iter=2,
        n_iter_no_change=3,
        random_state=0,
        priors={0: Unimodality("minimum")},
    ).fit(X, y)
    selected = model._network_.selected_candidate
    model._network_.params[model._network_.turn_index][selected] = 3.0

    assert model.predict(np.asarray([[2.9]]))[0] <= model.predict(np.asarray([[2.0]]))[0]
    assert model.predict(np.asarray([[3.1]]))[0] <= model.predict(np.asarray([[4.0]]))[0]


def test_soft_unimodality_turning_point_gradient_matches_finite_difference():
    X = np.linspace(-1, 1, 12).reshape(-1, 1)
    y = (X[:, 0] - 0.2) ** 2
    model = MLPRegressor(
        hidden_layer_sizes=(3,),
        max_iter=1,
        random_state=0,
        priors={0: Unimodality("minimum")},
    ).fit(X, y)
    model._network_.set_temperature(0.3)
    _, gradients = model._loss_and_gradients(X, y, None)
    parameter = model._network_.params[model._network_.turn_index]
    epsilon = 1e-6
    parameter[0] += epsilon
    plus = model._loss_and_gradients(X, y, None)[0]
    parameter[0] -= 2 * epsilon
    minus = model._loss_and_gradients(X, y, None)[0]
    parameter[0] += epsilon

    assert gradients[model._network_.turn_index][0] == pytest.approx(
        (plus - minus) / (2 * epsilon), rel=1e-5, abs=1e-7
    )

    logit = model._network_.params[model._network_.logit_index]
    logit[0] += epsilon
    plus = model._loss_and_gradients(X, y, None)[0]
    logit[0] -= 2 * epsilon
    minus = model._loss_and_gradients(X, y, None)[0]
    logit[0] += epsilon
    assert gradients[model._network_.logit_index][0] == pytest.approx(
        (plus - minus) / (2 * epsilon), rel=1e-5, abs=1e-7
    )


def test_unimodality_softmax_hardens_to_exactly_one_candidate():
    X = np.linspace(-2, 2, 40).reshape(-1, 1)
    y = (X[:, 0] - 0.3) ** 2
    model = MLPRegressor(
        hidden_layer_sizes=(5,),
        max_iter=10,
        n_iter_no_change=20,
        unimodality_n_candidates=5,
        random_state=0,
        priors={0: Unimodality("minimum")},
    ).fit(X, y)

    assert model.turning_points_.shape == (5,)
    assert np.count_nonzero(model.unimodality_candidate_weights_) == 1
    assert model.unimodality_candidate_weights_.sum() == 1
    assert model.turning_point_ in model.turning_points_


def test_unimodality_convex_combines_with_value_bound():
    X = np.linspace(-2, 2, 40).reshape(-1, 1)
    y = (X[:, 0] - 0.2) ** 2
    model = MLPRegressor(
        hidden_layer_sizes=(6,),
        max_iter=20,
        n_iter_no_change=30,
        random_state=0,
        priors=Priors(
            value=ValueBound(lower=-0.5, upper=1.5),
            features={0: Unimodality("minimum")},
        ),
    ).fit(X, y)
    prediction = model.predict(np.linspace(-100, 100, 2001).reshape(-1, 1))
    turn = np.argmin(prediction)

    assert prediction.min() >= -0.5
    assert prediction.max() <= 1.5
    assert np.diff(prediction[: turn + 1]).max(initial=0.0) <= 1e-10
    assert np.diff(prediction[turn:]).min(initial=0.0) >= -1e-10


def test_monotonicity_makes_unimodality_priors_redundant():
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = X[:, 0] ** 3
    model = _model(
        {0: (Increasing(), Unimodality("minimum"), Unimodality("maximum"))}
    ).fit(X, y)

    assert model.architecture_.kind == "dense"
    assert model.n_architecture_candidates_ == 1
    assert model.turning_point_ is None


@pytest.mark.parametrize(
    ("unimodality", "curvature", "kind"),
    [(Unimodality("minimum"), Convex(), "convex"), (Unimodality("maximum"), Concave(), "concave")],
)
def test_matching_curvature_makes_unimodality_redundant(unimodality, curvature, kind):
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = X[:, 0] ** 2
    model = _model(
        Priors(features={0: unimodality}, curvature=curvature)
    ).fit(X, y)

    assert model.architecture_.kind == kind
    assert model.n_architecture_candidates_ == 1


def test_both_unimodality_directions_select_a_monotonic_candidate():
    X = np.linspace(-2, 2, 40).reshape(-1, 1)
    y = X[:, 0]
    model = _model({0: (Unimodality("minimum"), Unimodality("maximum"))}).fit(X, y)
    difference = np.diff(model.predict(np.linspace(-20, 20, 1001).reshape(-1, 1)))

    assert model.n_architecture_candidates_ == 2
    assert difference.min() >= -1e-10 or difference.max() <= 1e-10


@pytest.mark.parametrize(
    ("unimodality", "curvature", "curvature_sign"),
    [(Unimodality("minimum"), Concave(), -1), (Unimodality("maximum"), Convex(), 1)],
)
def test_opposite_curvature_and_unimodality_select_monotonic_icnn(
    unimodality, curvature, curvature_sign
):
    X = np.linspace(-2, 2, 40).reshape(-1, 1)
    y = np.sin(X[:, 0])
    model = _model(
        Priors(features={0: unimodality}, curvature=curvature)
    ).fit(X, y)
    prediction = model.predict(np.linspace(-20, 20, 1001).reshape(-1, 1))
    difference = np.diff(prediction)

    assert model.n_architecture_candidates_ == 2
    assert difference.min() >= -1e-10 or difference.max() <= 1e-10
    assert curvature_sign * np.diff(prediction, n=2).min() >= -1e-8


def test_unimodality_rejects_multidimensional_input_and_slope_bound():
    X, y = _data()
    with pytest.raises(ValueError, match="exactly one input feature"):
        _model({0: Unimodality("minimum")}).fit(X, y)
    with pytest.raises(TypeError, match="SlopeBound"):
        _model({0: SlopeBound(lower=-1, upper=1)}).fit(X, y)


def test_early_stopping_validation_split_selects_architecture(monkeypatch):
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = X[:, 0]

    def fake_train(self, X, y, weights, validation, rng):
        self.n_iter_ = 1
        self.loss_ = 0.0

    def validation_loss(self, validation):
        return 0.0 if self.architecture_.monotonic_cst[0] == 1 else 10.0

    def training_loss(self, X, y, weights):
        loss = 10.0 if self.architecture_.monotonic_cst[0] == 1 else 0.0
        return loss, [np.zeros_like(parameter) for parameter in self._network_.params]

    monkeypatch.setattr(MLPRegressor, "_train", fake_train)
    monkeypatch.setattr(MLPRegressor, "_validation_loss", validation_loss)
    monkeypatch.setattr(MLPRegressor, "_loss_and_gradients", training_loss)

    model = _model({0: (Unimodality("minimum"), Unimodality("maximum"))}).set_params(
        early_stopping=True
    ).fit(X, y)

    assert model.architecture_.monotonic_cst[0] == 1


def test_unimodality_warm_start_reuses_only_the_same_architecture():
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = (X[:, 0] - 0.3) ** 2
    model = _model({0: Unimodality("minimum")}).set_params(warm_start=True).fit(X, y)
    unimodal_network = model._network_

    model.fit(X, y)
    assert model._network_ is unimodal_network
    assert model._network_.temperature == 0

    model.set_params(priors=None).fit(X, y)
    assert model._network_ is not unimodal_network
