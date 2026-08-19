"""Scikit-learn-compatible constrained multilayer perceptron regression."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted, validate_data

from ..priors._mlp_compiler import compile_mlp_priors
from ._networks import DenseNetwork, ICNNNetwork, UnimodalNetwork, _effective


class MLPRegressor(RegressorMixin, BaseEstimator):
    """MLP regressor with hard global shape priors."""

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="tanh",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
        unimodality_n_candidates=5,
        unimodality_temperature=1.0,
        unimodality_soft_fraction=0.75,
        priors=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.unimodality_n_candidates = unimodality_n_candidates
        self.unimodality_temperature = unimodality_temperature
        self.unimodality_soft_fraction = unimodality_soft_fraction
        self.priors = priors

    def fit(self, X, y, sample_weight=None):
        """Fit the constrained neural network."""
        self._validate_hyperparameters()
        X_array, y_array = validate_data(
            self, X, y, reset=True, y_numeric=True
        )
        if y_array.ndim != 1:
            raise ValueError("MLPRegressor currently supports single-output regression.")
        weights = _validate_sample_weight(sample_weight, X_array.shape[0])
        self.priors_, architectures = compile_mlp_priors(
            self.priors,
            n_features=X_array.shape[1],
            feature_names=getattr(self, "feature_names_in_", None),
        )
        hidden_sizes = _normalize_hidden_sizes(self.hidden_layer_sizes)
        rng = np.random.default_rng(self.random_state)
        X_train, y_train, weight_train, validation = self._validation_split(
            X_array, y_array, weights, rng
        )
        fitted_candidates = []
        for architecture in architectures:
            signature = self._network_signature(
                architecture, X_array, hidden_sizes
            )
            reuse_network = (
                len(architectures) == 1
                and self.warm_start
                and hasattr(self, "_network_")
                and getattr(self, "_network_signature_", None) == signature
            )
            if not reuse_network:
                self._network_ = self._make_network(
                    architecture, X_array, hidden_sizes, rng
                )
            self.architecture_ = architecture
            self._network_signature_ = signature
            self.loss_curve_ = []
            self.validation_scores_ = []
            self.best_validation_score_ = None
            self._train(X_train, y_train, weight_train, validation, rng)
            score = (
                self._validation_loss(validation)
                if validation is not None
                else self._loss_and_gradients(X_train, y_train, weight_train)[0]
            )
            fitted_candidates.append(
                (
                    score,
                    self._network_,
                    architecture,
                    signature,
                    self.loss_curve_,
                    self.validation_scores_,
                    self.best_validation_score_,
                    self.n_iter_,
                    self.loss_,
                )
            )
        (
            _,
            self._network_,
            self.architecture_,
            self._network_signature_,
            self.loss_curve_,
            self.validation_scores_,
            self.best_validation_score_,
            self.n_iter_,
            self.loss_,
        ) = min(fitted_candidates, key=lambda candidate: candidate[0])
        self.n_architecture_candidates_ = len(architectures)
        self.turning_point_ = (
            self._network_.turning_point
            if isinstance(self._network_, UnimodalNetwork)
            else None
        )
        self.turning_points_ = (
            self._network_.turning_points.copy()
            if isinstance(self._network_, UnimodalNetwork)
            else None
        )
        self.unimodality_candidate_weights_ = (
            self._network_.candidate_weights.copy()
            if isinstance(self._network_, UnimodalNetwork)
            else None
        )
        self.n_outputs_ = 1
        self.out_activation_ = "identity"
        self._set_public_weight_attributes()
        return self

    def _network_signature(self, architecture, X, hidden_sizes):
        activation = (
            self.activation
            if architecture.kind in {"dense", "unimodal_minimum", "unimodal_maximum"}
            else "softplus"
        )
        return (
            architecture.kind,
            X.shape[1],
            hidden_sizes,
            tuple(architecture.monotonic_cst),
            activation,
            _unimodality_input_scale(X)
            if architecture.kind in {"unimodal_minimum", "unimodal_maximum"}
            else None,
            self.unimodality_n_candidates
            if architecture.kind in {"unimodal_minimum", "unimodal_maximum"}
            else None,
        )

    def _make_network(self, architecture, X, hidden_sizes, rng):
        if architecture.kind == "dense":
            return DenseNetwork(
                (X.shape[1], *hidden_sizes, 1),
                self.activation,
                architecture.monotonic_cst,
                rng,
            )
        if architecture.kind in {"unimodal_minimum", "unimodal_maximum"}:
            return UnimodalNetwork(
                hidden_sizes,
                self.activation,
                architecture.kind,
                _unimodality_initial_turns(X, self.unimodality_n_candidates),
                _unimodality_input_scale(X),
                rng,
            )
        return ICNNNetwork(
            X.shape[1],
            hidden_sizes,
            architecture.monotonic_cst,
            architecture.kind,
            rng,
        )

    def predict(self, X):
        """Predict using the fitted constrained network."""
        check_is_fitted(self, "_network_")
        X_array = validate_data(self, X, reset=False)
        return self._network_.forward(
            X_array, self.architecture_.value_bounds
        )[0]

    def _train(self, X, y, weights, validation, rng):
        params = self._network_.params
        velocities = [np.zeros_like(parameter) for parameter in params]
        first = [np.zeros_like(parameter) for parameter in params]
        second = [np.zeros_like(parameter) for parameter in params]
        batch_size = _resolve_batch_size(self.batch_size, X.shape[0])
        best_loss, best_params, no_improvement, step = np.inf, None, 0, 0
        unimodal = isinstance(self._network_, UnimodalNetwork)
        soft_iterations = (
            min(
                self.max_iter - 1,
                round(self.unimodality_soft_fraction * self.max_iter),
            )
            if unimodal
            else 0
        )
        for iteration in range(self.max_iter):
            if unimodal:
                if iteration < soft_iterations:
                    progress = iteration / max(soft_iterations - 1, 1)
                    temperature = self.unimodality_temperature * 1e-3**progress
                else:
                    temperature = 0.0
                self._network_.set_temperature(temperature)
                self.unimodality_temperature_ = temperature
                if iteration == soft_iterations:
                    self._select_hard_unimodality_candidate(X, y, weights)
                    best_loss, best_params, no_improvement = np.inf, None, 0
                    for index in (
                        self._network_.turn_index,
                        self._network_.logit_index,
                    ):
                        velocities[index].fill(0)
                        first[index].fill(0)
                        second[index].fill(0)
            indices = rng.permutation(X.shape[0]) if self.shuffle else np.arange(X.shape[0])
            for start in range(0, X.shape[0], batch_size):
                batch = indices[start : start + batch_size]
                step += 1
                _, gradients = self._loss_and_gradients(
                    X[batch], y[batch], None if weights is None else weights[batch]
                )
                rate = self._learning_rate(step)
                if self.solver == "adam":
                    _adam_update(
                        params, gradients, first, second, step, rate,
                        self.beta_1, self.beta_2, self.epsilon,
                    )
                else:
                    _sgd_update(
                        params, gradients, velocities, rate,
                        self.momentum, self.nesterovs_momentum,
                    )
            train_loss = self._loss_and_gradients(X, y, weights)[0]
            self.loss_curve_.append(train_loss)
            monitored = self._validation_loss(validation) if validation else train_loss
            if validation:
                self.validation_scores_.append(-monitored)
                self.best_validation_score_ = max(self.validation_scores_)
            if monitored < best_loss - self.tol:
                best_loss = monitored
                best_params = [parameter.copy() for parameter in params]
                no_improvement = 0
            else:
                no_improvement += 1
            self.n_iter_, self.loss_ = iteration + 1, train_loss
            if self.verbose:
                print(f"Iteration {self.n_iter_}, loss = {train_loss:.8f}")
            if no_improvement >= self.n_iter_no_change and iteration >= soft_iterations:
                break
        else:
            warnings.warn(
                "Maximum iterations reached before convergence.",
                ConvergenceWarning,
                stacklevel=2,
            )
        if validation and best_params is not None:
            for parameter, best_parameter in zip(
                self._network_.params, best_params, strict=True
            ):
                parameter[...] = best_parameter
        if unimodal:
            self._network_.set_temperature(0.0)
            self.unimodality_temperature_ = 0.0

    def _select_hard_unimodality_candidate(self, X, y, weights):
        logits = self._network_.params[self._network_.logit_index]
        losses = []
        for candidate in range(len(logits)):
            logits.fill(0.0)
            logits[candidate] = 1.0
            losses.append(self._loss_and_gradients(X, y, weights)[0])
        selected = int(np.argmin(losses))
        logits.fill(0.0)
        logits[selected] = 1.0

    def _loss_and_gradients(self, X, y, weights):
        predictions, cache = self._network_.forward(
            X, self.architecture_.value_bounds
        )
        effective_weights = np.ones_like(y) if weights is None else weights
        weight_sum = effective_weights.sum()
        residual = predictions - y
        loss = float(np.dot(effective_weights, residual**2) / weight_sum)
        prediction_gradient = 2 * effective_weights * residual / weight_sum
        candidate_raw_gradient = None
        if (
            isinstance(self._network_, UnimodalNetwork)
            and self._network_.temperature > 0
        ):
            candidate_predictions, candidate_output_gradient = (
                self._network_.candidate_predictions(
                    cache, self.architecture_.value_bounds
                )
            )
            candidate_residual = candidate_predictions - y[:, None]
            n_candidates = candidate_predictions.shape[1]
            if n_candidates > 1:
                loss += float(
                    np.sum(effective_weights[:, None] * candidate_residual**2)
                    / (weight_sum * n_candidates)
                )
                candidate_raw_gradient = (
                    2
                    * effective_weights[:, None]
                    * candidate_residual
                    * candidate_output_gradient
                    / (weight_sum * n_candidates)
                )
        if isinstance(self._network_, UnimodalNetwork):
            gradients = self._network_.backward(
                prediction_gradient,
                cache,
                candidate_raw_gradient=candidate_raw_gradient,
            )
        else:
            gradients = self._network_.backward(prediction_gradient, cache)
        if self.alpha:
            for index, parameter in enumerate(self._network_.params):
                if (
                    isinstance(self._network_, UnimodalNetwork)
                    and index
                    in {self._network_.turn_index, self._network_.logit_index}
                ):
                    continue
                loss += self.alpha * float(np.sum(parameter**2)) / (2 * weight_sum)
                gradients[index] += self.alpha * parameter / weight_sum
        return loss, gradients

    def _validation_split(self, X, y, weights, rng):
        if not self.early_stopping:
            return X, y, weights, None
        n_validation = max(1, round(self.validation_fraction * X.shape[0]))
        if n_validation >= X.shape[0]:
            raise ValueError("validation_fraction leaves no training samples.")
        indices = rng.permutation(X.shape[0])
        validation_indices, training_indices = indices[:n_validation], indices[n_validation:]
        validation = (
            X[validation_indices], y[validation_indices],
            None if weights is None else weights[validation_indices],
        )
        return (
            X[training_indices], y[training_indices],
            None if weights is None else weights[training_indices], validation,
        )

    def _validation_loss(self, validation):
        return self._loss_and_gradients(*validation)[0]

    def _learning_rate(self, step):
        if self.learning_rate == "invscaling":
            return self.learning_rate_init / step**self.power_t
        return self.learning_rate_init

    def _validate_hyperparameters(self):
        if self.solver not in {"adam", "sgd"}:
            raise ValueError("solver must be 'adam' or 'sgd'.")
        if self.learning_rate not in {"constant", "invscaling"}:
            raise ValueError("learning_rate must be 'constant' or 'invscaling'.")
        if self.activation not in {"identity", "logistic", "tanh", "relu"}:
            raise ValueError("Unsupported activation.")
        if self.max_iter <= 0 or self.learning_rate_init <= 0:
            raise ValueError("max_iter and learning_rate_init must be positive.")
        if self.alpha < 0 or self.n_iter_no_change <= 0:
            raise ValueError("alpha must be non-negative and n_iter_no_change positive.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1).")
        if self.unimodality_temperature <= 0:
            raise ValueError("unimodality_temperature must be positive.")
        if (
            not isinstance(self.unimodality_n_candidates, int)
            or isinstance(self.unimodality_n_candidates, bool)
            or self.unimodality_n_candidates <= 0
        ):
            raise ValueError("unimodality_n_candidates must be a positive integer.")
        if not 0 <= self.unimodality_soft_fraction < 1:
            raise ValueError("unimodality_soft_fraction must be in [0, 1).")

    def _set_public_weight_attributes(self):
        if isinstance(self._network_, DenseNetwork):
            self.coefs_ = [
                _effective(self._network_.params[i], self._network_.constraints[i])
                for i in range(0, len(self._network_.params), 2)
            ]
            self.intercepts_ = self._network_.params[1::2]
        else:
            self.coefs_, self.intercepts_ = [], []


def _normalize_hidden_sizes(value):
    sizes = (value,) if isinstance(value, int) else tuple(value)
    if not sizes or not all(isinstance(size, int) and size > 0 for size in sizes):
        raise ValueError("hidden_layer_sizes must contain positive integers.")
    return sizes


def _unimodality_input_scale(X):
    scale = float(np.std(X[:, 0]))
    return scale if scale > np.finfo(float).eps else 1.0


def _unimodality_initial_turns(X, n_candidates):
    if n_candidates == 1:
        return np.asarray([np.median(X[:, 0])], dtype=float)
    quantiles = np.linspace(0.1, 0.9, n_candidates)
    return np.asarray(np.quantile(X[:, 0], quantiles), dtype=float)


def _resolve_batch_size(value, n_samples):
    if value == "auto":
        return min(200, n_samples)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("batch_size must be 'auto' or a positive integer.")
    return min(value, n_samples)


def _validate_sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (n_samples,) or np.any(weights < 0):
        raise ValueError("sample_weight must be non-negative with shape (n_samples,).")
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError(
            "sample_weight must contain at least one non-zero finite weight."
        )
    return weights


def _adam_update(params, grads, first, second, step, rate, beta_1, beta_2, epsilon):
    for i, (parameter, gradient) in enumerate(zip(params, grads, strict=True)):
        first[i] = beta_1 * first[i] + (1 - beta_1) * gradient
        second[i] = beta_2 * second[i] + (1 - beta_2) * gradient**2
        first_hat = first[i] / (1 - beta_1**step)
        second_hat = second[i] / (1 - beta_2**step)
        parameter -= rate * first_hat / (np.sqrt(second_hat) + epsilon)


def _sgd_update(params, grads, velocities, rate, momentum, nesterov):
    for i, (parameter, gradient) in enumerate(zip(params, grads, strict=True)):
        velocities[i] = momentum * velocities[i] - rate * gradient
        update = momentum * velocities[i] - rate * gradient if nesterov else velocities[i]
        parameter += update


__all__ = ["MLPRegressor"]
