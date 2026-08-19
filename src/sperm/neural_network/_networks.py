"""NumPy neural-network graphs with hard parameterized shape constraints."""

from __future__ import annotations

import copy
from itertools import pairwise

import numpy as np


def _sigmoid(x):
    positive = x >= 0
    result = np.empty_like(x)
    result[positive] = 1 / (1 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


def _softplus(x):
    return np.logaddexp(0, x)


def _effective(raw, constraint):
    if constraint is None:
        return raw
    return np.where(constraint == 0, raw, constraint * _softplus(raw))


def _raw_gradient(gradient, raw, constraint):
    if constraint is None:
        return gradient
    factor = np.where(constraint == 0, 1.0, constraint * _sigmoid(raw))
    return gradient * factor


def _activate(x, name):
    if name == "identity":
        return x
    if name == "logistic":
        return _sigmoid(x)
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0)
    raise ValueError("activation must be 'identity', 'logistic', 'tanh', or 'relu'.")


def _activation_gradient(x, name):
    if name == "identity":
        return np.ones_like(x)
    if name == "logistic":
        value = _sigmoid(x)
        return value * (1 - value)
    if name == "tanh":
        value = np.tanh(x)
        return 1 - value**2
    return (x > 0).astype(float)


def _output_transform(raw, bounds):
    lower, upper = bounds
    if np.isfinite(lower) and np.isfinite(upper):
        probability = _sigmoid(raw)
        return lower + (upper - lower) * probability, (
            (upper - lower) * probability * (1 - probability)
        )
    if np.isfinite(lower):
        return lower + _softplus(raw), _sigmoid(raw)
    if np.isfinite(upper):
        return upper - _softplus(-raw), _sigmoid(-raw)
    return raw, np.ones_like(raw)


class DenseNetwork:
    """Ordinary or partially monotone dense MLP."""

    def __init__(self, layer_sizes, activation, monotonic_cst, rng):
        self.activation = activation
        self.params = []
        self.constraints = []
        constrained = np.any(monotonic_cst)
        for layer, (fan_in, fan_out) in enumerate(pairwise(layer_sizes)):
            scale = np.sqrt(2 / (fan_in + fan_out))
            self.params.extend(
                [rng.normal(0, scale, (fan_in, fan_out)), np.zeros(fan_out)]
            )
            if not constrained:
                weight_constraint = None
            elif layer == 0:
                weight_constraint = np.broadcast_to(
                    monotonic_cst[:, None],
                    (fan_in, fan_out),
                ).copy()
            else:
                weight_constraint = np.ones((fan_in, fan_out))
            self.constraints.extend([weight_constraint, None])

    def forward(self, X, bounds):
        activations = [X]
        preactivations = []
        effective_weights = []
        value = X
        n_layers = len(self.params) // 2
        for layer in range(n_layers):
            raw_weight, bias = self.params[2 * layer : 2 * layer + 2]
            weight = _effective(raw_weight, self.constraints[2 * layer])
            effective_weights.append(weight)
            preactivation = value @ weight + bias
            preactivations.append(preactivation)
            value = (
                preactivation
                if layer == n_layers - 1
                else _activate(preactivation, self.activation)
            )
            activations.append(value)
        raw = value[:, 0]
        prediction, output_gradient = _output_transform(raw, bounds)
        cache = activations, preactivations, effective_weights, output_gradient
        return prediction, cache

    def backward(self, prediction_gradient, cache, *, return_input_gradient=False):
        activations, preactivations, weights, output_gradient = cache
        delta = (prediction_gradient * output_gradient)[:, None]
        gradients = [None] * len(self.params)
        for layer in reversed(range(len(weights))):
            gradient_weight = activations[layer].T @ delta
            gradients[2 * layer] = _raw_gradient(
                gradient_weight,
                self.params[2 * layer],
                self.constraints[2 * layer],
            )
            gradients[2 * layer + 1] = delta.sum(axis=0)
            input_gradient = delta @ weights[layer].T
            if layer:
                delta = input_gradient * _activation_gradient(
                    preactivations[layer - 1],
                    self.activation,
                )
        if return_input_gradient:
            return gradients, input_gradient
        return gradients


class UnimodalNetwork:
    """One-dimensional single-valley or single-peak network."""

    def __init__(
        self, hidden_sizes, activation, kind, initial_turns, input_scale, rng
    ):
        self.kind = kind
        self.input_scale = float(input_scale)
        layer_sizes = (1, *hidden_sizes, 1)
        turns = np.asarray(initial_turns, dtype=float)
        template = (
            DenseNetwork(layer_sizes, activation, np.ones(1), rng),
            DenseNetwork(layer_sizes, activation, np.ones(1), rng),
        )
        self.branches = [
            (copy.deepcopy(template[0]), copy.deepcopy(template[1]))
            for _ in turns
        ]
        self.params = [
            parameter
            for branches in self.branches
            for branch in branches
            for parameter in branch.params
        ]
        self.turn_index = len(self.params)
        self.logit_index = self.turn_index + 1
        self.params.extend([turns, np.zeros_like(turns), np.zeros(1)])
        self.temperature = 0.0

    @property
    def turning_point(self):
        return float(self.turning_points[self.selected_candidate])

    @property
    def turning_points(self):
        return self.params[self.turn_index]

    @property
    def selected_candidate(self):
        return int(np.argmax(self.params[self.logit_index]))

    @property
    def candidate_weights(self):
        if self.temperature == 0:
            weights = np.zeros_like(self.params[self.logit_index])
            weights[self.selected_candidate] = 1.0
            return weights
        logits = self.params[self.logit_index] / self.temperature
        exponential = np.exp(logits - np.max(logits))
        return exponential / exponential.sum()

    def set_temperature(self, temperature):
        self.temperature = float(temperature)

    def _hinge(self, value):
        scaled_value = value / self.input_scale
        if self.temperature == 0:
            return (
                np.maximum(scaled_value, 0),
                (scaled_value > 0).astype(float) / self.input_scale,
                0.0,
            )
        scaled = scaled_value / self.temperature
        return (
            self.temperature * _softplus(scaled),
            _sigmoid(scaled) / self.input_scale,
            self.temperature * np.log(2.0),
        )

    def forward(self, X, bounds):
        left_caches = []
        right_caches = []
        left_hinge_gradients = []
        right_hinge_gradients = []
        candidate_components = []
        left_reference_caches = []
        right_reference_caches = []
        for turn, (left_branch, right_branch) in zip(
            self.turning_points, self.branches, strict=True
        ):
            left_input, left_hinge_gradient, _ = self._hinge(turn - X[:, 0])
            right_input, right_hinge_gradient, _ = self._hinge(X[:, 0] - turn)
            left_value, left_cache = left_branch.forward(
                left_input[:, None], (-np.inf, np.inf)
            )
            right_value, right_cache = right_branch.forward(
                right_input[:, None], (-np.inf, np.inf)
            )
            reference = self._hinge(np.asarray([0.0]))[2]
            left_reference, left_reference_cache = left_branch.forward(
                np.asarray([[reference]]), (-np.inf, np.inf)
            )
            right_reference, right_reference_cache = right_branch.forward(
                np.asarray([[reference]]), (-np.inf, np.inf)
            )
            left_caches.append(left_cache)
            right_caches.append(right_cache)
            left_reference_caches.append(left_reference_cache)
            right_reference_caches.append(right_reference_cache)
            left_hinge_gradients.append(left_hinge_gradient)
            right_hinge_gradients.append(right_hinge_gradient)
            candidate_components.append(
                left_value
                - left_reference[0]
                + right_value
                - right_reference[0]
            )
        orientation = 1.0 if self.kind == "unimodal_minimum" else -1.0
        candidate_components = np.column_stack(candidate_components)
        weights = self.candidate_weights
        mixed_component = candidate_components @ weights
        raw = self.params[-1][0] + orientation * mixed_component
        prediction, output_gradient = _output_transform(raw, bounds)
        cache = (
            left_caches,
            right_caches,
            left_reference_caches,
            right_reference_caches,
            left_hinge_gradients,
            right_hinge_gradients,
            candidate_components,
            mixed_component,
            weights,
            output_gradient,
            orientation,
        )
        return prediction, cache

    def candidate_predictions(self, cache, bounds):
        candidate_components = cache[6]
        orientation = cache[-1]
        candidate_raw = self.params[-1][0] + orientation * candidate_components
        return _output_transform(candidate_raw, bounds)

    def backward(self, prediction_gradient, cache, candidate_raw_gradient=None):
        (
            left_caches,
            right_caches,
            left_reference_caches,
            right_reference_caches,
            left_hinge_gradients,
            right_hinge_gradients,
            candidate_components,
            mixed_component,
            weights,
            output_gradient,
            orientation,
        ) = cache
        output_raw_gradient = prediction_gradient * output_gradient
        component_gradient = output_raw_gradient * orientation
        branch_gradients = []
        turn_gradients = np.zeros_like(self.turning_points)
        for candidate, weight in enumerate(weights):
            candidate_gradient = component_gradient * weight
            if candidate_raw_gradient is not None:
                candidate_gradient += (
                    candidate_raw_gradient[:, candidate] * orientation
                )
            left_branch, right_branch = self.branches[candidate]
            candidate_left, left_input_gradient = left_branch.backward(
                candidate_gradient,
                left_caches[candidate],
                return_input_gradient=True,
            )
            candidate_right, right_input_gradient = right_branch.backward(
                candidate_gradient,
                right_caches[candidate],
                return_input_gradient=True,
            )
            left_reference_gradients = left_branch.backward(
                np.asarray([-candidate_gradient.sum()]),
                left_reference_caches[candidate],
            )
            right_reference_gradients = right_branch.backward(
                np.asarray([-candidate_gradient.sum()]),
                right_reference_caches[candidate],
            )
            branch_gradients.extend(
                gradient + reference
                for gradient, reference in zip(
                    candidate_left, left_reference_gradients, strict=True
                )
            )
            branch_gradients.extend(
                gradient + reference
                for gradient, reference in zip(
                    candidate_right, right_reference_gradients, strict=True
                )
            )
            turn_gradients[candidate] = np.sum(
                left_input_gradient[:, 0] * left_hinge_gradients[candidate]
                - right_input_gradient[:, 0] * right_hinge_gradients[candidate]
            )
        if self.temperature == 0:
            logit_gradients = np.zeros_like(weights)
        else:
            logit_gradients = np.sum(
                output_raw_gradient[:, None]
                * orientation
                * weights[None, :]
                * (candidate_components - mixed_component[:, None]),
                axis=0,
            ) / self.temperature
        return [
            *branch_gradients,
            turn_gradients,
            logit_gradients,
            np.asarray(
                [
                    output_raw_gradient.sum()
                    + (
                        0.0
                        if candidate_raw_gradient is None
                        else candidate_raw_gradient.sum()
                    )
                ]
            ),
        ]


class ICNNNetwork:
    """Input-convex network with input skip connections at every layer."""

    def __init__(self, n_features, hidden_sizes, monotonic_cst, kind, rng):
        self.kind = kind
        self.hidden_sizes = hidden_sizes
        self.params = []
        self.constraints = []
        input_sign = monotonic_cst if kind == "convex" else -monotonic_cst

        for layer, width in enumerate(hidden_sizes):
            scale = np.sqrt(2 / (n_features + width))
            self.params.append(rng.normal(0, scale, (n_features, width)))
            self.constraints.append(
                np.broadcast_to(input_sign[:, None], (n_features, width)).copy()
            )
            if layer:
                previous = hidden_sizes[layer - 1]
                self.params.append(rng.normal(0, scale, (previous, width)))
                self.constraints.append(np.ones((previous, width)))
            self.params.append(np.zeros(width))
            self.constraints.append(None)

        last_width = hidden_sizes[-1]
        self.output_weight_index = len(self.params)
        self.params.extend(
            [
                rng.normal(0, np.sqrt(2 / last_width), (last_width, 1)),
                rng.normal(0, np.sqrt(2 / n_features), (n_features, 1)),
                np.zeros(1),
            ]
        )
        self.constraints.extend(
            [
                np.ones((last_width, 1)),
                input_sign[:, None].copy(),
                None,
            ]
        )

    def forward(self, X, bounds):
        hidden = []
        preactivations = []
        effective = []
        parameter_index = 0
        previous = None
        for layer in range(len(self.hidden_sizes)):
            input_weight = _effective(
                self.params[parameter_index],
                self.constraints[parameter_index],
            )
            effective.append(input_weight)
            parameter_index += 1
            preactivation = X @ input_weight
            if layer:
                recurrent = _effective(
                    self.params[parameter_index],
                    self.constraints[parameter_index],
                )
                effective.append(recurrent)
                parameter_index += 1
                preactivation += previous @ recurrent
            preactivation += self.params[parameter_index]
            parameter_index += 1
            previous = _softplus(preactivation)
            preactivations.append(preactivation)
            hidden.append(previous)

        output_weight = _effective(
            self.params[parameter_index], self.constraints[parameter_index]
        )
        input_output = _effective(
            self.params[parameter_index + 1],
            self.constraints[parameter_index + 1],
        )
        effective.extend([output_weight, input_output])
        convex_raw = (
            previous @ output_weight
            + X @ input_output
            + self.params[parameter_index + 2]
        )[:, 0]
        orientation = 1.0 if self.kind == "convex" else -1.0
        raw = orientation * convex_raw
        prediction, output_gradient = _output_transform(raw, bounds)
        cache = X, hidden, preactivations, effective, output_gradient, orientation
        return prediction, cache

    def backward(self, prediction_gradient, cache):
        X, hidden, preactivations, effective, output_gradient, orientation = cache
        delta_output = (prediction_gradient * output_gradient * orientation)[:, None]
        gradients = [np.zeros_like(parameter) for parameter in self.params]
        output_index = self.output_weight_index
        output_weight = effective[-2]
        gradients[output_index] = _raw_gradient(
            hidden[-1].T @ delta_output,
            self.params[output_index],
            self.constraints[output_index],
        )
        gradients[output_index + 1] = _raw_gradient(
            X.T @ delta_output,
            self.params[output_index + 1],
            self.constraints[output_index + 1],
        )
        gradients[output_index + 2] = delta_output.sum(axis=0)
        delta_hidden = delta_output @ output_weight.T

        indices = []
        parameter_index = 0
        effective_index = 0
        for layer in range(len(self.hidden_sizes)):
            input_index = parameter_index
            input_effective = effective[effective_index]
            parameter_index += 1
            effective_index += 1
            recurrent_index = None
            recurrent_effective = None
            if layer:
                recurrent_index = parameter_index
                recurrent_effective = effective[effective_index]
                parameter_index += 1
                effective_index += 1
            bias_index = parameter_index
            parameter_index += 1
            indices.append(
                (
                    input_index,
                    input_effective,
                    recurrent_index,
                    recurrent_effective,
                    bias_index,
                )
            )

        for layer in reversed(range(len(self.hidden_sizes))):
            dpre = delta_hidden * _sigmoid(preactivations[layer])
            input_index, _, recurrent_index, recurrent, bias_index = indices[layer]
            gradients[input_index] = _raw_gradient(
                X.T @ dpre,
                self.params[input_index],
                self.constraints[input_index],
            )
            gradients[bias_index] = dpre.sum(axis=0)
            if recurrent_index is not None:
                gradients[recurrent_index] = _raw_gradient(
                    hidden[layer - 1].T @ dpre,
                    self.params[recurrent_index],
                    self.constraints[recurrent_index],
                )
                delta_hidden = dpre @ recurrent.T
        return gradients


__all__ = ["DenseNetwork", "ICNNNetwork", "UnimodalNetwork"]
