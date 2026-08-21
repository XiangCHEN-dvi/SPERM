"""Greedy regression trees with bounded leaves and monotone value ranges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class _Node:
    indices: np.ndarray
    value_lower: float
    value_upper: float
    value: float
    depth: int
    feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None
    gain: float = 0.0


class _BoundedTree:
    """Fit squared-error leaves inside inherited hard value intervals."""

    def __init__(
        self,
        *,
        max_depth,
        max_leaf_nodes,
        min_samples_split,
        min_samples_leaf,
        min_impurity_decrease,
        max_features,
        monotonic_cst,
        value_bounds,
        random_state,
    ):
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.monotonic_cst = monotonic_cst
        self.value_bounds = value_bounds
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        weights = (
            np.ones(y.shape[0], dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        lower, upper = self.value_bounds
        root_value = _bounded_mean(y, weights, lower, upper)
        self.root_ = _Node(
            np.arange(y.shape[0]), lower, upper, root_value, depth=0
        )
        leaves = [self.root_]
        rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.feature_gains_ = np.zeros(self.n_features_in_, dtype=float)

        while self.max_leaf_nodes is None or len(leaves) < self.max_leaf_nodes:
            best = None
            for leaf in leaves:
                candidate = self._best_split(leaf, X, y, weights, rng)
                if candidate is not None and (
                    best is None or candidate[0] > best[0]
                ):
                    best = candidate
            if best is None or best[0] <= self.min_impurity_decrease:
                break
            gain, leaf, feature, threshold, left, right = best
            leaf.feature = feature
            leaf.threshold = threshold
            leaf.left = left
            leaf.right = right
            leaf.gain = gain
            self.feature_gains_[feature] += gain
            leaves.remove(leaf)
            leaves.extend((left, right))

        self.leaves_ = tuple(leaves)
        total_gain = self.feature_gains_.sum()
        self.feature_importances_ = (
            self.feature_gains_ / total_gain
            if total_gain > 0
            else np.zeros_like(self.feature_gains_)
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.asarray([self._predict_row(row) for row in X], dtype=float)

    def get_depth(self):
        return max(leaf.depth for leaf in self.leaves_)

    def get_n_leaves(self):
        return len(self.leaves_)

    def _predict_row(self, row):
        node = self.root_
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def _best_split(self, leaf, X, y, weights, rng):
        if self.max_depth is not None and leaf.depth >= self.max_depth:
            return None
        required = max(self.min_samples_split, 2 * self.min_samples_leaf)
        if leaf.indices.size < required:
            return None

        parent_loss = _squared_loss(
            y[leaf.indices], weights[leaf.indices], leaf.value
        )
        best = None
        for feature in _sample_features(
            self.n_features_in_, self.max_features, rng
        ):
            order = leaf.indices[
                np.argsort(X[leaf.indices, feature], kind="mergesort")
            ]
            ordered_x = X[order, feature]
            for position in range(
                self.min_samples_leaf,
                order.size - self.min_samples_leaf + 1,
            ):
                if ordered_x[position - 1] == ordered_x[position]:
                    continue
                left_indices = order[:position]
                right_indices = order[position:]
                left_weight = weights[left_indices].sum()
                right_weight = weights[right_indices].sum()
                if left_weight <= 0 or right_weight <= 0:
                    continue
                left_mean = np.average(y[left_indices], weights=weights[left_indices])
                right_mean = np.average(
                    y[right_indices], weights=weights[right_indices]
                )
                direction = (
                    0
                    if self.monotonic_cst is None
                    else self.monotonic_cst[feature]
                )
                left_value, right_value = _bounded_pair(
                    left_mean,
                    right_mean,
                    left_weight,
                    right_weight,
                    leaf.value_lower,
                    leaf.value_upper,
                    direction,
                )
                left_lower = right_lower = leaf.value_lower
                left_upper = right_upper = leaf.value_upper
                if direction:
                    midpoint = (left_value + right_value) / 2
                    if direction > 0:
                        left_upper = min(left_upper, midpoint)
                        right_lower = max(right_lower, midpoint)
                    else:
                        left_lower = max(left_lower, midpoint)
                        right_upper = min(right_upper, midpoint)
                loss = _squared_loss(
                    y[left_indices], weights[left_indices], left_value
                )
                loss += _squared_loss(
                    y[right_indices], weights[right_indices], right_value
                )
                gain = parent_loss - loss
                if best is None or gain > best[0]:
                    threshold = (ordered_x[position - 1] + ordered_x[position]) / 2
                    left = _Node(
                        left_indices,
                        left_lower,
                        left_upper,
                        left_value,
                        leaf.depth + 1,
                    )
                    right = _Node(
                        right_indices,
                        right_lower,
                        right_upper,
                        right_value,
                        leaf.depth + 1,
                    )
                    best = (gain, leaf, feature, threshold, left, right)
        return best


def _bounded_pair(
    mean_left,
    mean_right,
    weight_left,
    weight_right,
    lower,
    upper,
    direction,
):
    left = float(np.clip(mean_left, lower, upper))
    right = float(np.clip(mean_right, lower, upper))
    ordered = direction == 0 or (direction > 0 and left <= right) or (
        direction < 0 and left >= right
    )
    if ordered:
        return left, right
    pooled = (weight_left * mean_left + weight_right * mean_right) / (
        weight_left + weight_right
    )
    pooled = float(np.clip(pooled, lower, upper))
    return pooled, pooled


def _bounded_mean(y, weights, lower, upper):
    return float(np.clip(np.average(y, weights=weights), lower, upper))


def _squared_loss(y, weights, value):
    return float(np.dot(weights, (y - value) ** 2))


def _sample_features(n_features, max_features, rng):
    if max_features is None or max_features == 1.0:
        count = n_features
    elif isinstance(max_features, int):
        count = max_features
    elif isinstance(max_features, float):
        count = max(1, int(np.ceil(max_features * n_features)))
    elif max_features == "sqrt":
        count = max(1, int(np.sqrt(n_features)))
    elif max_features == "log2":
        count = max(1, int(np.log2(n_features)))
    else:
        raise ValueError("Unsupported max_features for a bounded tree.")
    if not 1 <= count <= n_features:
        raise ValueError("max_features must select between 1 and n_features.")
    return rng.choice(n_features, size=count, replace=False)


__all__ = ["_BoundedTree"]
