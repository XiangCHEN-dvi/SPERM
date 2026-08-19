"""One-dimensional greedy trees with hard unimodality constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Leaf:
    indices: np.ndarray
    lower: float
    upper: float
    value: float
    depth: int


class _UnimodalTree:
    """Grow a piecewise-constant tree while preserving a unimodal leaf sequence."""

    def __init__(
        self,
        *,
        kind: str,
        max_depth: int | None,
        max_leaf_nodes: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        value_bounds: tuple[float, float],
        turning_point: float | None = None,
    ) -> None:
        self.kind = kind
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.value_bounds = value_bounds
        self.turning_point = turning_point

    def fit(self, X, y, sample_weight=None):
        x = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float)
        weights = (
            np.ones_like(y, dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        root_value = np.clip(
            np.average(y, weights=weights),
            *self.value_bounds,
        )
        self.leaves_ = [
            _Leaf(
                indices=np.arange(x.size),
                lower=-np.inf,
                upper=np.inf,
                value=float(root_value),
                depth=0,
            )
        ]

        while self.max_leaf_nodes is None or len(self.leaves_) < self.max_leaf_nodes:
            best = None
            for leaf_index, leaf in enumerate(self.leaves_):
                candidate = self._best_leaf_split(
                    leaf_index,
                    leaf,
                    x,
                    y,
                    weights,
                )
                if candidate is not None and (
                    best is None or candidate[0] > best[0]
                ):
                    best = candidate
            if best is None or best[0] <= self.min_impurity_decrease:
                break
            _, leaf_index, left, right = best
            self.leaves_[leaf_index : leaf_index + 1] = [left, right]

        self.thresholds_ = np.asarray(
            [leaf.upper for leaf in self.leaves_[:-1]],
            dtype=float,
        )
        self.values_ = np.asarray([leaf.value for leaf in self.leaves_], dtype=float)
        self.feature_importances_ = np.ones(1, dtype=float)
        self.turning_point_ = self._infer_turning_point(x)
        return self

    def predict(self, X):
        x = np.asarray(X, dtype=float).reshape(-1)
        positions = np.searchsorted(self.thresholds_, x, side="left")
        return self.values_[positions]

    def get_depth(self):
        return max(leaf.depth for leaf in self.leaves_)

    def get_n_leaves(self):
        return len(self.leaves_)

    def _best_leaf_split(self, leaf_index, leaf, x, y, weights):
        if self.max_depth is not None and leaf.depth >= self.max_depth:
            return None
        if leaf.indices.size < max(self.min_samples_split, 2 * self.min_samples_leaf):
            return None

        order = leaf.indices[np.argsort(x[leaf.indices], kind="mergesort")]
        x_ordered = x[order]
        parent_loss = _squared_loss(y[order], weights[order], leaf.value)
        existing_values = [item.value for item in self.leaves_]
        best = None

        for position in range(self.min_samples_leaf, order.size - self.min_samples_leaf + 1):
            if x_ordered[position - 1] == x_ordered[position]:
                continue
            left_indices = order[:position]
            right_indices = order[position:]
            threshold = (x_ordered[position - 1] + x_ordered[position]) / 2
            left_weight = weights[left_indices].sum()
            right_weight = weights[right_indices].sum()
            if left_weight <= 0 or right_weight <= 0:
                continue
            left_mean = np.average(y[left_indices], weights=weights[left_indices])
            right_mean = np.average(y[right_indices], weights=weights[right_indices])
            values = self._optimal_children(
                existing_values,
                leaf_index,
                left_mean,
                right_mean,
                left_weight,
                right_weight,
                threshold,
            )
            if values is None:
                continue
            left_value, right_value = values
            loss = _squared_loss(y[left_indices], weights[left_indices], left_value)
            loss += _squared_loss(
                y[right_indices],
                weights[right_indices],
                right_value,
            )
            gain = parent_loss - loss
            if best is None or gain > best[0]:
                left = _Leaf(
                    left_indices,
                    leaf.lower,
                    threshold,
                    left_value,
                    leaf.depth + 1,
                )
                right = _Leaf(
                    right_indices,
                    threshold,
                    leaf.upper,
                    right_value,
                    leaf.depth + 1,
                )
                best = (gain, leaf_index, left, right)
        return best

    def _optimal_children(
        self,
        old_values,
        leaf_index,
        left_mean,
        right_mean,
        left_weight,
        right_weight,
        threshold,
    ):
        n_values = len(old_values) + 1
        valleys = self._candidate_turns(n_values, threshold)
        best = None
        for turn in valleys:
            constraints = _constraints_for_turn(
                old_values,
                leaf_index,
                turn,
                self.kind,
                self.value_bounds,
            )
            if constraints is None:
                continue
            candidate = _solve_two_values(
                left_mean,
                right_mean,
                left_weight,
                right_weight,
                constraints,
            )
            if candidate is None:
                continue
            loss = left_weight * (candidate[0] - left_mean) ** 2
            loss += right_weight * (candidate[1] - right_mean) ** 2
            if best is None or loss < best[0]:
                best = (loss, *candidate)
        return None if best is None else (best[1], best[2])

    def _candidate_turns(self, n_values, threshold):
        if self.turning_point is None:
            return range(n_values)
        return (int(np.searchsorted(self._new_boundaries(threshold), self.turning_point)),)

    def _new_boundaries(self, threshold):
        boundaries = list(self.thresholds_) if hasattr(self, "thresholds_") else [
            leaf.upper for leaf in self.leaves_[:-1]
        ]
        boundaries.append(threshold)
        return np.sort(np.asarray(boundaries, dtype=float))

    def _infer_turning_point(self, x):
        if self.turning_point is not None:
            return self.turning_point
        target = (
            np.min(self.values_)
            if self.kind == "minimum"
            else np.max(self.values_)
        )
        positions = np.flatnonzero(np.isclose(self.values_, target))
        leaf = self.leaves_[int(positions[len(positions) // 2])]
        data = x[leaf.indices]
        return float((data.min() + data.max()) / 2)


def _constraints_for_turn(old_values, leaf_index, turn, kind, value_bounds):
    fixed = list(old_values)
    fixed[leaf_index : leaf_index + 1] = [None, None]
    lower_a = lower_b = value_bounds[0]
    upper_a = upper_b = value_bounds[1]
    relation = None

    for edge in range(len(fixed) - 1):
        descending = edge < turn
        if kind == "maximum":
            descending = not descending
        left, right = fixed[edge], fixed[edge + 1]
        if left is not None and right is not None:
            if (descending and left < right) or (not descending and left > right):
                return None
            continue
        if left is None and right is None:
            relation = "ge" if descending else "le"
        elif left is None:
            if edge == leaf_index:
                if descending:
                    lower_a = max(lower_a, right)
                else:
                    upper_a = min(upper_a, right)
            else:
                if descending:
                    lower_b = max(lower_b, right)
                else:
                    upper_b = min(upper_b, right)
        else:
            if edge + 1 == leaf_index:
                if descending:
                    upper_a = min(upper_a, left)
                else:
                    lower_a = max(lower_a, left)
            else:
                if descending:
                    upper_b = min(upper_b, left)
                else:
                    lower_b = max(lower_b, left)

    if lower_a > upper_a or lower_b > upper_b:
        return None
    return lower_a, upper_a, lower_b, upper_b, relation


def _solve_two_values(mean_a, mean_b, weight_a, weight_b, constraints):
    lower_a, upper_a, lower_b, upper_b, relation = constraints
    a = float(np.clip(mean_a, lower_a, upper_a))
    b = float(np.clip(mean_b, lower_b, upper_b))
    if relation is None or (relation == "le" and a <= b) or (relation == "ge" and a >= b):
        return a, b

    lower = max(lower_a, lower_b)
    upper = min(upper_a, upper_b)
    if lower > upper:
        return None
    pooled = (weight_a * mean_a + weight_b * mean_b) / (weight_a + weight_b)
    pooled = float(np.clip(pooled, lower, upper))
    return pooled, pooled


def _squared_loss(y, weights, value):
    return float(np.dot(weights, (y - value) ** 2))
