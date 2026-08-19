"""Clamped B-spline feature maps for finite-rank Gaussian processes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class FeatureSpline:
    """One-dimensional B-spline with linear extrapolation at both tails."""

    knots: np.ndarray
    degree: int
    lower: float
    upper: float

    @property
    def n_basis(self) -> int:
        return len(self.knots) - self.degree - 1

    def design(self, x, derivative=0):
        """Evaluate basis functions or derivatives with linear tails."""
        values = np.asarray(x, dtype=float)
        clipped = np.clip(values, self.lower, self.upper)
        spline = BSpline(self.knots, np.eye(self.n_basis), self.degree)
        if derivative <= self.degree:
            result = spline(clipped, nu=derivative)
        else:
            result = np.zeros((len(values), self.n_basis))

        outside_left = values < self.lower
        outside_right = values > self.upper
        if derivative == 0:
            if np.any(outside_left):
                value = spline(self.lower)
                slope = spline(self.lower, nu=1)
                result[outside_left] = value + np.outer(
                    values[outside_left] - self.lower, slope
                )
            if np.any(outside_right):
                value = spline(self.upper)
                slope = spline(self.upper, nu=1)
                result[outside_right] = value + np.outer(
                    values[outside_right] - self.upper, slope
                )
        elif derivative == 1:
            result[outside_left] = spline(self.lower, nu=1)
            result[outside_right] = spline(self.upper, nu=1)
        else:
            result[outside_left | outside_right] = 0.0
        return np.asarray(result)

    def derivative_coefficients(self, order):
        """Map spline control points to derivative B-spline coefficients."""
        if order < 0 or order > self.degree:
            raise ValueError("order must lie between zero and the spline degree.")
        transform = np.eye(self.n_basis)
        knots = self.knots
        degree = self.degree
        for _ in range(order):
            denominator = knots[degree + 1 : -1] - knots[1 : -degree - 1]
            difference = np.zeros((transform.shape[0] - 1, transform.shape[0]))
            rows = np.arange(difference.shape[0])
            difference[rows, rows] = -degree / denominator
            difference[rows, rows + 1] = degree / denominator
            transform = difference @ transform
            knots = knots[1:-1]
            degree -= 1
        return transform


@dataclass(frozen=True)
class AdditiveSplineBasis:
    """Identifiable additive spline basis with one shared intercept."""

    features: tuple[FeatureSpline, ...]

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_basis(self) -> int:
        return self.features[0].n_basis

    @property
    def n_coefficients(self) -> int:
        return 1 + self.n_features * (self.n_basis - 1)

    def transform(self, X):
        columns = [np.ones(len(X))]
        for feature, spline in enumerate(self.features):
            columns.append(spline.design(X[:, feature])[:, 1:])
        return np.column_stack(columns)

    def control_map(self, feature):
        """Map model coefficients to all controls of one additive component."""
        mapping = np.zeros((self.n_basis, self.n_coefficients))
        start = 1 + feature * (self.n_basis - 1)
        mapping[1:, start : start + self.n_basis - 1] = np.eye(
            self.n_basis - 1
        )
        return mapping

    def value_control_map(self):
        """Map coefficients to total spline controls for a scalar input."""
        if self.n_features != 1:
            raise ValueError("Global ValueBound currently requires one input feature.")
        mapping = self.control_map(0)
        mapping[:, 0] = 1.0
        return mapping

    def derivative_control_map(self, feature, order):
        return (
            self.features[feature].derivative_coefficients(order)
            @ self.control_map(feature)
        )

    def endpoint_derivative_map(self, feature):
        derivative = self.derivative_control_map(feature, 1)
        return derivative[0], derivative[-1]


def make_basis(X, *, n_basis, degree):
    """Construct the same clamped spline space regardless of shape priors."""
    features = []
    n_interior = n_basis - degree - 1
    for feature in range(X.shape[1]):
        values = X[:, feature]
        lower = float(np.min(values))
        upper = float(np.max(values))
        if not lower < upper:
            raise ValueError("Each input feature must contain at least two values.")
        interior = (
            np.linspace(lower, upper, n_interior + 2)[1:-1]
            if n_interior
            else np.empty(0)
        )
        knots = np.concatenate(
            (
                np.repeat(lower, degree + 1),
                interior,
                np.repeat(upper, degree + 1),
            )
        )
        features.append(FeatureSpline(knots, degree, lower, upper))
    return AdditiveSplineBasis(tuple(features))


__all__ = ["AdditiveSplineBasis", "FeatureSpline", "make_basis"]
