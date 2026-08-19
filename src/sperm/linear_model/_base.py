"""Shape-constrained linear regression estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import linalg
from scipy.optimize import lsq_linear
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from ..priors._compiler import compile_linear_priors


def solve_bounded_least_squares(X, y, bounds, *, tol=1e-10) -> np.ndarray:
    """Solve bounded least squares, including exactly fixed coefficients."""
    lower, upper = bounds
    fixed = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    coefficients = np.zeros(X.shape[1], dtype=X.dtype)

    if np.any(fixed):
        coefficients[fixed] = lower[fixed]
        y = y - X[:, fixed] @ coefficients[fixed]

    free = ~fixed
    if not np.any(free):
        return coefficients

    if np.all(np.isneginf(lower[free])) and np.all(np.isposinf(upper[free])):
        condition = max(X.shape) * np.finfo(X.dtype).eps
        coefficients[free] = linalg.lstsq(
            X[:, free],
            y,
            cond=condition,
            lapack_driver="gelsd",
        )[0]
        return coefficients

    result = lsq_linear(
        X[:, free],
        y,
        bounds=(lower[free], upper[free]),
        tol=tol,
    )
    if not result.success:
        raise RuntimeError(f"Least-squares optimization failed: {result.message}")
    coefficients[free] = result.x
    return coefficients


class BaseConstrainedLinearRegressor(RegressorMixin, BaseEstimator, ABC):
    """Shared implementation for constrained linear regressors."""

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        copy_X: bool = True,
        priors=None,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.priors = priors

    def fit(self, X, y, sample_weight=None):
        """Fit the constrained regression model."""
        self._validate_parameters()
        X, y = validate_data(
            self,
            X,
            y,
            reset=True,
            y_numeric=True,
            multi_output=False,
            dtype=[np.float64, np.float32],
            copy=self.copy_X,
        )

        weights = self._validate_sample_weight(sample_weight, X)
        self.priors_, bounds = compile_linear_priors(
            self.priors,
            n_features=X.shape[1],
            feature_names=getattr(self, "feature_names_in_", None),
        )
        self.coefficient_bounds_ = bounds

        X_centered, y_centered, X_offset, y_offset = self._center_data(
            X, y, weights
        )
        X_weighted, y_weighted = self._apply_sample_weight(
            X_centered, y_centered, weights
        )

        self.coef_ = self._solve(X_weighted, y_weighted, bounds)
        self.intercept_ = float(y_offset - X_offset @ self.coef_)
        return self

    def predict(self, X):
        """Predict target values for ``X``."""
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X = validate_data(
            self,
            X,
            reset=False,
            dtype=[np.float64, np.float32],
        )
        return X @ self.coef_ + self.intercept_

    def _validate_parameters(self) -> None:
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(self.copy_X, (bool, np.bool_)):
            raise TypeError("copy_X must be a boolean.")

    @staticmethod
    def _validate_sample_weight(sample_weight, X) -> np.ndarray | None:
        if sample_weight is None:
            return None

        weights = check_array(
            sample_weight,
            ensure_2d=False,
            dtype=X.dtype,
            input_name="sample_weight",
        )
        if weights.ndim != 1 or weights.shape[0] != X.shape[0]:
            raise ValueError("sample_weight must have shape (n_samples,).")
        if np.any(weights < 0):
            raise ValueError("sample_weight cannot contain negative values.")
        if not np.any(weights > 0):
            raise ValueError("sample_weight cannot contain only zero weights.")
        return weights

    def _center_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if not self.fit_intercept:
            return X, y, np.zeros(X.shape[1], dtype=X.dtype), 0.0

        X_offset = np.average(X, axis=0, weights=sample_weight)
        y_offset = float(np.average(y, weights=sample_weight))
        return X - X_offset, y - y_offset, X_offset, y_offset

    @staticmethod
    def _apply_sample_weight(
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if sample_weight is None:
            return X, y

        scale = np.sqrt(sample_weight)
        return X * scale[:, np.newaxis], y * scale

    @abstractmethod
    def _solve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Solve the estimator-specific constrained optimization problem."""


class LinearRegression(BaseConstrainedLinearRegressor):
    """Ordinary least squares with coefficient shape priors.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    copy_X : bool, default=True
        Whether input validation may copy ``X``.
    priors : Priors or mapping, default=None
        Per-feature slope priors. A mapping is accepted as shorthand.
        ``ValueBound`` is not supported by this base model.
    """

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        copy_X: bool = True,
        priors=None,
    ) -> None:
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            priors=priors,
        )

    def _solve(self, X, y, bounds) -> np.ndarray:
        return solve_bounded_least_squares(X, y, bounds)


__all__ = ["LinearRegression"]
