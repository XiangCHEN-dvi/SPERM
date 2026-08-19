"""Shape-constrained ridge regression."""

from __future__ import annotations

from numbers import Real

import numpy as np

from ._base import BaseConstrainedLinearRegressor, solve_bounded_least_squares


class Ridge(BaseConstrainedLinearRegressor):
    """L2-regularized linear regression with coefficient shape priors.

    Parameters
    ----------
    alpha : float, default=1.0
        Non-negative L2 regularization strength.
    fit_intercept : bool, default=True
        Whether to fit an intercept. The intercept is not regularized.
    copy_X : bool, default=True
        Whether input validation may copy ``X``.
    tol : float, default=1e-6
        Convergence tolerance passed to ``scipy.optimize.lsq_linear``.
    priors : Priors or mapping, default=None
        Per-feature slope priors. A mapping is accepted as shorthand.
        ``ValueBound`` is not supported by this base model.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        copy_X: bool = True,
        tol: float = 1e-6,
        priors=None,
    ) -> None:
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            priors=priors,
        )
        self.alpha = alpha
        self.tol = tol

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if not isinstance(self.alpha, Real) or isinstance(
            self.alpha, (bool, np.bool_)
        ):
            raise TypeError("alpha must be a real number.")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be non-negative and finite.")
        if not isinstance(self.tol, Real) or isinstance(
            self.tol, (bool, np.bool_)
        ):
            raise TypeError("tol must be a real number.")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be positive and finite.")

    def _solve(self, X, y, bounds) -> np.ndarray:
        if self.alpha > 0:
            regularizer = np.sqrt(self.alpha) * np.eye(X.shape[1], dtype=X.dtype)
            X = np.vstack((X, regularizer))
            y = np.concatenate((y, np.zeros(X.shape[1], dtype=y.dtype)))

        return solve_bounded_least_squares(X, y, bounds, tol=self.tol)


__all__ = ["Ridge"]
