"""Analytic finite-rank spline GP with a constrained posterior mean."""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.optimize import LinearConstraint, linprog, minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._basis import make_basis
from ._constraints import compile_gpr_constraint_candidates, normalize_gpr_priors


class GaussianProcessRegressor(RegressorMixin, BaseEstimator):
    """Additive P-spline GP whose posterior mean obeys global shape priors.

    The coefficient posterior is Gaussian. Shape priors are imposed by the
    Mahalanobis projection of its mean onto a polyhedron; posterior draws are
    therefore not shape constrained.
    """

    def __init__(
        self,
        *,
        n_basis=20,
        degree=3,
        alpha=1e-2,
        prior_variance=100.0,
        smoothness=1.0,
        normalize_y=False,
        optimizer_tol=1e-8,
        max_iter=1000,
        priors=None,
    ):
        self.n_basis = n_basis
        self.degree = degree
        self.alpha = alpha
        self.prior_variance = prior_variance
        self.smoothness = smoothness
        self.normalize_y = normalize_y
        self.optimizer_tol = optimizer_tol
        self.max_iter = max_iter
        self.priors = priors

    def fit(self, X, y, sample_weight=None):
        """Fit the analytic coefficient posterior and constrain its mean."""
        self._validate_parameters()
        X, y = validate_data(self, X, y, reset=True, y_numeric=True)
        if X.shape[0] < 2:
            raise ValueError("GaussianProcessRegressor requires more than one sample.")
        if y.ndim != 1:
            raise ValueError("GaussianProcessRegressor supports single-output regression.")
        weights = _sample_weight(sample_weight, X.shape[0])
        self.priors_ = normalize_gpr_priors(
            self.priors, X.shape[1], getattr(self, "feature_names_in_", None)
        )
        basis_data = X if weights is None else X[weights > 0]
        self.basis_ = make_basis(
            basis_data,
            n_basis=self.n_basis,
            degree=self.degree,
        )
        design = self.basis_.transform(X)
        effective_weights = np.ones(X.shape[0]) if weights is None else weights
        y_offset = float(np.average(y, weights=effective_weights)) if self.normalize_y else 0.0
        target = y - y_offset

        prior_precision = self._prior_precision()
        precision = prior_precision + (
            (design.T * effective_weights) @ design / self.alpha
        )
        rhs = design.T @ (effective_weights * target) / self.alpha
        factor = linalg.cho_factor(precision, lower=True, check_finite=False)
        unconstrained = linalg.cho_solve(factor, rhs, check_finite=False)
        unconstrained[0] += y_offset
        covariance = linalg.cho_solve(
            factor, np.eye(precision.shape[0]), check_finite=False
        )

        candidates = compile_gpr_constraint_candidates(
            self.basis_, self.priors_
        )
        self.n_unimodality_candidates_ = len(candidates)
        self.unconstrained_posterior_mean_ = unconstrained
        (
            self.posterior_mean_,
            constraint_matrix,
            constraint_bound,
            self.unimodality_candidate_,
        ) = self._project_candidates(
            unconstrained, precision, candidates
        )
        self.posterior_covariance_ = covariance
        self.prior_precision_ = prior_precision
        self.constraint_matrix_ = constraint_matrix
        self.constraint_bound_ = constraint_bound
        self.n_basis_features_ = design.shape[1]
        return self

    def predict(self, X, return_std=False, return_cov=False):
        """Return the analytic marginal or joint Gaussian prediction."""
        check_is_fitted(self, "posterior_mean_")
        if return_std and return_cov:
            raise ValueError("return_std and return_cov cannot both be True.")
        X = validate_data(self, X, reset=False)
        design = self.basis_.transform(X)
        mean = design @ self.posterior_mean_
        if return_cov:
            return mean, design @ self.posterior_covariance_ @ design.T
        if return_std:
            variance = np.einsum(
                "ij,jk,ik->i", design, self.posterior_covariance_, design
            )
            return mean, np.sqrt(np.maximum(variance, 0.0))
        return mean

    def sample_y(self, X, n_samples=1, random_state=None):
        """Sample the analytic Gaussian approximation."""
        mean, covariance = self.predict(X, return_cov=True)
        rng = np.random.default_rng(random_state)
        return rng.multivariate_normal(mean, covariance, size=n_samples).T

    def _prior_precision(self):
        precision = np.eye(self.basis_.n_coefficients) / self.prior_variance
        if self.smoothness == 0:
            return precision
        for feature in range(self.basis_.n_features):
            controls = self.basis_.control_map(feature)
            difference = np.diff(np.eye(self.n_basis), n=2, axis=0) @ controls
            precision += self.smoothness * difference.T @ difference
        return precision

    def _project_candidates(self, mean, precision, candidates):
        best = None
        iterations = 0
        for matrix, bound, label in candidates:
            try:
                projected, objective, n_iter = self._project(
                    mean, precision, matrix, bound
                )
            except ValueError:
                continue
            iterations += n_iter
            if best is None or objective < best[1]:
                best = (projected, objective, matrix, bound, label)
        if best is None:
            raise ValueError("The requested shape priors are infeasible.")
        self.n_iter_ = max(iterations, 1)
        return best[0], best[2], best[3], best[4]

    def _project(self, mean, precision, matrix, bound):
        if matrix.shape[0] == 0 or np.all(
            matrix @ mean <= bound + self.optimizer_tol
        ):
            return mean.copy(), 0.0, 1

        factor = linalg.cholesky(precision, lower=True, check_finite=False)
        whitened_matrix = linalg.solve_triangular(
            factor,
            matrix.T,
            lower=True,
            check_finite=False,
        ).T
        whitened_bound = bound - matrix @ mean
        feasibility = linprog(
            np.zeros_like(mean),
            A_ub=whitened_matrix,
            b_ub=whitened_bound,
            bounds=[(None, None)] * len(mean),
            method="highs",
        )
        if not feasibility.success:
            raise ValueError("The requested shape priors are infeasible.")

        def objective(value):
            return 0.5 * value @ value

        def gradient(value):
            return value

        result = minimize(
            objective,
            feasibility.x,
            jac=gradient,
            constraints=LinearConstraint(
                whitened_matrix,
                -np.inf,
                whitened_bound,
            ),
            method="SLSQP",
            options={"ftol": self.optimizer_tol, "maxiter": self.max_iter},
        )
        projected = mean + linalg.solve_triangular(
            factor.T,
            result.x,
            lower=False,
            check_finite=False,
        )
        if not result.success or np.any(matrix @ projected > bound + 1e-6):
            raise ValueError(f"The requested shape priors are infeasible: {result.message}")
        return projected, float(result.fun), max(int(result.nit), 1)

    def _validate_parameters(self):
        if not isinstance(self.degree, int) or self.degree < 2:
            raise ValueError("degree must be an integer of at least two.")
        if not isinstance(self.n_basis, int) or self.n_basis < self.degree + 1:
            raise ValueError("n_basis must be an integer greater than degree.")
        if not np.isscalar(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not np.isscalar(self.prior_variance) or self.prior_variance <= 0:
            raise ValueError("prior_variance must be positive.")
        if not np.isscalar(self.smoothness) or self.smoothness < 0:
            raise ValueError("smoothness must be non-negative.")
        if not isinstance(self.normalize_y, (bool, np.bool_)):
            raise TypeError("normalize_y must be a boolean.")
        if self.optimizer_tol <= 0 or self.max_iter <= 0:
            raise ValueError("optimizer_tol and max_iter must be positive.")


def _sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float)
    if (
        weights.shape != (n_samples,)
        or np.any(weights < 0)
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError(
            "sample_weight must be finite and non-negative with shape (n_samples,)."
        )
    if weights.sum() <= 0:
        raise ValueError("sample_weight must contain at least one non-zero value.")
    return weights


__all__ = ["GaussianProcessRegressor"]
