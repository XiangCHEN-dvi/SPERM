"""Shared behavior for shape-prior tree regressors."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone

from ..priors._priors import Monotonicity, Priors
from ..priors._tree_compiler import compile_tree_priors


class _TreePriorMixin:
    """Compile tree priors and project predictions into value bounds."""

    def _prepare_priors(self, X) -> None:
        if self.priors is None:
            self.priors_ = None
            self.value_bounds_ = (-np.inf, np.inf)
            self.monotonic_cst = None
            self.unimodality_constraint_ = None
            return

        feature_names = None
        if hasattr(X, "columns") and all(isinstance(name, str) for name in X.columns):
            feature_names = np.asarray(X.columns, dtype=object)

        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError("Expected a 2-dimensional feature matrix.")

        (
            self.priors_,
            monotonic_cst,
            self.value_bounds_,
            self.unimodality_constraint_,
        ) = compile_tree_priors(
            self.priors,
            n_features=X_array.shape[1],
            feature_names=feature_names,
        )
        self.monotonic_cst = monotonic_cst

        if self.unimodality_constraint_ is not None and X_array.shape[1] != 1:
            raise ValueError(
                "Unimodality currently requires exactly one input feature."
            )

    def predict(self, X):
        if hasattr(self, "_monotonic_candidate_estimator_"):
            predictions = self._monotonic_candidate_estimator_.predict(X)
        elif (
            hasattr(self, "_unimodality_predict")
            and getattr(self, "unimodality_constraint_", None) is not None
        ):
            predictions = self._unimodality_predict(X)
        else:
            predictions = super().predict(X)
        return np.clip(predictions, *self.value_bounds_)

    def _fit_unknown_monotonic_direction(
        self,
        X,
        y,
        sample_weight=None,
        **fit_params,
    ):
        """Fit both monotonic directions and retain the lower-loss candidate."""
        feature, kind = self.unimodality_constraint_
        if kind != "monotonic":
            raise RuntimeError("Expected an unknown-direction monotonic constraint.")

        candidates = []
        y_array = np.asarray(y, dtype=float)
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        for direction in ("increasing", "decreasing"):
            features = dict(self.priors_.features)
            features[feature] = (Monotonicity(direction),)
            priors = Priors(
                value=self.priors_.value,
                features=features,
                curvature=self.priors_.curvature,
            )
            candidate = clone(self).set_params(priors=priors)
            candidate.fit(X, y, sample_weight=sample_weight, **fit_params)
            residual = candidate.predict(X) - y_array
            loss = float(
                np.mean(residual**2)
                if weights is None
                else np.average(residual**2, weights=weights)
            )
            candidates.append((loss, candidate, direction))

        _, selected, direction = min(candidates, key=lambda item: item[0])
        self._monotonic_candidate_estimator_ = selected
        self.monotonic_direction_ = direction
        for name, value in vars(selected).items():
            if name.endswith("_") and name not in {
                "priors_",
                "unimodality_constraint_",
                "value_bounds_",
            }:
                setattr(self, name, value)
        return self
