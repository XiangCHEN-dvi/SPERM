"""Histogram gradient boosting regression with shape priors."""

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingRegressor as SKHistGradientBoostingRegressor,
)
from sklearn.utils.validation import validate_data

from ._base import _TreePriorMixin
from ._decision_tree import _validate_sample_weight
from ._unimodal_tree import _UnimodalTree


class GradientBoostingRegressor(_TreePriorMixin, SKHistGradientBoostingRegressor):
    """Histogram gradient boosting regression with shape priors."""

    def __init__(
        self,
        loss="squared_error",
        *,
        quantile=None,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features="from_dtype",
        interaction_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
        priors=None,
    ) -> None:
        super().__init__(
            loss=loss,
            quantile=quantile,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_features=max_features,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=None,
            interaction_cst=interaction_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )
        self.priors = priors

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        *,
        X_val=None,
        y_val=None,
        sample_weight_val=None,
    ):
        self._prepare_priors(X)
        if (
            self.unimodality_constraint_ is not None
            and self.unimodality_constraint_[1] == "monotonic"
        ):
            return self._fit_unknown_monotonic_direction(
                X,
                y,
                sample_weight,
                X_val=X_val,
                y_val=y_val,
                sample_weight_val=sample_weight_val,
            )
        if self.unimodality_constraint_ is not None:
            return self._fit_unimodality(
                X,
                y,
                sample_weight,
                X_val,
                y_val,
                sample_weight_val,
            )
        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            X_val=X_val,
            y_val=y_val,
            sample_weight_val=sample_weight_val,
        )

    def _fit_unimodality(
        self,
        X,
        y,
        sample_weight,
        X_val,
        y_val,
        sample_weight_val,
    ):
        if self.loss != "squared_error" or self.warm_start:
            raise ValueError(
                "Unimodal gradient boosting requires loss='squared_error' "
                "and warm_start=False."
            )
        if self.l2_regularization != 0:
            raise ValueError(
                "l2_regularization is not supported with unimodality."
            )
        if X_val is not None or y_val is not None or sample_weight_val is not None:
            raise ValueError(
                "Explicit validation data is not supported with unimodality."
            )
        X_array, y_array = validate_data(
            self,
            X,
            y,
            reset=True,
            y_numeric=True,
        )
        if y_array.ndim != 1:
            raise ValueError("Unimodality supports single-output regression.")
        weights = _validate_sample_weight(sample_weight, X_array.shape[0])
        effective_weights = (
            np.ones(X_array.shape[0], dtype=float) if weights is None else weights
        )
        self._unimodality_baseline_ = float(
            np.average(y_array, weights=effective_weights)
        )
        raw_predictions = np.full_like(
            y_array, self._unimodality_baseline_, dtype=float
        )
        self._unimodality_estimators_ = []
        turning_point = None
        _, kind = self.unimodality_constraint_

        for _ in range(self.max_iter):
            residual = y_array - raw_predictions
            tree = _UnimodalTree(
                kind=kind,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_split=2 * self.min_samples_leaf,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=0.0,
                value_bounds=(-np.inf, np.inf),
                turning_point=turning_point,
            ).fit(X_array, residual, weights)
            if turning_point is None:
                turning_point = tree.turning_point_
            update = tree.predict(X_array)
            raw_predictions += self.learning_rate * update
            self._unimodality_estimators_.append(tree)

        self.turning_point_ = turning_point
        self._predictors = [[tree] for tree in self._unimodality_estimators_]
        return self

    def _unimodality_predict(self, X):
        X_array = validate_data(self, X, reset=False)
        predictions = np.full(
            X_array.shape[0], self._unimodality_baseline_, dtype=float
        )
        for tree in self._unimodality_estimators_:
            predictions += self.learning_rate * tree.predict(X_array)
        return predictions


__all__ = ["GradientBoostingRegressor"]
