"""Random forest regression with shape priors."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from sklearn.utils.validation import validate_data

from ._base import _TreePriorMixin
from ._bounded_tree import _BoundedTree
from ._decision_tree import (
    _validate_bounded_tree_parameters,
    _validate_sample_weight,
)
from ._unimodal_tree import _UnimodalTree


class RandomForestRegressor(_TreePriorMixin, SKRandomForestRegressor):
    """Random forest regression with value and monotonicity priors."""

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        priors=None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=None,
        )
        self.priors = priors

    def fit(self, X, y, sample_weight=None):
        self._prepare_priors(X)
        if (
            self.unimodality_constraint_ is not None
            and self.unimodality_constraint_[1] == "monotonic"
        ):
            return self._fit_unknown_monotonic_direction(X, y, sample_weight)
        if self.unimodality_constraint_ is not None:
            return self._fit_unimodality(X, y, sample_weight)
        if np.isfinite(self.value_bounds_).any():
            return self._fit_bounded(X, y, sample_weight)
        return super().fit(X, y, sample_weight=sample_weight)

    def _fit_unimodality(self, X, y, sample_weight):
        if self.criterion != "squared_error" or self.oob_score or self.warm_start:
            raise ValueError(
                "Unimodal random forests require criterion='squared_error', "
                "oob_score=False, and warm_start=False."
            )
        if self.ccp_alpha != 0 or self.min_weight_fraction_leaf != 0:
            raise ValueError(
                "ccp_alpha and min_weight_fraction_leaf are not supported "
                "with unimodality."
            )
        if not isinstance(self.min_samples_leaf, int) or not isinstance(
            self.min_samples_split, int
        ):
            raise TypeError(
                "Unimodality requires integer min_samples_leaf "
                "and min_samples_split."
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
        rng = np.random.default_rng(self.random_state)
        sample_size = _bootstrap_size(self.max_samples, X_array.shape[0])
        _, kind = self.unimodality_constraint_
        self.estimators_ = []
        turning_point = None

        for _ in range(self.n_estimators):
            indices = (
                rng.integers(0, X_array.shape[0], size=sample_size)
                if self.bootstrap
                else np.arange(X_array.shape[0])
            )
            tree_weights = None if weights is None else weights[indices]
            tree = _UnimodalTree(
                kind=kind,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                value_bounds=self.value_bounds_,
                turning_point=turning_point,
            ).fit(X_array[indices], y_array[indices], tree_weights)
            if turning_point is None:
                turning_point = tree.turning_point_
            self.estimators_.append(tree)

        self.turning_point_ = turning_point
        self.n_outputs_ = 1
        self._value_bounds_embedded_ = True
        return self

    def _fit_bounded(self, X, y, sample_weight):
        _validate_bounded_tree_parameters(self)
        if self.oob_score or self.warm_start:
            raise ValueError(
                "oob_score and warm_start are not supported with training-time "
                "ValueBound."
            )
        if not self.bootstrap and self.max_samples is not None:
            raise ValueError("max_samples requires bootstrap=True.")
        X_array, y_array = validate_data(
            self,
            X,
            y,
            reset=True,
            y_numeric=True,
        )
        if y_array.ndim != 1:
            raise ValueError("ValueBound supports single-output regression.")
        weights = _validate_sample_weight(sample_weight, X_array.shape[0])
        rng = np.random.default_rng(self.random_state)
        sample_size = _bootstrap_size(self.max_samples, X_array.shape[0])
        self._bounded_estimators_ = []

        for _ in range(self.n_estimators):
            indices = (
                rng.integers(0, X_array.shape[0], size=sample_size)
                if self.bootstrap
                else np.arange(X_array.shape[0])
            )
            tree_weights = None if weights is None else weights[indices]
            tree = _BoundedTree(
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                monotonic_cst=self.monotonic_cst,
                value_bounds=self.value_bounds_,
                random_state=int(rng.integers(np.iinfo(np.int32).max)),
            ).fit(X_array[indices], y_array[indices], tree_weights)
            self._bounded_estimators_.append(tree)

        self.estimators_ = self._bounded_estimators_
        self.n_outputs_ = 1
        self._value_bounds_embedded_ = True
        return self

    def _unimodality_predict(self, X):
        X_array = validate_data(self, X, reset=False)
        return np.mean([tree.predict(X_array) for tree in self.estimators_], axis=0)

    def _bounded_predict(self, X):
        X_array = validate_data(self, X, reset=False)
        return np.mean(
            [tree.predict(X_array) for tree in self._bounded_estimators_], axis=0
        )

    @property
    def feature_importances_(self):
        if hasattr(self, "_bounded_estimators_"):
            importances = np.mean(
                [tree.feature_importances_ for tree in self._bounded_estimators_],
                axis=0,
            )
            total = importances.sum()
            return importances / total if total > 0 else importances
        if getattr(self, "unimodality_constraint_", None) is not None:
            return np.mean(
                [tree.feature_importances_ for tree in self.estimators_], axis=0
            )
        return super().feature_importances_


def _bootstrap_size(max_samples, n_samples):
    if max_samples is None:
        return n_samples
    if isinstance(max_samples, int):
        if not 1 <= max_samples <= n_samples:
            raise ValueError("max_samples must be between 1 and n_samples.")
        return max_samples
    if isinstance(max_samples, float) and 0 < max_samples <= 1:
        return max(1, round(max_samples * n_samples))
    raise ValueError("max_samples must be an int, a float in (0, 1], or None.")


__all__ = ["RandomForestRegressor"]
