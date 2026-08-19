"""Decision tree regression with shape priors."""

import numpy as np
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
from sklearn.utils.validation import validate_data

from ._base import _TreePriorMixin
from ._unimodal_tree import _UnimodalTree


class DecisionTreeRegressor(_TreePriorMixin, SKDecisionTreeRegressor):
    """Decision tree regression with value and monotonicity priors."""

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        priors=None,
    ) -> None:
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            monotonic_cst=None,
        )
        self.priors = priors

    def fit(self, X, y, sample_weight=None, check_input=True):
        self._prepare_priors(X)
        if (
            self.unimodality_constraint_ is not None
            and self.unimodality_constraint_[1] == "monotonic"
        ):
            return self._fit_unknown_monotonic_direction(
                X, y, sample_weight, check_input=check_input
            )
        if self.unimodality_constraint_ is not None:
            return self._fit_unimodality(X, y, sample_weight)
        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
        )

    def _fit_unimodality(self, X, y, sample_weight):
        if self.criterion != "squared_error" or self.splitter != "best":
            raise ValueError(
                "Unimodality requires criterion='squared_error' "
                "and splitter='best'."
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
        _, kind = self.unimodality_constraint_
        self._unimodal_tree_ = _UnimodalTree(
            kind=kind,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            value_bounds=self.value_bounds_,
        ).fit(X_array, y_array, weights)
        self.n_outputs_ = 1
        self.turning_point_ = self._unimodal_tree_.turning_point_
        return self

    def _unimodality_predict(self, X):
        X_array = validate_data(self, X, reset=False)
        return self._unimodal_tree_.predict(X_array)

    @property
    def feature_importances_(self):
        if getattr(self, "unimodality_constraint_", None) is not None:
            return self._unimodal_tree_.feature_importances_
        return super().feature_importances_

    def get_depth(self):
        if getattr(self, "unimodality_constraint_", None) is not None:
            return self._unimodal_tree_.get_depth()
        return super().get_depth()

    def get_n_leaves(self):
        if getattr(self, "unimodality_constraint_", None) is not None:
            return self._unimodal_tree_.get_n_leaves()
        return super().get_n_leaves()


def _validate_sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (n_samples,):
        raise ValueError("sample_weight must have shape (n_samples,).")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must contain finite non-negative values.")
    return weights


__all__ = ["DecisionTreeRegressor"]
