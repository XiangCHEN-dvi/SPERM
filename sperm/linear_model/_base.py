import numpy as np
from scipy.optimize import lsq_linear
from sklearn.utils.validation import _check_sample_weight
from sklearn.linear_model import LinearRegression as SKLearnLinearRegression
from sklearn.linear_model._base import _preprocess_data, _rescale_data
from .._shape_prior import *

class LinearRegression(SKLearnLinearRegression):
    def __init__(self, fit_intercept=True, copy_X=True, shape_prior=None):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

        if shape_prior is None:
            self.shape_prior = ShapePrior('linear', [])
        elif type(shape_prior)==str:
            self.shape_prior = ShapePrior('linear', [shape_prior])
        elif type(shape_prior)==list:
            self.shape_prior = ShapePrior('linear', shape_prior)
        else:
            raise ValueError("Invalid shape_prior input.")
        
    def fit(self, X, y, sample_weight=None):
        self._validate_params()
        X, y = self._validate_data(
            X, y, y_numeric=True, multi_output=True
        )        

        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=X.dtype, only_non_negative=True
        )

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y,
            fit_intercept=self.fit_intercept,
            normalize=False,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        X, y, _ = _rescale_data(X, y, sample_weight)

        lb = [-np.inf] * X.shape[1]
        ub = [np.inf] * X.shape[1]
        for p in self.shape_prior.prior_list:
            if p[1]=='increasing':
                lb[p[0]] = max(0, lb[p[0]])
            elif p[1]=='decreasing':
                ub[p[0]] = min(0, ub[p[0]])
            elif p[1]=='Lipschitz':
                lb[p[0]] = max(-p[2], lb[p[0]])
                ub[p[0]] = min( p[2], ub[p[0]])

        res = lsq_linear(X, y, bounds=(lb, ub))
        if res.success:
            self.coef_ = res.x
            self._set_intercept(X_offset, y_offset, X_scale)
            return self
        else:
            raise RuntimeError("fitting failed.")
        