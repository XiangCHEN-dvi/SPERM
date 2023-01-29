import numpy as np
import numbers
from scipy.optimize import minimize
from sklearn.utils.validation import _check_sample_weight
from sklearn.linear_model._ridge import _BaseRidge as SKLearn_BaseRidge
from sklearn.linear_model._base import _preprocess_data, _rescale_data
from .._shape_prior import *

class Ridge(SKLearn_BaseRidge):
    def __init__(self, alpha=1.0, fit_intercept=True, copy_X=True,
                 tol=1e-4, shape_prior=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.tol = tol

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

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y,
            fit_intercept=self.fit_intercept,
            normalize=False,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

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

        def func(w):
            residual = X.dot(w) - y
            f = 0.5 * residual.dot(residual) + 0.5 * self.alpha * w.dot(w)
            grad = X.T @ residual + self.alpha * w
            return f, grad

        res = minimize(func, np.zeros(X.shape[1]), method='L-BFGS-B', jac=True,
                       bounds=list(zip(lb, ub)), tol=self.tol)
        if res.success:
            self.coef_ = res.x
            self._set_intercept(X_offset, y_offset, X_scale)
            return self
        else:
            raise RuntimeError("fitting failed.")
        