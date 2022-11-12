import numpy as np
import numbers
from scipy.optimize import minimize
from sklearn.utils.validation import _check_sample_weight
from sklearn.linear_model import _BaseRidge as SKLearn_BaseRidge
from .._shape_prior import *

class Ridge(SKLearn_BaseRidge):
    def __init__(self, alpha=1.0, fit_intercept=True, copy_X=True,
                 max_iter=None, tol=1e-3, solver="auto", random_state=None,
                 shape_prior=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

        if shape_prior is None:
            self.shape_prior = ShapePrior('linear', [])
        elif type(shape_prior)==str:
            self.shape_prior = ShapePrior('linear', [shape_prior])
        elif type(shape_prior)==list:
            self.shape_prior = ShapePrior('linear', shape_prior)
        else:
            raise ValueError("Invalid shape_prior input.")

    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_data(
            X, y, y_numeric=True, multi_output=True
        )        

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        assert isinstance(alpha, numbers.Number):

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y,
            fit_intercept=self.fit_intercept,
            normalize=False,
            copy=self.copy_X,
            sample_weight=sample_weight,
            return_mean=True,
        )

        if sample_weight is not None:
            X, y = _rescale_data(X, y, sample_weight)

        bounds = [[-np.inf, np.inf]] * X.shape[1]

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

        def fun():
            return

        res = minimize(fun, x0, method='L-BFGS-B', bounds=(lb, ub))
        if res.success:
            self.coef_ = res.x
            self._set_intercept(X_offset, y_offset, X_scale)
            return self
        else:
            raise RuntimeError("fitting failed.")
        