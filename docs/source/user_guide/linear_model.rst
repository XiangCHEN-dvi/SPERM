Linear Model
============

For linear models, the shape prior embeddings are relatively trivial:

- It is degrading to consider nonnegative / nonpositive / convex / concave / unimodality priors for a linear model.
- For increasing / decreasing / Lipschitz priors, it means to add bound constraints to the slope of the model.

Both linear and ridge regression are implemented as scikit-learn-compatible
estimators using only its public API. Their constrained optimization problems
are solved with ``scipy.optimize.lsq_linear``. Ridge regression is expressed as
an augmented least-squares system, with the intercept excluded from
regularization.
