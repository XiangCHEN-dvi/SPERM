Linear Model
============

For linear models, the shape prior embeddings are relatively trivial:

- It is degrading to consider nonnegative / nonpositive / convex / concave / quasi-convex / quasi-concave priors for a linear model.
- For increasing / decreasing / Lipschitz priors, it means to add bound constraints to the slope of the model.

For now, we consider both linear and ridge regression models. For the LinearRegression class, the scipy.optimize.lsq_linear optimizer is used. For the Ridge class, the L-BFGS-B method through the scipy.optimize.minimize wrapper is utilized.
