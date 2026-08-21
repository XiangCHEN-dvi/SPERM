# Guarantees and Uncertainty

Unless a model page says otherwise, priors apply over the complete input
space. Enforcement is architectural, algebraic, or part of tree construction;
it is not a penalty evaluated only on sampled points.

The guarantee concerns the public prediction function. In particular, the
Gaussian-process estimator constrains the posterior mean while retaining a
closed-form Gaussian coefficient covariance. Its uncertainty bands and
posterior samples are therefore not themselves shape constrained.

Some full-domain requirements collapse a model. A globally bounded affine
function, for example, must be constant. A convex function with a finite
global upper bound is also constant. SPERM rejects such combinations rather
than silently returning a less expressive estimator.

Tree predictions are piecewise constant. They can be monotone, but a
nonconstant tree cannot obey a finite classical slope bound across a jump.
This is why monotonicity and slope bounds remain distinct primitives.
