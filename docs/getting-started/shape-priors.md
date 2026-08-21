# Shape Priors

SPERM separates priors by the part of the function they constrain.

Output
: `ValueBound` places global lower and/or upper bounds on predictions.

Feature-wise
: `Monotonicity`, `SlopeBound`, and `Unimodality` describe how the prediction
  changes along one feature while the others are held fixed.

Cross-feature
: `Convex` and `Concave` constrain the function jointly over its inputs.

Convenience constructors normalize to canonical primitives: `Nonnegative`
and `Nonpositive` become value bounds; `Increasing` and `Decreasing` become
monotonicity; and `Lipschitz(L)` becomes a slope interval `[-L, L]`.

## Current Support

```{include} ../../README.md
:start-after: <!-- functionality-matrix-start -->
:end-before: <!-- functionality-matrix-end -->
```

These symbols describe the implementation, not a claim that every prior is
appropriate for every dataset. See {doc}`guarantees` for what “global” means.
