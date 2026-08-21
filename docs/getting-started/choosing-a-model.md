# Choosing a Model

Choose the family from the structure and uncertainty your problem needs.

| Family | Best starting point | Important limitation |
| --- | --- | --- |
| Linear | Interpretable global effects and fast fitting | Value bounds would force a full-domain affine model to be constant |
| Tree | Nonlinear tabular data with value or monotonic constraints | Piecewise-constant predictions cannot have a finite nonzero slope bound |
| MLP | Flexible smooth functions and cross-feature convexity | Training is nonconvex; slope bounds are not implemented |
| GPR | Smooth estimates with analytic uncertainty | Uses a finite-rank additive spline representation |

See {doc}`shape-priors` for the precise support matrix and
{doc}`../examples/index` for side-by-side fits.
