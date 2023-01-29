# SPERM

SPERM (Shape Prior Embedded Regression Models) targets providing flexible shape prior (nonnegativity, monotonicity, convexity, quasi-convexity, etc.) embeddings into base regression models (linear models, tree-based models, gaussian process regressors, MLPs, etc.), with an API as compatible to [scikit-learn](https://scikit-learn.org/) as possible. There have been many research works on this direction, but normally providing one or a few specific shape prior embeddings into one base model. We hope to fill the gap between research and application by integrating the proposed methods into one package.

An overall look at which shape priors are supported on which base models currently:

|                              | linear models |
| ---------------------------- |:-------------:|
| nonnegative / nonpositive    |      X        |
| increasing / decreasing      |      √        |
| Lipschitz                    |      √        |
| quasi-convex / quasi-concave |      X        |
| convex / concave             |      X        |

- √: supported
- -: not yet supported
- X: not supported (it is impossible or degrading to provide such shape priors on the base model)
