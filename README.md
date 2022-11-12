# SPERM

SPERM (Shape Prior Embedded Regression Models) targets providing flexible shape prior (nonnegativity, monotonicity, convexity, quasi-convexity, etc.) embedding into base regression models (linear models, tree-based models, gaussian process regressors, MLPs, etc.), with a [scikit-learn](https://scikit-learn.org/) compatible API. There have been many research works on this direction, but normally providing 1 specific shape prior embedding into 1 base model. We hope to integrate the proposed methods into one package to make it practical.

An overall look at which shape priors are supported on which base models:

|                              | linear models | polynomial models | tree-based models |  GPR  |  MLP  |
| ---------------------------- |:-------------:|:-----------------:|:-----------------:|:-----:|:-----:|
| nonnegative / nonpositive    |      X        |       √           |       √           |       |   √   |
| increasing / decreasing      |      √        |       √           |       √           |       |   √   |
| Lipschitz                    |      √        |       √           |       X           |       |   -   |
| quasi-convex / quasi-concave |      X        |       X           |       √           |       |   X   |
| convex / concave             |      X        |       √           |       X           |       |   √   |

- √: supported
- -: not yet supported
- X: not supported (it is impossible or degrading to provide such shape priors on the base model)

# Literature Review

## Survey

[Monotonic classification: an overview on algorithms, performance measures and data sets](https://arxiv.org/abs/1811.07155).

## Polynomial Regression

- [Fitting Monotonic Polynomials to Data, 1994](https://moam.info/fitting-monotonic-polynomials-to-data_5b94b281097c47f8618b46ae.html).
- [Revisiting fitting monotone polynomials to data, 2013](https://dl.acm.org/doi/abs/10.1007/s00180-012-0390-5).
- [Polynomial regression under shape constraints, 2014](https://hal.inria.fr/hal-01073514).
- [Fast and flexible methods for monotone polynomial fitting, 2016](https://www.tandfonline.com/doi/abs/10.1080/00949655.2016.1139582?journalCode=gscs20).
- [Fitting monotone polynomials in mixed effects models, 2019](https://dl.acm.org/doi/abs/10.1007/s11222-017-9797-8).

## Tree-Based Models

- [XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html). Abandon candidate splits if monotonic constraint will be violated.

## Multi-Layer Perceptrons

- [Monotonic Networks, NIPS 1997](https://papers.nips.cc/paper/1997/hash/83adc9225e4deb67d7ce42d58fe5157c-Abstract.html). Networks with monotonicity constraints have been studied dating back as early as year 1997. In this paper, it is achived by setting the network strucure as y=min(max(Wx)) where W is positive. The max operations allow to describe locally convex regions, and the min operation connects these local regions to guarantee the universal approximation capability.
- [Unconstrained Monotonic Neural Networks, NeurIPS 2019](https://arxiv.org/abs/1908.05164). Monotonicity is achieved by integrating a positive function.
- [Certified Monotonic Neural Networks, NeurIPS 2020](https://arxiv.org/abs/2011.10219). Monotonicity is guaranteed by solving a mixed integer linear programming problem.
- [Input Convex Neural Networks, ICML 2017](https://arxiv.org/abs/1609.07152). Convexity is guaranteed by utilizing the fact that the composition of a convex and convex non-decreasing function is also convex.
