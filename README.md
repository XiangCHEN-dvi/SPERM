# SPERM

SPERM (Shape-Prior-Embedded Regression Models) embeds shape priors such as
value bounds, monotonicity, and slope bounds into regression models while
keeping an API compatible with scikit-learn.

## Quick start

```python
import numpy as np

from sperm.linear_model import Ridge
from sperm.priors import Increasing, Lipschitz

X = np.arange(5).reshape(-1, 1)
y = np.array([1, 3.1, 5.5, 7.5, 9.9])

regressor = Ridge(
    priors={0: (Increasing(), Lipschitz(2))},
).fit(X, y)
y_pred = regressor.predict(X)
```

## Prior model

Priors are normalized into explicit, feature-independent types:

| Semantic helper | Canonical prior                 |
| --------------- | ------------------------------- |
| `Nonnegative()` | `ValueBound(lower=0)`           |
| `Nonpositive()` | `ValueBound(upper=0)`           |
| `Increasing()`  | `Monotonicity("increasing")`    |
| `Decreasing()`  | `Monotonicity("decreasing")`    |
| `Lipschitz(L)`  | `SlopeBound(lower=-L, upper=L)` |

`Unimodality("minimum")` and `Unimodality("maximum")` require a feature slice
to have a single valley or peak, respectively, while all other features are
held fixed. The current implementations support this coordinate-wise property
only when the model has one input feature.

Global value priors and per-feature slope priors can be combined explicitly:

```python
from sperm.priors import Increasing, Priors, ValueBound

priors = Priors(
    value=ValueBound(lower=0, upper=10),
    features={0: Increasing()},
)
```

All priors apply over the complete input space. A finite `ValueBound` would
force an affine linear model to be constant, so `LinearRegression` and `Ridge`
reject it instead of silently degrading the model.

## Multilayer perceptron

`MLPRegressor` implements its own NumPy forward pass, backpropagation,
mini-batch training, Adam/SGD optimizers, and early-stopping pipeline while
following the public scikit-learn estimator protocol. It does not inherit from
scikit-learn's private multilayer-perceptron implementation.

```python
from sperm.neural_network import MLPRegressor
from sperm.priors import Convex, Increasing, Priors, ValueBound

priors = Priors(
    value=ValueBound(lower=0),
    features={0: Increasing()},
    curvature=Convex(),
)
model = MLPRegressor(priors=priors, random_state=0).fit(X, y)
```

Without curvature, value bounds use monotone output transformations and
monotonicity uses sign-parameterized paths. Convexity uses an input-convex
neural network with input skip connections and nonnegative recurrent weights;
concavity uses its negative form. One-dimensional unimodality uses two
monotone branches with several unconstrained, trainable turning-point
candidates. A shared temperature anneals both the Softplus hinges and the
softmax candidate weights; training then finishes with exact ReLU hinges and
one argmax candidate, guaranteeing the global shape. The compiler rejects
globally degrading combinations such as convexity
with a finite upper bound or concavity with a finite lower bound. Slope bounds
are not implemented for MLPs yet.

## Tree-based models

The tree API includes a decision tree, a random forest, and histogram gradient
boosting exposed under the familiar gradient-boosting name:

```python
from sperm.priors import Increasing, Priors, ValueBound
from sperm.tree_model import (
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

priors = Priors(
    value=ValueBound(lower=0, upper=10),
    features={0: Increasing()},
)

models = [
    DecisionTreeRegressor(priors=priors),
    RandomForestRegressor(priors=priors),
    GradientBoostingRegressor(priors=priors),
]
```

Monotonicity is enforced during fitting by the underlying tree algorithms.
Value bounds are enforced exactly on public predictions by clipping their
outputs; clipping preserves monotonicity. Tree models reject `SlopeBound`,
because a nonconstant piecewise-constant function has no finite global slope
bound.

All three tree regressors also support one-dimensional
`Unimodality("minimum")` and `Unimodality("maximum")` priors. Their constrained
grower accepts a split only when
replacing its parent leaf with the two optimized child values leaves the full,
ordered leaf sequence single-valley or single-peak. Existing splits and leaf
values are never revised. Random forest and gradient boosting learn the
turning point from the first tree and share it across later trees, so averaging
or adding the trees preserves the global shape.

```python
from sperm.priors import Unimodality

model = RandomForestRegressor(priors={0: Unimodality("minimum")}).fit(X, y)
```

Unimodality currently requires exactly one input feature and squared-error
training. When combined with monotonicity, it is redundant and is automatically
normalized to the monotonicity prior.

## Gaussian process

`GaussianProcessRegressor` is a finite-rank, additive Gaussian process built
from a common clamped B-spline basis and a P-spline Gaussian smoothness prior.
Gaussian observation noise gives a closed-form coefficient posterior. The
estimator then projects only that posterior's mean into the requested global
shape class, while retaining its analytic covariance:

```python
from sperm.gaussian_process import GaussianProcessRegressor
from sperm.priors import Convex, Increasing, Priors, ValueBound

model = GaussianProcessRegressor(
    n_basis=20,
    priors=Priors(
        value=ValueBound(lower=0),
        features={0: Increasing()},
        curvature=Convex(),
    ),
).fit(X, y)

mean, std = model.predict(X, return_std=True)
```

The resulting Gaussian is the closest one, in the posterior precision metric,
whose mean coefficients satisfy the priors. Consequently, the posterior mean
is constrained globally and `predict(..., return_std=True/return_cov=True)`
remains analytic; posterior draws from `sample_y` are not shape constrained.

The basis is identical with and without priors. `ValueBound`, `Monotonicity`,
`SlopeBound`, and `Convex`/`Concave` compile to linear constraints on spline
controls and their first or second divided differences. Linear extrapolation
extends derivative and curvature constraints to the complete real line;
value bounds additionally constrain the tail slopes. Constraint counts grow
linearly with `n_basis`. Mathematically degrading full-domain combinations,
including convexity with a finite upper bound and concavity with a finite lower
bound, are rejected. Global value bounds currently require one input feature.
One-dimensional minimum-mode and maximum-mode unimodality enumerate the
possible turning positions of the derivative spline. Each position is a small convex
quadratic program with linear constraints; the closest constrained posterior
mean is selected. These priors can be combined with value, slope, monotonicity,
and curvature priors. Redundant combinations are simplified before fitting.

Text specifications are available as an explicit configuration boundary:

```python
from sperm.priors import parse_priors

priors = parse_priors([
    "nonnegative",
    "0:increasing",
    "0:Lipschitz:2",
])
```

## Installation

```shell
python -m pip install -U sperm
```

## Functionalities

The following table describes what is technically achievable over the complete
input space. It is a design map rather than a statement about the current
implementation.

|                  | linear model | Tree-based models |  MLP  |  GPR  |
| ---------------- | :----------: | :---------------: | :---: | :---: |
| ValueBound       |      X       |         √         |   √   |   √   |
| Monotonicity     |      √       |         √         |   √   |   √   |
| SlopeBound       |      √       |         X         |   -   |   √   |
| Unimodality      |      X       |         √         |   √   |   √   |
| Convexity        |      X       |         X         |   √   |   √   |

- **√**: supported.
- **-**: not yet supported.
- **X**: not supported (it is trivial or impossible or degrading to provide such shape priors on the base model).



Interpretation by model family:

- **Linear model:** coefficient and norm restrictions give exact slope bounds.
  A finite global value bound would force all slopes to zero and is therefore
  rejected. Affine functions are both convex and concave, and are unimodal in
  both modes, but this is only a trivial subset of those function classes.
- **MLP:** bounded output transformations, monotone architectures, universal
  Lipschitz architectures, and input-convex neural networks provide strong
  constructions. Unimodality needs a specialized architecture or optimization
  formulation.
- **Tree-based models:** bounded leaf values give exact value bounds. Standard
  trees can enforce monotonicity by coordinating leaf predictions across
  ordered splits, and monotone step functions retain broad approximation power.
  However, piecewise-constant trees are discontinuous, so a nonconstant tree
  cannot have a finite global slope or Lipschitz bound, or be globally convex.
  Those combinations require continuous piecewise-linear model trees,
  constrained leaves, or a post-processing/projection method. Unimodality
  additionally requires globally coordinated leaf regions rather than
  independent split constraints.
- **GPR:** the clamped B-spline/P-spline sieve is dense in the corresponding
  value-bounded, derivative-bounded, monotone, unimodal, and
  convex/concave function classes on compact subsets as `n_basis` grows.
  Linear constraints on spline controls and divided differences guarantee each
  convex candidate over the complete input space. Unimodality uses a linear
  number of turning-point candidates rather than a single convex feasible set.

Representative references include
[universal Lipschitz GroupSort networks](https://proceedings.mlr.press/v97/anil19a.html),
[input-convex neural networks](https://proceedings.mlr.press/v70/amos17b.html),
[continuous piecewise-linear decision trees](https://doi.org/10.1016/j.eswa.2020.114173),
and
[Gaussian processes with linear-operator inequality constraints](https://jmlr.org/papers/v20/19-065.html).

## Known limitations

- All models currently support single-output regression only.
- Priors currently hold over the complete input space rather than a selected
  interval or domain.
