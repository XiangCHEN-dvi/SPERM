---
html_theme.sidebar_secondary.remove: true
---

# Shape-Prior-Embedded Regression Models

SPERM embeds global shape knowledge into familiar regression models. Add value
bounds, monotonicity, slope bounds, unimodality, or convexity through one
consistent, scikit-learn-compatible API.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item}
:columns: 12 7 7 7

```python
from sperm.linear_model import Ridge
from sperm.priors import Increasing, Lipschitz

model = Ridge(
    priors={0: (Increasing(), Lipschitz(2))}
).fit(X, y)

y_pred = model.predict(X_test)
```

:::

:::{grid-item-card} Start in minutes
:columns: 12 5 5 5
:class-card: sd-shadow-sm

Install SPERM, fit your first constrained model, and learn how priors are
attached to features.

```{button-ref} getting-started/index
:color: primary
:expand:
Getting started
```
:::
::::

## One Prior Language, Four Model Families

::::{grid} 1 2 2 4
:gutter: 2

:::{grid-item-card} Linear models
:link: user-guide/linear-models
:link-type: doc

Affine regression with exact coefficient constraints.
:::

:::{grid-item-card} Tree models
:link: user-guide/tree-models
:link-type: doc

Decision trees, random forests, and gradient boosting.
:::

:::{grid-item-card} Neural networks
:link: user-guide/mlp
:link-type: doc

Shape-preserving MLP and input-convex architectures.
:::

:::{grid-item-card} Gaussian processes
:link: user-guide/gaussian-process
:link-type: doc

Analytic uncertainty with a globally constrained posterior mean.
:::
::::

## Shape Guarantees Are Part of the Model

Priors are enforced over the complete input space, not checked only at the
training samples. SPERM validates combinations before fitting and rejects
requirements that would make a model trivial or mathematically inconsistent.

::::{grid} 1 3 3 3
:gutter: 2

:::{grid-item-card} Understand the semantics
:link: getting-started/shape-priors
:link-type: doc

Learn which properties are output-wide, feature-wise, or cross-feature.
:::

:::{grid-item-card} Explore complete examples
:link: examples/index
:link-type: doc

Compare constrained and unconstrained regressors on reproducible experiments.
:::

:::{grid-item-card} Use the Python API
:link: api/index
:link-type: doc

Browse public estimators, prior primitives, parameters, and methods.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
user-guide/index
examples/index
api/index
development/index
```
