# SPERM

SPERM (Shape Prior Embedded Regression Models) targets
- providing flexible shape prior (nonnegativity, monotonicity, convexity, quasi-convexity, etc.) embeddings,
- into base regression models (linear models, tree-based models, gaussian process regressors, MLPs, etc.),
- with an API as compatible to [scikit-learn](https://scikit-learn.org/) as possible.

There have been many research works on this direction, but normally providing one or a few specific shape prior embeddings into one base model. We hope to fill the gap between research and application by integrating the proposed methods into one package.

A quick example:

```python
import numpy as np
from sperm.linear_model import Ridge

X = np.array([0, 1, 2, 3, 4]).reshape([-1, 1])
y = np.array([1, 3.1, 5.5, 7.5, 9.9])
shape_prior = ['0:increasing', '0:Lipschitz:2']
reg = Ridge(shape_prior=shape_prior).fit(X, y)
y_pred = reg.predict(X)
```

Detailed documentation is hosted on [readthedocs.org](???).

# Installation

SPERM is packaged and distributed with [PyPI](https://pypi.org/project/sperm/), hence can be easily installed by excuting

```shell
python3 -m pip install -U sperm
```

# Functionalities

An overall look at which shape priors are supported on which base models currently:

|                              | linear model | multi-layer perceptron | decision tree |
| ---------------------------- |:------------:|:----------------------:|:-------------:|
| nonnegative / nonpositive    |      X       |            √           |       √       |
| increasing / decreasing      |      √       |            √           |       √       |
| Lipschitz                    |      √       |            -           |       X       |
| quasi-convex / quasi-concave |      X       |            -           |       √       |
| convex / concave             |      X       |            √           |       X       |

- √: supported with universal approximation capability
- ⍻: supported without universal approximation capability
- -: not yet supported
- X: not supported (it is impossible or degrading to provide such shape priors on the base model)

# Known Limitations

- For now we only consider the situation that properties should hold for $\forall x \in \mathbb{R}$ for each feature dimension, while in some cases the properties are only expected to be held for certain intervals.
