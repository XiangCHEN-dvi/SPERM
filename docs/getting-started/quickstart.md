# Quickstart

Fit a ridge model whose prediction must increase with the first feature and
whose slope must stay between 0 and 2 everywhere:

```python
import numpy as np

from sperm.linear_model import Ridge
from sperm.priors import Increasing, SlopeBound

X = np.linspace(-2, 2, 40).reshape(-1, 1)
y = X[:, 0] ** 3 + np.random.default_rng(0).normal(scale=0.5, size=40)

model = Ridge(
    alpha=1.0,
    priors={0: (Increasing(), SlopeBound(upper=2.0))},
).fit(X, y)

y_pred = model.predict(X)
```

For priors that concern the output itself, use {class}`sperm.priors.Priors`:

```python
from sperm.priors import Priors, ValueBound
from sperm.tree_model import RandomForestRegressor

model = RandomForestRegressor(
    priors=Priors(value=ValueBound(lower=0, upper=10)),
    random_state=0,
).fit(X, y)
```

The model validates and normalizes the specification during `fit`. Unsupported
or degrading combinations raise an explicit error.
