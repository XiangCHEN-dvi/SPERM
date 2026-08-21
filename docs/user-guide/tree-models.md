# Tree Models

SPERM provides {class}`sperm.tree_model.DecisionTreeRegressor`,
{class}`sperm.tree_model.RandomForestRegressor`, and
{class}`sperm.tree_model.GradientBoostingRegressor`.

```python
from sperm.priors import Increasing, Priors, ValueBound
from sperm.tree_model import GradientBoostingRegressor

model = GradientBoostingRegressor(
    priors=Priors(
        value=ValueBound(lower=0, upper=10),
        features={0: Increasing()},
    ),
    random_state=0,
).fit(X, y)
```

Monotonicity is enforced while trees grow. Decision trees optimize leaf values
inside the requested value interval and evaluate candidate splits with the
resulting constrained loss. Random forests apply the same rule to every tree;
gradient boosting clips its final additive prediction. One-dimensional
unimodality accepts a split only when the complete ordered leaf sequence
remains single-valley or single-peak; forests and boosting share the turning
point learned by the first tree.

Unimodality currently requires one input feature and squared-error training.
Slope bounds and cross-feature convexity are incompatible with nonconstant
piecewise-constant predictions.
