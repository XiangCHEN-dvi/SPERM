# Declaring Priors

For feature-only constraints, a mapping is the shortest form:

```python
from sperm.priors import Increasing, Lipschitz

priors = {0: (Increasing(), Lipschitz(2.0))}
```

Keys are zero-based feature indices. A tuple means that every listed prior
must hold for that feature. Feature names are deliberately not inferred from
array or dataframe metadata.

Use the structured form for output bounds or cross-feature curvature:

```python
from sperm.priors import Convex, Priors, SlopeBound, ValueBound

priors = Priors(
    value=ValueBound(lower=0),
    features={0: SlopeBound(lower=0, upper=3)},
    curvature=Convex(),
)
```

Prefer canonical primitives in reusable configuration. Helpers such as
`Increasing()` remain convenient at call sites and parse to the same internal
representation.
