# Combining Priors

Use {class}`sperm.priors.Priors` when a specification contains both output and
feature properties:

```python
from sperm.priors import Convex, Increasing, Priors, ValueBound

priors = Priors(
    value=ValueBound(lower=0),
    features={0: Increasing()},
    curvature=Convex(),
)
```

The compiler canonicalizes helpers, intersects compatible bounds, removes
redundancies, and rejects contradictions. For example, increasing plus
minimum-mode unimodality reduces to increasing; two slope bounds become their
intersection.

Combination support still depends on the model architecture. A transformation
that enforces a finite value bound need not preserve convexity, so the MLP uses
compatible parameterizations instead of blindly stacking output transforms.
Consult the relevant {doc}`../user-guide/index` page before relying on a complex
combination.
