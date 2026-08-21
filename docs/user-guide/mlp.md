# Neural Networks

{class}`sperm.neural_network.MLPRegressor` implements its own public estimator,
forward pass, backpropagation, Adam/SGD optimization, and early stopping. It
does not depend on scikit-learn's private MLP implementation.

```python
from sperm.neural_network import MLPRegressor
from sperm.priors import Convex, Increasing, Priors, ValueBound

model = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    priors=Priors(
        value=ValueBound(lower=0),
        features={0: Increasing()},
        curvature=Convex(),
    ),
    random_state=0,
).fit(X, y)
```

Monotonic paths use sign-parameterized weights. Convexity uses an
input-convex neural network with input skip connections and nonnegative
recurrent weights; concavity uses its negative form. One-dimensional
unimodality trains several turning-point candidates, anneals a Softplus hinge
and softmax selection, then finishes with an exact ReLU hinge and one candidate.

Slope bounds are not implemented. Some finite output-bound and curvature
combinations are rejected because the bounding transform would destroy the
requested curvature.
