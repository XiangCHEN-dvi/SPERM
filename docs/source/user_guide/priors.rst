Priors
======

Priors use explicit, feature-independent types. ``ValueBound`` bounds the model
output, ``Monotonicity`` specifies an order direction, ``SlopeBound`` bounds a
partial derivative, and ``Unimodality("minimum")`` /
``Unimodality("maximum")`` represent a coordinate-wise single valley or peak.
Semantic helper functions normalize
directly into these types::

   from sperm.priors import Increasing, Nonnegative
   from sperm.priors import Monotonicity, ValueBound

   assert Increasing() == Monotonicity("increasing")
   assert Nonnegative() == ValueBound(lower=0)

Per-feature monotonicity and slope priors can be passed as a mapping from
feature indices or column names::

   from sperm.linear_model import Ridge
   from sperm.priors import Increasing, Lipschitz

   model = Ridge(
       priors={
           0: (Increasing(), Lipschitz(2.0)),
       }
   )

Keeping the feature outside the prior object makes priors reusable and avoids
duplicating a feature identifier when several priors apply to the same feature.
For tabular input with string column names, the mapping key may be a column
name instead of an integer position.

Global value priors and per-feature slope priors can be combined with the
``Priors`` container::

   from sperm.priors import Priors, ValueBound

   priors = Priors(
       value=ValueBound(lower=0, upper=1),
       features={0: Increasing()},
   )

All priors currently apply over the complete input space. A nonconstant affine
function is unbounded above and below on that space, so applying any finite
``ValueBound`` to a linear model would necessarily fix all coefficients to
zero. ``LinearRegression`` and ``Ridge`` therefore reject ``ValueBound`` rather
than silently degrading to a constant model.

Text specifications are supported only through an explicit parser, which is
intended for configuration files and command-line interfaces::

   from sperm.priors import parse_priors

   priors = parse_priors([
       "nonnegative",
       "temperature:increasing",
       "temperature:Lipschitz:2",
   ])

Tree-based models
-----------------

The decision-tree, random-forest, and gradient-boosting regressors in
``sperm.tree_model`` accept ``ValueBound`` and ``Monotonicity``.
Monotonicity is enforced by the fitting algorithm. Value
bounds are applied as an exact prediction projection; because clipping is
monotone, combining both priors preserves monotonicity. Tree regressors reject
``SlopeBound`` because their piecewise-constant predictions are discontinuous.

``GradientBoostingRegressor`` uses scikit-learn's histogram gradient-boosting
implementation, which provides native monotonic constraints.

The three regressors also accept ``Unimodality("minimum")`` and
``Unimodality("maximum")`` for
single-feature regression. During tree growth, every candidate split is scored
with the best two child values that keep the complete ordered leaf sequence
single-valley or single-peak. Previously created splits and leaf values are not
changed. For random forests and gradient boosting, the first tree learns the
turning point and all later trees share it, ensuring that their average or sum
retains the requested shape::

   from sperm.priors import Unimodality
   from sperm.tree_model import RandomForestRegressor

   model = RandomForestRegressor(priors={0: Unimodality("minimum")})

Unimodality training currently requires exactly one input feature and squared
error. It does not use prediction-time leaf projection. When an unimodality
prior is combined with monotonicity, it is redundant and the compiler
normalizes the combination to monotonicity.
