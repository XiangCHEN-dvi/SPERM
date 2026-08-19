Multi-Layer Perceptron
======================

``MLPRegressor`` is an independent NumPy implementation that follows the
public scikit-learn estimator protocol. It provides a model graph, analytical
backpropagation, mini-batch training, Adam and SGD, sample weights, warm start,
and early stopping without inheriting from scikit-learn's private MLP classes.

Supported priors
----------------

The implementation supports ``ValueBound``, per-feature ``Monotonicity``,
one-dimensional ``Unimodality``, ``Convex``, ``Concave``,
and their non-degrading combinations::

   from sperm.neural_network import MLPRegressor
   from sperm.priors import Convex, Increasing, Priors, ValueBound

   priors = Priors(
       value=ValueBound(lower=0),
       features={0: Increasing()},
       curvature=Convex(),
   )
   model = MLPRegressor(priors=priors, random_state=0)

An unconstrained model uses a standard dense MLP. A monotonic model uses
monotone activations and sign-parameterized paths. Convex models use an ICNN
with an input skip connection at every hidden layer, Softplus activations, and
nonnegative hidden-to-hidden and hidden-to-output weights. Concavity is
implemented as the negative of the convex graph.

Value bounds are architecture-level hard constraints. General dense models
use increasing Softplus or sigmoid output maps. A lower-bounded convex model
uses an increasing convex Softplus output, while an upper-bounded concave model
uses the corresponding negative construction.

One-dimensional unimodal models join decreasing and increasing monotone
branches at several unconstrained, trainable turning-point candidates.
Training uses one temperature to anneal both the Softplus hinges and softmax
candidate weights. The final stage and all predictions use exact ReLU hinges
and one argmax candidate, so the fitted model has a global hard single-valley
or single-peak guarantee. ``unimodality_n_candidates`` controls the candidate
count; ``unimodality_temperature`` and ``unimodality_soft_fraction`` control
annealing. Monotonicity and matching curvature priors remove redundant
unimodality constraints. Combining both modes, or combining unimodality with
the opposite curvature,
selects between increasing and decreasing constrained candidates.

Incompatible combinations
-------------------------

All priors apply on the complete input space. A globally convex function with
a finite upper bound, or a globally concave function with a finite lower
bound, must be constant. The compiler rejects these combinations instead of
silently degrading the model. Slope bounds are not implemented for MLPs.
Non-redundant unimodality currently requires exactly one input feature.
