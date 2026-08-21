"""Show gradient boosting with every non-redundant supported prior subset."""

import matplotlib.pyplot as plt
import numpy as np
from _combination_plot import Combination, plot_combinations

from sperm.priors import Increasing, Priors, Unimodality, ValueBound
from sperm.tree_model import GradientBoostingRegressor


def signal(x):
    """A bounded increasing signal, hence also minimum-mode unimodal."""
    return 1.2 + 0.75 * np.tanh(0.9 * x)


def make_model(priors):
    return GradientBoostingRegressor(
        max_iter=120,
        max_leaf_nodes=8,
        min_samples_leaf=3,
        learning_rate=0.08,
        early_stopping=False,
        random_state=0,
        priors=priors,
    )


def plot_example():
    rng = np.random.default_rng(4)
    X = np.linspace(-3, 3, 60).reshape(-1, 1)
    y = signal(X[:, 0]) + 0.22 * np.sin(5 * X[:, 0])
    y += rng.normal(0, 0.16, X.shape[0])
    grid = np.linspace(-3.4, 3.4, 500).reshape(-1, 1)
    value = ValueBound(lower=0.5, upper=2.0)
    increasing = Increasing()
    unimodal = Unimodality("minimum")
    rows = (
        (
            Combination("ValueBound", Priors(value=value)),
            Combination("Monotonicity", Priors(features={0: increasing})),
            Combination("Unimodality", Priors(features={0: unimodal})),
        ),
        (
            Combination(
                "ValueBound + Monotonicity",
                Priors(value=value, features={0: increasing}),
            ),
            Combination(
                "ValueBound + Unimodality",
                Priors(value=value, features={0: unimodal}),
            ),
        ),
    )
    return plot_combinations(
        X=X,
        y=y,
        grid=grid,
        signal=signal,
        rows=rows,
        make_model=make_model,
        output_name="tree_model_prior_combinations.png",
    )


if __name__ == "__main__":
    figure, output = plot_example()
    print(f"Saved figure to {output}")
    plt.show()
