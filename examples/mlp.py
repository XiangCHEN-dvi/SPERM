"""Show an MLP with all subsets of three compatible shape priors."""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from _combination_plot import Combination, plot_combinations

from sperm.neural_network import MLPRegressor
from sperm.priors import Convex, Increasing, Priors, ValueBound


def signal(x):
    """Softplus is nonnegative, increasing, and convex on the real line."""
    return np.logaddexp(0, x)


def make_model(priors):
    return MLPRegressor(
        hidden_layer_sizes=(12,),
        learning_rate_init=0.005,
        max_iter=1800,
        n_iter_no_change=100,
        random_state=0,
        priors=priors,
    )


def _priors(selected):
    return Priors(
        value=ValueBound(lower=0) if "ValueBound" in selected else None,
        features={0: Increasing()} if "Mono" in selected else {},
        curvature=Convex() if "Convex" in selected else None,
    )


def plot_example():
    rng = np.random.default_rng(5)
    X = np.linspace(-4, 4, 72).reshape(-1, 1)
    y = signal(X[:, 0]) + rng.normal(0, 0.28, X.shape[0])
    grid = np.linspace(-4.5, 4.5, 600).reshape(-1, 1)
    names = ("ValueBound", "Mono", "Convex")
    rows = tuple(
        tuple(
            Combination(" + ".join(selected), _priors(selected))
            for selected in combinations(names, size)
        )
        for size in range(1, len(names) + 1)
    )
    return plot_combinations(
        X=X,
        y=y,
        grid=grid,
        signal=signal,
        rows=rows,
        make_model=make_model,
        output_name="mlp_prior_combinations.png",
    )


if __name__ == "__main__":
    figure, output = plot_example()
    print(f"Saved figure to {output}")
    plt.show()
