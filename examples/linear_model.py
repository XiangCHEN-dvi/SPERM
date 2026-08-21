"""Show LinearRegression with one and two simultaneous shape priors."""

import matplotlib.pyplot as plt
import numpy as np
from _combination_plot import Combination, plot_combinations

from sperm.linear_model import LinearRegression
from sperm.priors import Increasing, Priors, SlopeBound


def signal(x):
    """Underlying increasing function with slope below the requested cap."""
    return 0.7 * x + 0.15


def make_model(priors):
    return LinearRegression(priors=priors)


def plot_example():
    rng = np.random.default_rng(2)
    X = np.linspace(-3, 3, 36).reshape(-1, 1)
    y = signal(X[:, 0]) + rng.normal(0, 0.7, X.shape[0])
    y[[2, 31]] += np.array([1.8, -1.5])
    grid = np.linspace(-3.4, 3.4, 500).reshape(-1, 1)
    increasing = Increasing()
    slope = SlopeBound(upper=0.8)
    rows = (
        (
            Combination("Monotonicity", Priors(features={0: increasing})),
            Combination("SlopeBound", Priors(features={0: slope})),
        ),
        (
            Combination(
                "Monotonicity + SlopeBound",
                Priors(features={0: (increasing, slope)}),
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
        output_name="linear_model_prior_combinations.png",
    )


if __name__ == "__main__":
    figure, output = plot_example()
    print(f"Saved figure to {output}")
    plt.show()
