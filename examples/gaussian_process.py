"""Show a GPR with every subset of four compatible shape priors."""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from _combination_plot import Combination, plot_combinations

from sperm.gaussian_process import GaussianProcessRegressor
from sperm.priors import Convex, Increasing, Priors, SlopeBound, ValueBound


def signal(x):
    """Softplus has values above zero and derivatives between zero and one."""
    return np.logaddexp(0, x)


def make_model(priors):
    return GaussianProcessRegressor(
        n_basis=16,
        alpha=0.04,
        smoothness=0.1,
        priors=priors,
    )


def _priors(selected):
    feature_priors = []
    if "Mono" in selected:
        feature_priors.append(Increasing())
    if "Slope" in selected:
        feature_priors.append(SlopeBound(upper=1))
    return Priors(
        value=ValueBound(lower=0) if "ValueBound" in selected else None,
        features={0: tuple(feature_priors)} if feature_priors else {},
        curvature=Convex() if "Convex" in selected else None,
    )


def plot_example():
    rng = np.random.default_rng(6)
    X = np.linspace(-4, 4, 64).reshape(-1, 1)
    y = signal(X[:, 0]) + rng.normal(0, 0.22, X.shape[0])
    grid = np.linspace(-4.5, 4.5, 600).reshape(-1, 1)
    names = ("ValueBound", "Mono", "Slope", "Convex")
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
        output_name="gpr_prior_combinations.png",
        uncertainty=True,
    )


if __name__ == "__main__":
    figure, output = plot_example()
    print(f"Saved figure to {output}")
    plt.show()
