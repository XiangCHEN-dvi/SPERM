"""Compare scalar-to-scalar regressors with and without shape priors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sperm.gaussian_process import GaussianProcessRegressor
from sperm.linear_model import LinearRegression, Ridge
from sperm.neural_network import MLPRegressor
from sperm.priors import Convex, Increasing, Priors, SlopeBound, Unimodality, ValueBound
from sperm.tree_model import (
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)


@dataclass(frozen=True)
class Experiment:
    name: str
    priors: Priors
    signal: object
    supported_by: frozenset[str]


MODEL_NAMES = (
    "Linear",
    "Ridge",
    "Decision tree",
    "Random forest",
    "Gradient boosting",
    "MLP",
    "GPR",
)

MODEL_GROUPS = (
    ("Linear models", ("Linear", "Ridge")),
    ("Tree models", ("Decision tree", "Random forest", "Gradient boosting")),
    ("MLP", ("MLP",)),
    ("GPR", ("GPR",)),
)

MODEL_COLORS = {
    "Linear": "#4477AA",
    "Ridge": "#EE6677",
    "Decision tree": "#4477AA",
    "Random forest": "#EE6677",
    "Gradient boosting": "#228833",
    "MLP": "#4477AA",
    "GPR": "#4477AA",
}

UNSUPPORTED_REASONS = {
    ("ValueBound [-1, 1]", "Linear models"): "degrading",
    ("SlopeBound [-0.5, 0.5]", "Tree models"): "impossible",
    ("SlopeBound [-0.5, 0.5]", "MLP"): "not yet supported",
    ("Unimodality", "Linear models"): "trivial",
    ("Convex", "Linear models"): "trivial",
    ("Convex", "Tree models"): "impossible",
}

EXPERIMENTS = (
    Experiment(
        "ValueBound [-1, 1]",
        Priors(value=ValueBound(lower=-1, upper=1)),
        lambda x: 1.5 * np.sin(1.4 * x),
        frozenset(
            {"Decision tree", "Random forest", "Gradient boosting", "MLP", "GPR"}
        ),
    ),
    Experiment(
        "Increasing",
        Priors(features={0: Increasing()}),
        lambda x: x * (x - 1) * (x + 1),
        frozenset(MODEL_NAMES),
    ),
    Experiment(
        "SlopeBound [-0.5, 0.5]",
        Priors(features={0: SlopeBound(lower=-0.5, upper=0.5)}),
        lambda x: 0.35 * x + 0.12 * np.sin(2 * x),
        frozenset({"Linear", "Ridge", "GPR"}),
    ),
    Experiment(
        "Unimodality",
        Priors(features={0: Unimodality("minimum")}),
        lambda x: (
            -5 * np.exp(-0.5 * (x - 0.4) ** 2)
            + np.minimum(x + 2.5, 0)
            + np.minimum(2.5 - x, 0)
        ),
        frozenset(
            {"Decision tree", "Random forest", "Gradient boosting", "MLP", "GPR"}
        ),
    ),
    Experiment(
        "Convex",
        Priors(curvature=Convex()),
        lambda x: (np.abs(x) - 1.0) ** 2,
        frozenset({"MLP", "GPR"}),
    ),
)


def make_model(name, priors):
    """Create comparably small estimators suitable for a visual example."""
    if name == "Linear":
        return LinearRegression(priors=priors)
    if name == "Ridge":
        return Ridge(alpha=1.0, priors=priors)
    if name == "Decision tree":
        return DecisionTreeRegressor(max_depth=5, min_samples_leaf=4, priors=priors)
    if name == "Random forest":
        return RandomForestRegressor(
            n_estimators=40,
            max_depth=5,
            min_samples_leaf=3,
            random_state=0,
            priors=priors,
        )
    if name == "Gradient boosting":
        return GradientBoostingRegressor(
            max_iter=120,
            max_leaf_nodes=8,
            min_samples_leaf=3,
            learning_rate=0.08,
            early_stopping=False,
            random_state=0,
            priors=priors,
        )
    if name == "MLP":
        has_unimodality_prior = any(
            isinstance(prior, Unimodality)
            for prior in (() if priors is None else priors.features.values())
        )
        return MLPRegressor(
            hidden_layer_sizes=(12,),
            learning_rate_init=0.005,
            max_iter=3000 if has_unimodality_prior else 1500,
            n_iter_no_change=80,
            random_state=0,
            priors=priors,
        )
    if name == "GPR":
        return GaussianProcessRegressor(
            n_basis=16,
            alpha=0.04,
            smoothness=0.1,
            priors=priors,
        )
    raise ValueError(f"Unknown model: {name}")


def make_dataset(experiment, rng):
    """Use the same observations for constrained and unconstrained fits."""
    if experiment.name == "Increasing":
        x_min, x_max = -2.0, 2.0
    elif experiment.name == "Unimodality":
        x_min, x_max = -4.0, 3.5
    else:
        x_min, x_max = -3.0, 3.0
    x = np.linspace(x_min, x_max, 42)
    clean = experiment.signal(x)
    noise_scale = 0.48 if experiment.name == "Increasing" else 0.22
    noise = rng.normal(scale=noise_scale, size=x.shape)
    if experiment.name != "Increasing":
        noise[[5, 15, 27, 36]] += np.array([0.55, -0.65, 0.6, -0.5])
    return x.reshape(-1, 1), clean + noise


def plot_experiments():
    """Build and save the complete prior-by-regressor comparison grid."""
    rng = np.random.default_rng(7)
    figure, axes = plt.subplots(
        len(EXPERIMENTS),
        len(MODEL_GROUPS),
        figsize=(22, 18),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )

    for row, experiment in enumerate(EXPERIMENTS):
        X, y = make_dataset(experiment, rng)
        x_margin = 0.1 * np.ptp(X[:, 0])
        y_margin = 0.1 * np.ptp(y)
        x_limits = (X[:, 0].min() - x_margin, X[:, 0].max() + x_margin)
        y_limits = (y.min() - y_margin, y.max() + y_margin)
        grid = np.linspace(*x_limits, 500).reshape(-1, 1)
        for column, (group_name, model_names) in enumerate(MODEL_GROUPS):
            axis = axes[row, column]
            supported_models = tuple(
                name for name in model_names if name in experiment.supported_by
            )
            if row == 0:
                axis.set_title(group_name, fontsize=15)
            if column == 0:
                axis.set_ylabel(experiment.name, fontsize=14)
            if not supported_models:
                axis.set_xticks([])
                axis.set_yticks([])
                axis.grid(False)
                axis.patch.set_visible(False)
                for spine in axis.spines.values():
                    spine.set_visible(False)
                axis.plot(
                    [0.12, 0.88],
                    [0.12, 0.88],
                    transform=axis.transAxes,
                    color="#BBBBBB",
                    linewidth=2.5,
                    solid_capstyle="round",
                )
                axis.text(
                    0.5,
                    0.5,
                    UNSUPPORTED_REASONS[(experiment.name, group_name)],
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="#666666",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.9,
                    },
                )
                continue

            axis.scatter(X[:, 0], y, s=16, color="#333333", alpha=0.55)
            for model_name in supported_models:
                color = MODEL_COLORS[model_name]
                unconstrained = make_model(model_name, None).fit(X, y)
                constrained = make_model(model_name, experiment.priors).fit(X, y)
                if model_name == "GPR":
                    unconstrained_mean, unconstrained_std = unconstrained.predict(
                        grid, return_std=True
                    )
                    constrained_mean, constrained_std = constrained.predict(
                        grid, return_std=True
                    )
                    axis.fill_between(
                        grid[:, 0],
                        unconstrained_mean - 2 * unconstrained_std,
                        unconstrained_mean + 2 * unconstrained_std,
                        color=color,
                        alpha=0.08,
                        linewidth=0,
                    )
                    axis.fill_between(
                        grid[:, 0],
                        constrained_mean - 2 * constrained_std,
                        constrained_mean + 2 * constrained_std,
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                    )
                else:
                    unconstrained_mean = unconstrained.predict(grid)
                    constrained_mean = constrained.predict(grid)
                axis.plot(
                    grid[:, 0],
                    unconstrained_mean,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    label=f"{model_name} · w/o prior",
                )
                axis.plot(
                    grid[:, 0],
                    constrained_mean,
                    color=color,
                    linewidth=2,
                    label=f"{model_name} · w/ prior",
                )
            axis.grid(alpha=0.15)
            axis.tick_params(axis="both", labelbottom=False, labelleft=False)
            axis.legend(
                loc="best",
                fontsize=8.5,
                ncol=2 if len(model_names) > 1 else 1,
                framealpha=0.75,
            )
        for axis in axes[row]:
            axis.set_xlim(x_limits)
            axis.set_ylim(y_limits)
    output = Path(__file__).resolve().parents[1] / "assets" / "scalar2scalar.png"
    figure.savefig(output, dpi=160)
    return figure, output


if __name__ == "__main__":
    figure, output = plot_experiments()
    print(f"Saved figure to {output}")
    plt.show()
