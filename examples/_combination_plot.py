"""Shared plotting utilities for the prior-combination examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Combination:
    """A compact label and the corresponding estimator prior configuration."""

    label: str
    priors: object


def plot_combinations(
    *,
    X,
    y,
    grid,
    signal,
    rows,
    make_model,
    output_name,
    uncertainty=False,
):
    """Plot one row per prior count and save the resulting figure."""
    max_columns = max(len(row) for row in rows)
    figure = plt.figure(
        figsize=(3.7 * max_columns, 3.2 * len(rows)),
        constrained_layout=True,
    )
    layout = figure.add_gridspec(len(rows), 2 * max_columns)
    unconstrained = make_model(None).fit(X, y)
    baseline = unconstrained.predict(grid, return_std=True) if uncertainty else None
    colors = ("#4477AA", "#228833", "#CC6677", "#AA3377")

    for row_index, combinations in enumerate(rows):
        offset = max_columns - len(combinations)
        for column_index, combination in enumerate(combinations):
            start = offset + 2 * column_index
            axis = figure.add_subplot(layout[row_index, start : start + 2])
            constrained = make_model(combination.priors).fit(X, y)
            color = colors[min(row_index, len(colors) - 1)]

            axis.scatter(X[:, 0], y, s=18, color="#333333", alpha=0.5)
            axis.plot(
                grid[:, 0],
                signal(grid[:, 0]),
                color="#222222",
                linewidth=1.2,
                alpha=0.55,
                label="truth",
            )
            if uncertainty:
                baseline_mean, baseline_std = baseline
                constrained_mean, constrained_std = constrained.predict(
                    grid, return_std=True
                )
                axis.fill_between(
                    grid[:, 0],
                    baseline_mean - 2 * baseline_std,
                    baseline_mean + 2 * baseline_std,
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                )
                axis.fill_between(
                    grid[:, 0],
                    constrained_mean - 2 * constrained_std,
                    constrained_mean + 2 * constrained_std,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
            else:
                baseline_mean = unconstrained.predict(grid)
                constrained_mean = constrained.predict(grid)

            axis.plot(
                grid[:, 0],
                baseline_mean,
                color=color,
                linestyle="--",
                linewidth=1.6,
                label="w/o",
            )
            axis.plot(
                grid[:, 0],
                constrained_mean,
                color=color,
                linewidth=2.2,
                label="w/",
            )
            axis.set_title(combination.label, fontsize=13)
            if column_index == 0:
                count = row_index + 1
                axis.set_ylabel(f"{count} prior" + ("s" if count > 1 else ""))
            axis.tick_params(labelbottom=False, labelleft=False)
            axis.grid(alpha=0.15)
            axis.legend(fontsize=8.5, framealpha=0.8)

    output = Path(__file__).resolve().parents[1] / "assets" / output_name
    figure.savefig(output, dpi=160)
    return figure, output


__all__ = ["Combination", "plot_combinations"]
