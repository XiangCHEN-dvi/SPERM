"""Compile priors for tree-based regressors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._compiler import _normalize_container, _resolve_feature
from ._priors import Monotonicity, Priors, Unimodality, ValueBound


def compile_tree_priors(
    priors,
    *,
    n_features: int,
    feature_names=None,
) -> tuple[
    Priors,
    np.ndarray | None,
    tuple[float, float],
    tuple[int, str] | None,
]:
    """Compile value bounds and feature monotonicity for sklearn trees."""
    canonical = _normalize_container(priors)
    if canonical.curvature is not None:
        raise ValueError(
            "Convex and Concave priors are not supported by piecewise-constant "
            "tree models."
        )
    value_prior = _combine_value_bounds(canonical.value)
    monotonic = np.zeros(n_features, dtype=np.int8)
    resolved: dict[
        int,
        tuple[Monotonicity | Unimodality, ...],
    ] = {}
    unimodality_constraint = None

    for feature, value in canonical.features.items():
        index = _resolve_feature(feature, n_features, feature_names)
        feature_priors = _normalize_tree_feature_priors(value)
        unimodality_priors = tuple(
            prior
            for prior in feature_priors
            if isinstance(prior, Unimodality)
        )
        monotonic_priors = tuple(
            prior for prior in feature_priors if isinstance(prior, Monotonicity)
        )
        directions = {prior.direction for prior in monotonic_priors}
        if len(directions) > 1:
            raise ValueError(f"Conflicting monotonicity priors for feature {feature!r}.")

        # A monotone slice satisfies both unimodality modes, so an explicit
        # unimodality prior is redundant when monotonicity is set.
        if monotonic_priors:
            direction = monotonic_priors[0].direction
            monotonic[index] = 1 if direction == "increasing" else -1
            resolved[index] = (monotonic_priors[0],)
            continue

        if unimodality_priors:
            modes = {prior.mode for prior in unimodality_priors}
            if len(modes) > 1:
                if unimodality_constraint is not None:
                    raise ValueError(
                        "A tree model currently accepts one unimodal feature."
                    )
                unimodality_constraint = (index, "monotonic")
                resolved[index] = (
                    Unimodality("minimum"),
                    Unimodality("maximum"),
                )
                continue
            if unimodality_constraint is not None:
                raise ValueError(
                    "A tree model currently accepts unimodality for exactly "
                    "one feature."
                )
            mode = unimodality_priors[0].mode
            unimodality_constraint = (index, mode)
            resolved[index] = (unimodality_priors[0],)
            continue

    if value_prior is None:
        value_bounds = (-np.inf, np.inf)
    else:
        value_bounds = (
            -np.inf if value_prior.lower is None else value_prior.lower,
            np.inf if value_prior.upper is None else value_prior.upper,
        )

    compiled = Priors(value=value_prior, features=resolved)
    monotonic_cst = monotonic if np.any(monotonic) else None
    return compiled, monotonic_cst, value_bounds, unimodality_constraint


def _combine_value_bounds(value) -> ValueBound | None:
    if value is None:
        return None
    values = (value,) if isinstance(value, ValueBound) else tuple(value)
    if not values or not all(isinstance(item, ValueBound) for item in values):
        raise TypeError("Priors.value must contain one or more ValueBound objects.")
    lower = max(
        (item.lower for item in values if item.lower is not None),
        default=None,
    )
    upper = min(
        (item.upper for item in values if item.upper is not None),
        default=None,
    )
    return ValueBound(lower=lower, upper=upper)


def _normalize_tree_feature_priors(
    value,
) -> tuple[Monotonicity | Unimodality, ...]:
    prior_types = (Monotonicity, Unimodality)
    if isinstance(value, prior_types):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        normalized = tuple(value)
        if normalized and all(isinstance(item, prior_types) for item in normalized):
            return normalized
    raise TypeError(
        "Tree feature priors must contain Monotonicity or Unimodality objects; "
        "SlopeBound is not supported by tree-based models."
    )


__all__ = ["compile_tree_priors"]
