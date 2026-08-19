"""Compile compatible prior combinations for constrained neural networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._compiler import _normalize_container, _resolve_feature
from ._priors import (
    Concave,
    Convex,
    Monotonicity,
    Priors,
    Unimodality,
)
from ._tree_compiler import _combine_value_bounds


@dataclass(frozen=True)
class MLPArchitecture:
    """Normalized architecture requirements for an MLP."""

    kind: str
    monotonic_cst: np.ndarray
    value_bounds: tuple[float, float]


def compile_mlp_priors(priors, *, n_features, feature_names=None):
    """Validate prior interactions and select a constrained MLP architecture."""
    canonical = _normalize_container(priors)
    value = _combine_value_bounds(canonical.value)
    monotonic = np.zeros(n_features, dtype=np.int8)
    resolved = {}

    for feature, feature_priors in canonical.features.items():
        index = _resolve_feature(feature, n_features, feature_names)
        values = (
            tuple(feature_priors)
            if isinstance(feature_priors, (tuple, list))
            else (feature_priors,)
        )
        if not values or not all(
            isinstance(item, (Monotonicity, Unimodality))
            for item in values
        ):
            raise TypeError(
                "MLP feature priors must be Monotonicity or Unimodality; "
                "SlopeBound is not implemented."
            )
        monotonic_priors = tuple(
            item for item in values if isinstance(item, Monotonicity)
        )
        directions = {item.direction for item in monotonic_priors}
        if len(directions) > 1:
            raise ValueError(f"Conflicting monotonicity priors for feature {feature!r}.")
        if monotonic_priors:
            prior = monotonic_priors[0]
            monotonic[index] = 1 if prior.direction == "increasing" else -1
            resolved[index] = (prior,)
        else:
            resolved[index] = values

    lower = -np.inf if value is None or value.lower is None else value.lower
    upper = np.inf if value is None or value.upper is None else value.upper
    curvature = canonical.curvature
    if isinstance(curvature, Convex) and np.isfinite(upper):
        raise ValueError(
            "A globally convex function with a finite upper bound is constant; "
            "this degrading prior combination is not supported."
        )
    if isinstance(curvature, Concave) and np.isfinite(lower):
        raise ValueError(
            "A globally concave function with a finite lower bound is constant; "
            "this degrading prior combination is not supported."
        )

    resolved = _remove_curvature_redundancies(resolved, curvature)
    modes = {
        item.mode
        for values in resolved.values()
        for item in values
        if isinstance(item, Unimodality)
    }
    unimodal_features = {
        feature
        for feature, values in resolved.items()
        if any(isinstance(item, Unimodality) for item in values)
    }
    if modes and (n_features != 1 or unimodal_features != {0}):
        raise ValueError(
            "MLP unimodality currently requires exactly one input feature."
        )

    bounds = (lower, upper)
    if modes == {"minimum", "maximum"}:
        architectures = _monotonic_alternatives(curvature, bounds)
    elif modes:
        mode = next(iter(modes))
        opposite = (mode == "minimum" and isinstance(curvature, Concave)) or (
            mode == "maximum" and isinstance(curvature, Convex)
        )
        if opposite:
            architectures = _monotonic_alternatives(curvature, bounds)
        else:
            kind = f"unimodal_{mode}"
            architectures = (MLPArchitecture(kind, monotonic, bounds),)
    else:
        kind = (
            "convex"
            if isinstance(curvature, Convex)
            else "concave"
            if isinstance(curvature, Concave)
            else "dense"
        )
        architectures = (MLPArchitecture(kind, monotonic, bounds),)
    compiled = Priors(value=value, features=resolved, curvature=curvature)
    return compiled, architectures


def _remove_curvature_redundancies(resolved, curvature):
    result = {}
    for feature, values in resolved.items():
        retained = tuple(
            item
            for item in values
            if not (
                isinstance(item, Unimodality)
                and item.mode == "minimum"
                and isinstance(curvature, Convex)
            )
            and not (
                isinstance(item, Unimodality)
                and item.mode == "maximum"
                and isinstance(curvature, Concave)
            )
        )
        if retained:
            result[feature] = retained
    return result


def _monotonic_alternatives(curvature, bounds):
    kind = (
        "convex"
        if isinstance(curvature, Convex)
        else "concave"
        if isinstance(curvature, Concave)
        else "dense"
    )
    return tuple(
        MLPArchitecture(kind, np.asarray([direction], dtype=np.int8), bounds)
        for direction in (1, -1)
    )


__all__ = ["MLPArchitecture", "compile_mlp_priors"]
