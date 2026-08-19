"""Compile canonical priors for linear model solvers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

from ._priors import Monotonicity, Priors, SlopeBound


def compile_linear_priors(
    priors,
    *,
    n_features: int,
    feature_names=None,
) -> tuple[
    Priors,
    tuple[np.ndarray, np.ndarray],
]:
    """Resolve feature keys and compile full-domain linear priors."""
    canonical = _normalize_container(priors)
    if canonical.curvature is not None:
        raise ValueError(
            "Convex and Concave priors are not supported by the linear base model."
        )
    if canonical.value is not None:
        raise ValueError(
            "ValueBound is not supported by the linear base model over the "
            "complete input space."
        )
    resolved: dict[int, tuple[Monotonicity | SlopeBound, ...]] = {}
    lower = np.full(n_features, -np.inf, dtype=float)
    upper = np.full(n_features, np.inf, dtype=float)

    for feature, value in canonical.features.items():
        index = _resolve_feature(feature, n_features, feature_names)
        feature_priors = _normalize_feature_priors(value)
        resolved[index] = resolved.get(index, ()) + feature_priors

        for prior in feature_priors:
            if isinstance(prior, Monotonicity):
                if prior.direction == "increasing":
                    lower[index] = max(lower[index], 0.0)
                else:
                    upper[index] = min(upper[index], 0.0)
            else:
                if prior.lower is not None:
                    lower[index] = max(lower[index], prior.lower)
                if prior.upper is not None:
                    upper[index] = min(upper[index], prior.upper)

    if np.any(lower > upper):
        raise ValueError("The requested priors are infeasible for a linear model.")

    compiled = Priors(features=resolved)
    return compiled, (lower, upper)


def _normalize_container(priors) -> Priors:
    if priors is None:
        return Priors()
    if isinstance(priors, Priors):
        return priors
    if isinstance(priors, Mapping):
        return Priors(features=priors)
    raise TypeError("priors must be a Priors object or a feature mapping.")


def _resolve_feature(feature, n_features: int, feature_names) -> int:
    if isinstance(feature, Integral) and not isinstance(feature, (bool, np.bool_)):
        index = int(feature)
        if index < 0 or index >= n_features:
            raise ValueError(
                f"Prior references feature {index}, but the input has "
                f"only {n_features} features."
            )
        return index

    if isinstance(feature, str):
        if feature_names is None:
            raise ValueError(
                "String feature names require input data with named columns."
            )
        matches = np.flatnonzero(np.asarray(feature_names, dtype=object) == feature)
        if len(matches) == 0:
            raise ValueError(f"Unknown feature name: {feature!r}")
        if len(matches) > 1:
            raise ValueError(f"Feature name is not unique: {feature!r}")
        return int(matches[0])

    raise TypeError("Prior feature keys must be non-negative integers or strings.")


def _normalize_feature_priors(
    value,
) -> tuple[Monotonicity | SlopeBound, ...]:
    prior_types = (Monotonicity, SlopeBound)
    if isinstance(value, prior_types):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        normalized = tuple(value)
        if normalized and all(isinstance(item, prior_types) for item in normalized):
            return normalized
    raise TypeError(
        "Each feature must map to Monotonicity/SlopeBound priors or a non-empty "
        "sequence of them."
    )


__all__ = ["compile_linear_priors"]
