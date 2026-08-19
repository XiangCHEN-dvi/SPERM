"""Compile full-domain priors into linear spline-coefficient constraints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

from ..priors import (
    Concave,
    Convex,
    Monotonicity,
    Priors,
    SlopeBound,
    Unimodality,
    ValueBound,
)


def normalize_gpr_priors(priors, n_features, feature_names):
    """Normalize and validate priors supported by the additive spline GP."""
    if priors is None:
        canonical = Priors()
    elif isinstance(priors, Priors):
        canonical = priors
    elif isinstance(priors, Mapping):
        canonical = Priors(features=priors)
    else:
        raise TypeError("priors must be a Priors object or a feature mapping.")

    values = _sequence(canonical.value)
    if not all(isinstance(prior, ValueBound) for prior in values):
        raise TypeError("GPR value priors must be ValueBound objects.")
    lower = max((p.lower for p in values if p.lower is not None), default=None)
    upper = min((p.upper for p in values if p.upper is not None), default=None)
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("The requested ValueBound priors are infeasible.")
    value = None if not values else ValueBound(lower=lower, upper=upper)
    if value is not None and n_features != 1:
        raise ValueError("Global ValueBound currently requires exactly one feature.")

    resolved = {}
    for key, item in canonical.features.items():
        index = _resolve(key, n_features, feature_names)
        items = _sequence(item)
        if not items:
            raise TypeError("Feature prior sequences cannot be empty.")
        if not all(
            isinstance(p, (Monotonicity, Unimodality, SlopeBound))
            for p in items
        ):
            raise TypeError(
                "GPR feature priors must be Monotonicity, SlopeBound, or "
                "Unimodality."
            )
        resolved[index] = resolved.get(index, ()) + tuple(items)

    curvature = canonical.curvature
    if curvature is not None and not isinstance(curvature, (Convex, Concave)):
        raise TypeError("curvature must be Convex, Concave, or None.")
    if isinstance(curvature, Convex) and value is not None and value.upper is not None:
        raise ValueError("A globally convex function with a finite upper bound is degrading.")
    if isinstance(curvature, Concave) and value is not None and value.lower is not None:
        raise ValueError("A globally concave function with a finite lower bound is degrading.")

    resolved = {
        feature: _remove_redundant_unimodality(items, curvature)
        for feature, items in resolved.items()
    }
    unimodal_features = {
        feature
        for feature, items in resolved.items()
        if any(isinstance(p, Unimodality) for p in items)
    }
    if unimodal_features and (n_features != 1 or unimodal_features != {0}):
        raise ValueError(
            "GPR unimodality currently requires exactly one input feature."
        )

    normalized = Priors(value=value, features=resolved, curvature=curvature)
    _validate_global_combinations(normalized)
    return normalized


def compile_gpr_constraints(basis, priors):
    """Return ``A, b`` such that ``A @ coefficient <= b`` enforces priors."""
    rows = []
    bounds = []

    if priors.value is not None:
        controls = basis.value_control_map()
        if priors.value.lower is not None:
            _append(rows, bounds, -controls, -priors.value.lower)
        if priors.value.upper is not None:
            _append(rows, bounds, controls, priors.value.upper)

        left_slope, right_slope = basis.endpoint_derivative_map(0)
        if priors.value.lower is not None:
            rows.extend((left_slope, -right_slope))
            bounds.extend((0.0, 0.0))
        if priors.value.upper is not None:
            rows.extend((-left_slope, right_slope))
            bounds.extend((0.0, 0.0))

    for feature in range(basis.n_features):
        derivative = basis.derivative_control_map(feature, 1)
        lower, upper = _slope_limits(priors.features.get(feature, ()))
        if lower is not None:
            _append(rows, bounds, -derivative, -lower)
        if upper is not None:
            _append(rows, bounds, derivative, upper)

        if isinstance(priors.curvature, Convex):
            _append(
                rows,
                bounds,
                -basis.derivative_control_map(feature, 2),
                0.0,
            )
        elif isinstance(priors.curvature, Concave):
            _append(
                rows,
                bounds,
                basis.derivative_control_map(feature, 2),
                0.0,
            )

    if not rows:
        return np.empty((0, basis.n_coefficients)), np.empty(0)
    return np.asarray(rows, dtype=float), np.asarray(bounds, dtype=float)


def compile_gpr_constraint_candidates(basis, priors):
    """Compile the polyhedral union induced by one-dimensional unimodality."""
    common_matrix, common_bound = compile_gpr_constraints(basis, priors)
    modes = _unimodality_modes(priors.features.get(0, ()))
    if not modes:
        return ((common_matrix, common_bound, None),)

    derivative = basis.derivative_control_map(0, 1)
    if modes == {"minimum", "maximum"}:
        patterns = (
            ("increasing", -derivative),
            ("decreasing", derivative),
        )
    else:
        mode = next(iter(modes))
        patterns = []
        for turn in range(derivative.shape[0] + 1):
            if mode == "minimum":
                signs = np.concatenate((np.ones(turn), -np.ones(len(derivative) - turn)))
            else:
                signs = np.concatenate((-np.ones(turn), np.ones(len(derivative) - turn)))
            patterns.append(((mode, turn), signs[:, None] * derivative))

    candidates = []
    for label, matrix in patterns:
        candidates.append(
            (
                np.vstack((common_matrix, matrix)),
                np.concatenate((common_bound, np.zeros(matrix.shape[0]))),
                label,
            )
        )
    return tuple(candidates)


def _append(rows, bounds, matrix, bound):
    matrix = np.atleast_2d(matrix)
    rows.extend(matrix)
    bounds.extend(np.broadcast_to(bound, matrix.shape[0]))


def _slope_limits(feature_priors):
    lower, upper = None, None
    for prior in feature_priors:
        if isinstance(prior, Monotonicity):
            if prior.direction == "increasing":
                lower = _max_optional(lower, 0.0)
            else:
                upper = _min_optional(upper, 0.0)
        elif isinstance(prior, SlopeBound):
            lower = _max_optional(lower, prior.lower)
            upper = _min_optional(upper, prior.upper)
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("The requested slope priors are infeasible.")
    return lower, upper


def _validate_global_combinations(priors):
    value = priors.value
    for feature_priors in priors.features.values():
        lower, upper = _slope_limits(feature_priors)
        if value is not None:
            if lower is not None and lower > 0:
                raise ValueError(
                    "A globally value-bounded function cannot have a positive "
                    "lower slope bound."
                )
            if upper is not None and upper < 0:
                raise ValueError(
                    "A globally value-bounded function cannot have a negative "
                    "upper slope bound."
                )


def _remove_redundant_unimodality(items, curvature):
    if any(isinstance(prior, Monotonicity) for prior in items):
        return tuple(
            prior
            for prior in items
            if not isinstance(prior, Unimodality)
        )
    return tuple(
        prior
        for prior in items
        if not (
            isinstance(prior, Unimodality)
            and prior.mode == "minimum"
            and isinstance(curvature, Convex)
        )
        and not (
            isinstance(prior, Unimodality)
            and prior.mode == "maximum"
            and isinstance(curvature, Concave)
        )
    )


def _unimodality_modes(feature_priors):
    modes = set()
    for prior in feature_priors:
        if isinstance(prior, Unimodality):
            modes.add(prior.mode)
    return modes


def _sequence(value):
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return (value,)


def _resolve(key, n_features, feature_names):
    if isinstance(key, Integral) and not isinstance(key, (bool, np.bool_)):
        index = int(key)
        if 0 <= index < n_features:
            return index
        raise ValueError(f"Prior references feature {index}, but X has {n_features} features.")
    if isinstance(key, str):
        if feature_names is None:
            raise ValueError("String feature names require named input columns.")
        matches = np.flatnonzero(np.asarray(feature_names, dtype=object) == key)
        if len(matches) == 1:
            return int(matches[0])
        raise ValueError(f"Unknown or non-unique feature name: {key!r}")
    raise TypeError("Prior feature keys must be non-negative integers or strings.")


def _max_optional(left, right):
    return right if left is None else max(left, right) if right is not None else left


def _min_optional(left, right):
    return right if left is None else min(left, right) if right is not None else left


__all__ = [
    "compile_gpr_constraint_candidates",
    "compile_gpr_constraints",
    "normalize_gpr_priors",
]
