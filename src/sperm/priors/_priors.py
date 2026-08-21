"""Typed, feature-independent shape priors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal, TypeAlias


def _validate_bound(value, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite or None.")
    return float(value)


@dataclass(frozen=True, slots=True)
class ValueBound:
    """Bound model values over the model's complete input domain."""

    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        lower = _validate_bound(self.lower, "lower")
        upper = _validate_bound(self.upper, "upper")
        if lower is None and upper is None:
            raise ValueError("At least one value bound must be provided.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower cannot be greater than upper.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class SlopeBound:
    """Bound a feature's partial derivative over the complete input domain."""

    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        lower = _validate_bound(self.lower, "lower")
        upper = _validate_bound(self.upper, "upper")
        if lower is None and upper is None:
            raise ValueError("At least one slope bound must be provided.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower cannot be greater than upper.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class Monotonicity:
    """Require monotonic increase or decrease in one feature."""

    direction: Literal["increasing", "decreasing"]

    def __post_init__(self) -> None:
        if self.direction not in {"increasing", "decreasing"}:
            raise ValueError("direction must be 'increasing' or 'decreasing'.")


@dataclass(frozen=True, slots=True)
class Unimodality:
    """Require a feature slice to have one minimum or maximum mode.

    Other features are held fixed. A ``"minimum"`` mode means non-increasing
    then non-decreasing; a ``"maximum"`` mode means non-decreasing then
    non-increasing. Plateaus are allowed.
    """

    mode: Literal["minimum", "maximum"]

    def __post_init__(self) -> None:
        if self.mode not in {"minimum", "maximum"}:
            raise ValueError("mode must be 'minimum' or 'maximum'.")


@dataclass(frozen=True, slots=True)
class Convex:
    """Require the model output to be jointly convex in all input features."""


@dataclass(frozen=True, slots=True)
class Concave:
    """Require the model output to be jointly concave in all input features."""


Feature: TypeAlias = int | str
FeaturePriors: TypeAlias = Mapping[
    Feature,
    Monotonicity
    | Unimodality
    | SlopeBound
    | Sequence[Monotonicity | Unimodality | SlopeBound],
]


@dataclass(frozen=True, slots=True)
class Priors:
    """Group output, feature-wise, and cross-feature shape priors."""

    value: ValueBound | Sequence[ValueBound] | None = None
    features: FeaturePriors = field(default_factory=dict)
    curvature: Convex | Concave | None = None


def Nonnegative() -> ValueBound:
    """Create the prior ``f(x) >= 0``."""
    return ValueBound(lower=0.0)


def Nonpositive() -> ValueBound:
    """Create the prior ``f(x) <= 0``."""
    return ValueBound(upper=0.0)


def Increasing() -> Monotonicity:
    """Create an increasing monotonicity prior."""
    return Monotonicity("increasing")


def Decreasing() -> Monotonicity:
    """Create a decreasing monotonicity prior."""
    return Monotonicity("decreasing")


def Lipschitz(constant: float) -> SlopeBound:
    """Create a coordinate-wise Lipschitz prior."""
    constant = _validate_bound(constant, "constant")
    if constant is None or constant <= 0:
        raise ValueError("constant must be positive and finite.")
    return SlopeBound(lower=-constant, upper=constant)


Prior: TypeAlias = (
    Concave
    | Convex
    | Monotonicity
    | Unimodality
    | ValueBound
    | SlopeBound
)

__all__ = [
    "Concave",
    "Convex",
    "Decreasing",
    "Feature",
    "Increasing",
    "Lipschitz",
    "Monotonicity",
    "Nonnegative",
    "Nonpositive",
    "Prior",
    "Priors",
    "SlopeBound",
    "Unimodality",
    "ValueBound",
]
