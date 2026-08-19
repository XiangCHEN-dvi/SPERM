"""Explicit parsing helpers for text-based prior configuration."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ._priors import (
    Concave,
    Convex,
    Decreasing,
    Increasing,
    Lipschitz,
    Nonnegative,
    Nonpositive,
    Prior,
    Priors,
    Unimodality,
    ValueBound,
)


def parse_prior(specification: str) -> tuple[int | str | None, Prior]:
    """Parse one legacy semantic specification into a bound prior."""
    if not isinstance(specification, str):
        raise TypeError("A prior specification must be a string.")

    if specification == "nonnegative":
        return None, Nonnegative()
    if specification == "nonpositive":
        return None, Nonpositive()
    if specification == "convex":
        return None, Convex()
    if specification == "concave":
        return None, Concave()

    parts = specification.split(":")
    if len(parts) not in {2, 3} or not parts[0]:
        raise ValueError(f"Invalid prior specification: {specification!r}")

    feature: int | str
    try:
        feature = int(parts[0])
    except ValueError:
        feature = parts[0]

    kind = parts[1]
    if kind == "increasing" and len(parts) == 2:
        return feature, Increasing()
    if kind == "decreasing" and len(parts) == 2:
        return feature, Decreasing()
    if kind == "unimodal" and len(parts) == 3:
        return feature, Unimodality(parts[2])
    if kind == "Lipschitz" and len(parts) == 3:
        try:
            constant = float(parts[2])
        except ValueError as error:
            raise ValueError(
                f"Invalid Lipschitz constant: {specification!r}"
            ) from error
        return feature, Lipschitz(constant)

    raise ValueError(f"Invalid prior specification: {specification!r}")


def parse_priors(specifications: Iterable[str]) -> Priors:
    """Parse text specifications into the canonical priors container."""
    values: list[ValueBound] = []
    curvature = None
    features = defaultdict(list)

    for specification in specifications:
        feature, prior = parse_prior(specification)
        if feature is None:
            if isinstance(prior, ValueBound):
                values.append(prior)
            elif isinstance(prior, (Convex, Concave)):
                if curvature is not None and type(curvature) is not type(prior):
                    raise ValueError("Conflicting convex and concave priors.")
                curvature = prior
            else:  # pragma: no cover
                raise TypeError("Unsupported global prior.")
        else:
            features[feature].append(prior)

    return Priors(
        value=tuple(values) or None,
        features={feature: tuple(priors) for feature, priors in features.items()},
        curvature=curvature,
    )


__all__ = ["parse_prior", "parse_priors"]
