import pytest

from sperm.priors import (
    Concave,
    Convex,
    Decreasing,
    Increasing,
    Lipschitz,
    Monotonicity,
    Nonnegative,
    Nonpositive,
    Priors,
    SlopeBound,
    Unimodality,
    ValueBound,
    parse_prior,
    parse_priors,
)
from sperm.priors._compiler import compile_linear_priors


def test_prior_objects_do_not_contain_feature_information():
    assert Increasing() == Monotonicity("increasing")
    assert Decreasing() == Monotonicity("decreasing")
    assert Lipschitz(1.5) == SlopeBound(lower=-1.5, upper=1.5)
    assert Nonnegative() == ValueBound(lower=0)
    assert Nonpositive() == ValueBound(upper=0)


def test_unimodality_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        Unimodality("center")


def test_rejects_nonpositive_lipschitz_constant():
    with pytest.raises(ValueError, match="positive and finite"):
        Lipschitz(0)


def test_compiles_multiple_priors_for_each_feature():
    priors = {
        0: (Increasing(), Lipschitz(2)),
        1: Decreasing(),
    }

    resolved, (lower, upper) = compile_linear_priors(priors, n_features=2)

    assert resolved == Priors(
        features={
            0: (Increasing(), Lipschitz(2)),
            1: (Decreasing(),),
        }
    )
    assert lower.tolist() == [0.0, float("-inf")]
    assert upper.tolist() == [2.0, 0.0]


def test_compiles_named_features():
    resolved, bounds = compile_linear_priors(
        {"temperature": Increasing()},
        n_features=2,
        feature_names=["pressure", "temperature"],
    )

    assert resolved == Priors(features={1: (Increasing(),)})
    assert bounds[0].tolist() == [float("-inf"), 0.0]


def test_rejects_feature_index_outside_input():
    with pytest.raises(ValueError, match="only 2 features"):
        compile_linear_priors({2: Increasing()}, n_features=2)


def test_parse_prior_is_an_explicit_configuration_boundary():
    assert parse_prior("nonnegative") == (None, ValueBound(lower=0))
    assert parse_prior("temperature:Lipschitz:1.5") == (
        "temperature",
        Lipschitz(1.5),
    )
    assert parse_prior("0:unimodal:minimum") == (0, Unimodality("minimum"))
    assert parse_prior("0:unimodal:maximum") == (0, Unimodality("maximum"))
    assert parse_prior("convex") == (None, Convex())
    assert parse_prior("concave") == (None, Concave())
    assert parse_priors(["convex"]) == Priors(curvature=Convex())
    assert parse_priors(
        ["nonnegative", "0:increasing", "0:Lipschitz:2"]
    ) == Priors(
        value=(ValueBound(lower=0),),
        features={
            0: (Monotonicity("increasing"), SlopeBound(lower=-2, upper=2))
        },
    )


def test_linear_compiler_rejects_value_bound():
    with pytest.raises(
        ValueError,
        match="ValueBound is not supported by the linear base model",
    ):
        compile_linear_priors(
            Priors(value=ValueBound(lower=1, upper=2)),
            n_features=2,
        )
