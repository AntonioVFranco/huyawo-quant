"""Tests for normalized quantization recipe contracts."""

import pytest
from huyawo_quant.contracts import QuantRecipe
from pydantic import ValidationError


def test_quant_recipe_serializes_weight_only_recipe() -> None:
    recipe = QuantRecipe(
        algorithm="gptq",
        scheme="w4a16",
        weight_granularity="group",
        weight_group_size=128,
        weight_symmetric=True,
        activation_mode="none",
        ignored_modules=("lm_head",),
    )

    assert recipe.model_dump(mode="json") == {
        "algorithm": "gptq",
        "scheme": "w4a16",
        "weight_granularity": "group",
        "weight_group_size": 128,
        "weight_symmetric": True,
        "activation_mode": "none",
        "ignored_modules": ["lm_head"],
    }


def test_quant_recipe_serializes_weight_activation_recipe() -> None:
    recipe = QuantRecipe(
        algorithm="rtn",
        scheme="w8a8_int8",
        weight_granularity="channel",
        weight_symmetric=True,
        activation_mode="dynamic",
    )

    assert recipe.model_dump(mode="json") == {
        "algorithm": "rtn",
        "scheme": "w8a8_int8",
        "weight_granularity": "channel",
        "weight_group_size": None,
        "weight_symmetric": True,
        "activation_mode": "dynamic",
        "ignored_modules": [],
    }


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("algorithm", "autoround"),
        ("scheme", "w2a16"),
        ("weight_granularity", "block"),
        ("activation_mode", "adaptive"),
    ],
)
def test_quant_recipe_rejects_unknown_enum_values(
    field_name: str,
    value: str,
) -> None:
    payload: dict[str, object] = {
        "algorithm": "gptq",
        "scheme": "w4a16",
        "weight_granularity": "group",
        "weight_group_size": 128,
        "weight_symmetric": True,
        "activation_mode": "none",
        "ignored_modules": ("lm_head",),
    }
    payload[field_name] = value

    with pytest.raises(ValidationError):
        QuantRecipe.model_validate(payload)


@pytest.mark.parametrize("weight_group_size", [0, -1])
def test_quant_recipe_rejects_non_positive_group_size(
    weight_group_size: int,
) -> None:
    with pytest.raises(ValidationError):
        QuantRecipe(
            algorithm="gptq",
            scheme="w4a16",
            weight_granularity="group",
            weight_group_size=weight_group_size,
            weight_symmetric=True,
            activation_mode="none",
        )


def test_quant_recipe_requires_group_size_for_group_granularity() -> None:
    with pytest.raises(ValidationError):
        QuantRecipe(
            algorithm="gptq",
            scheme="w4a16",
            weight_granularity="group",
            weight_symmetric=True,
            activation_mode="none",
        )


@pytest.mark.parametrize("weight_granularity", ["tensor", "channel"])
def test_quant_recipe_forbids_group_size_for_non_group_granularity(
    weight_granularity: str,
) -> None:
    payload = {
        "algorithm": "rtn",
        "scheme": "w8a16",
        "weight_granularity": weight_granularity,
        "weight_group_size": 128,
        "weight_symmetric": True,
        "activation_mode": "none",
    }

    with pytest.raises(ValidationError):
        QuantRecipe.model_validate(payload)


@pytest.mark.parametrize("scheme", ["w4a16", "w8a16"])
@pytest.mark.parametrize("activation_mode", ["dynamic", "static"])
def test_quant_recipe_rejects_activation_quantization_for_weight_only_schemes(
    scheme: str,
    activation_mode: str,
) -> None:
    payload = {
        "algorithm": "rtn",
        "scheme": scheme,
        "weight_granularity": "channel",
        "weight_symmetric": True,
        "activation_mode": activation_mode,
    }

    with pytest.raises(ValidationError):
        QuantRecipe.model_validate(payload)


@pytest.mark.parametrize("scheme", ["w8a8_int8", "w8a8_fp8"])
def test_quant_recipe_requires_activation_quantization_for_w8a8_schemes(
    scheme: str,
) -> None:
    payload = {
        "algorithm": "rtn",
        "scheme": scheme,
        "weight_granularity": "channel",
        "weight_symmetric": True,
        "activation_mode": "none",
    }

    with pytest.raises(ValidationError):
        QuantRecipe.model_validate(payload)


@pytest.mark.parametrize("ignored_module", ["", "   "])
def test_quant_recipe_rejects_empty_ignored_module(
    ignored_module: str,
) -> None:
    with pytest.raises(ValidationError):
        QuantRecipe(
            algorithm="gptq",
            scheme="w4a16",
            weight_granularity="group",
            weight_group_size=128,
            weight_symmetric=True,
            activation_mode="none",
            ignored_modules=(ignored_module,),
        )


def test_quant_recipe_uses_strict_tuple_typing() -> None:
    with pytest.raises(ValidationError):
        QuantRecipe.model_validate(
            {
                "algorithm": "gptq",
                "scheme": "w4a16",
                "weight_granularity": "group",
                "weight_group_size": 128,
                "weight_symmetric": True,
                "activation_mode": "none",
                "ignored_modules": ["lm_head"],
            }
        )


def test_quant_recipe_uses_strict_boolean_typing() -> None:
    with pytest.raises(ValidationError):
        QuantRecipe.model_validate(
            {
                "algorithm": "gptq",
                "scheme": "w4a16",
                "weight_granularity": "group",
                "weight_group_size": 128,
                "weight_symmetric": 1,
                "activation_mode": "none",
            }
        )


def test_quant_recipe_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        QuantRecipe.model_validate(
            {
                "algorithm": "gptq",
                "scheme": "w4a16",
                "weight_granularity": "group",
                "weight_group_size": 128,
                "weight_symmetric": True,
                "activation_mode": "none",
                "backend": "llm_compressor",
            }
        )


def test_quant_recipe_is_immutable() -> None:
    recipe = QuantRecipe(
        algorithm="gptq",
        scheme="w4a16",
        weight_granularity="group",
        weight_group_size=128,
        weight_symmetric=True,
        activation_mode="none",
    )

    with pytest.raises(ValidationError):
        recipe.__setattr__("algorithm", "awq")
