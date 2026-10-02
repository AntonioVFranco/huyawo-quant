"""Native LLM Compressor recipe construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Protocol, Self, cast

from huyawo_quant.backends.llm_compressor import validate_llm_compressor_plan
from huyawo_quant.contracts import QuantPlan


class _NativeWeights(Protocol):
    num_bits: int
    type: object
    symmetric: bool
    group_size: int | None
    strategy: object

    def model_copy(
        self,
        *,
        update: Mapping[str, object],
    ) -> Self: ...


class _NativeScheme(Protocol):
    targets: list[str]
    weights: _NativeWeights | None
    input_activations: object | None
    output_activations: object | None

    def model_copy(
        self,
        *,
        update: Mapping[str, object],
    ) -> Self: ...


class _NativeModifier(Protocol):
    config_groups: Mapping[str, _NativeScheme]
    ignore: list[str]
    requires_calibration_data: bool


PresetFactory = Callable[[str, list[str]], object]
ModifierFactory = Callable[..., object]
RecipeFactory = Callable[[object], object]


def _native_enum_value(value: object) -> str:
    return str(getattr(value, "value", value)).lower()


def _load_gptq_w4a16_recipe_components() -> tuple[PresetFactory, ModifierFactory, RecipeFactory]:
    try:
        quantization_module = import_module("compressed_tensors.quantization")
        gptq_module = import_module("llmcompressor.modifiers.gptq")
        recipe_module = import_module("llmcompressor.recipe")
    except ModuleNotFoundError as error:
        missing_name = error.name or "<unknown>"

        raise RuntimeError(
            f"required native recipe dependency is not importable: {missing_name}"
        ) from error

    preset_factory = getattr(
        quantization_module,
        "preset_name_to_scheme",
        None,
    )
    modifier_factory = getattr(
        gptq_module,
        "GPTQModifier",
        None,
    )
    recipe_class = getattr(
        recipe_module,
        "Recipe",
        None,
    )
    recipe_factory = getattr(
        recipe_class,
        "from_modifiers",
        None,
    )

    if not callable(preset_factory):
        raise RuntimeError("compressed-tensors does not expose preset_name_to_scheme")

    if not callable(modifier_factory):
        raise RuntimeError("LLM Compressor does not expose GPTQModifier")

    if not callable(recipe_factory):
        raise RuntimeError("LLM Compressor Recipe does not expose from_modifiers")

    return (
        cast(PresetFactory, preset_factory),
        cast(ModifierFactory, modifier_factory),
        cast(RecipeFactory, recipe_factory),
    )


def _validate_gptq_w4a16_plan(
    plan: QuantPlan,
) -> None:
    validate_llm_compressor_plan(plan)

    if plan.recipe.algorithm != "gptq":
        raise ValueError("GPTQ W4A16 recipe requires algorithm=gptq")

    if plan.recipe.scheme != "w4a16":
        raise ValueError("GPTQ W4A16 recipe requires scheme=w4a16")

    if plan.recipe.activation_mode != "none":
        raise ValueError("GPTQ W4A16 recipe requires activation_mode=none")

    if plan.recipe.weight_granularity != "group":
        raise ValueError("GPTQ W4A16 recipe requires weight_granularity=group")

    group_size = plan.recipe.weight_group_size

    if group_size is None or group_size <= 0:
        raise ValueError("GPTQ W4A16 recipe requires a positive weight_group_size")

    if not plan.resolved_modules:
        raise ValueError("GPTQ W4A16 recipe requires resolved_modules")


def _validate_native_w4a16_preset(
    preset: _NativeScheme,
) -> _NativeWeights:
    if preset.targets != ["Linear"]:
        raise RuntimeError(f"unexpected W4A16 preset targets: {preset.targets}")

    weights = preset.weights

    if weights is None:
        raise RuntimeError("W4A16 preset does not define weight quantization")

    if weights.num_bits != 4:
        raise RuntimeError(f"unexpected W4A16 weight bit width: {weights.num_bits}")

    if _native_enum_value(weights.type) != "int":
        raise RuntimeError(f"unexpected W4A16 weight type: {weights.type}")

    if _native_enum_value(weights.strategy) != "group":
        raise RuntimeError(f"unexpected W4A16 weight strategy: {weights.strategy}")

    if weights.group_size != 128:
        raise RuntimeError(f"unexpected W4A16 preset group size: {weights.group_size}")

    if weights.symmetric is not True:
        raise RuntimeError(f"unexpected W4A16 preset symmetry: {weights.symmetric}")

    if preset.input_activations is not None:
        raise RuntimeError("W4A16 preset unexpectedly quantizes input activations")

    if preset.output_activations is not None:
        raise RuntimeError("W4A16 preset unexpectedly quantizes output activations")

    return weights


def build_llm_compressor_gptq_w4a16_recipe(
    plan: QuantPlan,
) -> object:
    """Build the native GPTQ W4A16 recipe for an exact QuantPlan."""

    _validate_gptq_w4a16_plan(plan)

    (
        preset_factory,
        modifier_factory,
        recipe_factory,
    ) = _load_gptq_w4a16_recipe_components()

    preset = cast(
        _NativeScheme,
        preset_factory(
            "W4A16",
            ["Linear"],
        ),
    )

    preset_weights = _validate_native_w4a16_preset(preset)

    group_size = plan.recipe.weight_group_size

    if group_size is None:
        raise RuntimeError("validated GPTQ W4A16 plan lost weight_group_size")

    native_weights = preset_weights.model_copy(
        update={
            "group_size": group_size,
            "symmetric": plan.recipe.weight_symmetric,
        }
    )

    native_targets = list(plan.resolved_modules)

    native_scheme = preset.model_copy(
        update={
            "targets": native_targets,
            "weights": native_weights,
        }
    )

    if native_scheme.targets != native_targets:
        raise RuntimeError("native W4A16 target mapping was not preserved")

    mapped_weights = native_scheme.weights

    if mapped_weights is None:
        raise RuntimeError("native W4A16 weights disappeared after mapping")

    if mapped_weights.group_size != group_size:
        raise RuntimeError("native W4A16 group-size mapping was not preserved")

    if mapped_weights.symmetric is not plan.recipe.weight_symmetric:
        raise RuntimeError("native W4A16 symmetry mapping was not preserved")

    native_ignore = list(plan.recipe.ignored_modules)

    modifier = cast(
        _NativeModifier,
        modifier_factory(
            config_groups={
                "group_0": native_scheme,
            },
            ignore=native_ignore,
            requires_calibration_data=True,
        ),
    )

    if modifier.requires_calibration_data is not True:
        raise RuntimeError("GPTQModifier unexpectedly does not require calibration data")

    if modifier.ignore != native_ignore:
        raise RuntimeError("GPTQModifier ignore mapping was not preserved")

    modifier_scheme = modifier.config_groups.get("group_0")

    if modifier_scheme is None:
        raise RuntimeError("GPTQModifier lost the W4A16 config group")

    if modifier_scheme.targets != native_targets:
        raise RuntimeError("GPTQModifier target mapping was not preserved")

    modifier_weights = modifier_scheme.weights

    if modifier_weights is None:
        raise RuntimeError("GPTQModifier lost W4A16 weight configuration")

    if modifier_weights.group_size != group_size:
        raise RuntimeError("GPTQModifier group-size mapping was not preserved")

    if modifier_weights.symmetric is not plan.recipe.weight_symmetric:
        raise RuntimeError("GPTQModifier symmetry mapping was not preserved")

    native_recipe = recipe_factory(modifier)

    if not callable(
        getattr(
            native_recipe,
            "to_dict",
            None,
        )
    ):
        raise RuntimeError("native GPTQ W4A16 recipe does not expose to_dict")

    return native_recipe
