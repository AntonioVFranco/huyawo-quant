"""Tests for LLM Compressor native recipe construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import huyawo_quant.backends.llm_compressor_recipes as recipes
import pytest
from huyawo_quant.backends import (
    build_llm_compressor_gptq_w4a16_recipe,
)
from huyawo_quant.contracts import (
    HardwareTarget,
    ModelIdentity,
    QuantPlan,
    QuantRecipe,
    RuntimeTarget,
)


@dataclass
class _FakeWeights:
    num_bits: int = 4
    type: object = "int"
    symmetric: bool = True
    group_size: int | None = 128
    strategy: object = "group"

    def model_copy(
        self,
        *,
        update: dict[str, object],
    ) -> _FakeWeights:
        values: dict[str, object] = {
            "num_bits": self.num_bits,
            "type": self.type,
            "symmetric": self.symmetric,
            "group_size": self.group_size,
            "strategy": self.strategy,
        }
        values.update(update)

        return _FakeWeights(
            num_bits=values["num_bits"],
            type=values["type"],
            symmetric=values["symmetric"],
            group_size=values["group_size"],
            strategy=values["strategy"],
        )


@dataclass
class _FakeScheme:
    targets: list[str]
    weights: _FakeWeights | None
    input_activations: object | None = None
    output_activations: object | None = None

    def model_copy(
        self,
        *,
        update: dict[str, object],
    ) -> _FakeScheme:
        targets = update.get(
            "targets",
            self.targets,
        )
        weights = update.get(
            "weights",
            self.weights,
        )

        return _FakeScheme(
            targets=list(targets),
            weights=weights,
            input_activations=self.input_activations,
            output_activations=self.output_activations,
        )


class _FakeModifier:
    def __init__(
        self,
        **kwargs: object,
    ) -> None:
        self.config_groups = kwargs["config_groups"]
        self.ignore = kwargs["ignore"]
        self.requires_calibration_data = kwargs["requires_calibration_data"]


@dataclass
class _FakeRecipe:
    modifier: object

    def to_dict(self) -> dict[str, object]:
        return {
            "modifier": self.modifier,
        }


def _make_plan(
    *,
    algorithm: str = "gptq",
    scheme: str = "w4a16",
    weight_granularity: str = "group",
    weight_group_size: int | None = 64,
    weight_symmetric: bool = False,
    ignored_modules: tuple[str, ...] = ("lm_head",),
) -> QuantPlan:
    return QuantPlan(
        model=ModelIdentity(
            model_id="org/model",
            requested_revision="main",
            resolved_revision="a" * 40,
        ),
        recipe=QuantRecipe(
            algorithm=algorithm,
            scheme=scheme,
            weight_granularity=weight_granularity,
            weight_group_size=weight_group_size,
            weight_symmetric=weight_symmetric,
            activation_mode=(
                "none"
                if scheme
                in {
                    "w4a16",
                    "w8a16",
                }
                else "dynamic"
            ),
            ignored_modules=ignored_modules,
        ),
        hardware=HardwareTarget(
            device_name="NVIDIA RTX A5000",
            device_count=1,
            memory_bytes_per_device=(24_564 * 1024 * 1024),
            compute_capability_major=8,
            compute_capability_minor=6,
        ),
        runtime=RuntimeTarget(
            runtime="vllm",
            version="0.0.0-test",
        ),
        backend="llm_compressor",
        backend_version="0.14.0",
        resolved_modules=(
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
        ),
    )


def _install_fake_native_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recipes,
        "validate_llm_compressor_plan",
        lambda _plan: None,
    )

    def preset_factory(
        name: str,
        targets: list[str],
    ) -> object:
        assert name == "W4A16"
        assert targets == ["Linear"]

        return _FakeScheme(
            targets=list(targets),
            weights=_FakeWeights(),
        )

    def modifier_factory(
        **kwargs: object,
    ) -> object:
        return _FakeModifier(**kwargs)

    def recipe_factory(
        modifier: object,
    ) -> object:
        return _FakeRecipe(modifier=modifier)

    monkeypatch.setattr(
        recipes,
        "_load_gptq_w4a16_recipe_components",
        lambda: (
            preset_factory,
            modifier_factory,
            recipe_factory,
        ),
    )


def test_build_gptq_w4a16_recipe_maps_exact_plan_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_native_components(monkeypatch)

    plan = _make_plan(
        weight_group_size=64,
        weight_symmetric=False,
        ignored_modules=(
            "lm_head",
            "model.embed_tokens",
        ),
    )

    result = build_llm_compressor_gptq_w4a16_recipe(plan)

    assert isinstance(
        result,
        _FakeRecipe,
    )

    modifier = result.modifier

    assert isinstance(
        modifier,
        _FakeModifier,
    )

    assert modifier.requires_calibration_data is True
    assert modifier.ignore == [
        "lm_head",
        "model.embed_tokens",
    ]

    scheme = modifier.config_groups["group_0"]

    assert scheme.targets == list(plan.resolved_modules)
    assert scheme.weights is not None
    assert scheme.weights.num_bits == 4
    assert scheme.weights.type == "int"
    assert scheme.weights.strategy == "group"
    assert scheme.weights.group_size == 64
    assert scheme.weights.symmetric is False
    assert scheme.input_activations is None
    assert scheme.output_activations is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "algorithm",
            "awq",
            "algorithm=gptq",
        ),
        (
            "scheme",
            "w8a16",
            "scheme=w4a16",
        ),
        (
            "weight_granularity",
            "channel",
            "weight_granularity=group",
        ),
    ),
)
def test_build_gptq_w4a16_recipe_rejects_wrong_recipe_shape(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setattr(
        recipes,
        "validate_llm_compressor_plan",
        lambda _plan: None,
    )

    arguments: dict[str, Any] = {
        "algorithm": "gptq",
        "scheme": "w4a16",
        "weight_granularity": "group",
        "weight_group_size": 64,
    }
    arguments[field] = value

    if field == "weight_granularity":
        arguments["weight_group_size"] = None

    plan = _make_plan(**arguments)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        build_llm_compressor_gptq_w4a16_recipe(plan)


def test_build_gptq_w4a16_recipe_rejects_empty_resolved_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recipes,
        "validate_llm_compressor_plan",
        lambda _plan: None,
    )

    plan = _make_plan().model_copy(
        update={
            "resolved_modules": (),
        }
    )

    with pytest.raises(
        ValueError,
        match="resolved_modules",
    ):
        build_llm_compressor_gptq_w4a16_recipe(plan)


def test_native_component_loader_reports_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recipes,
        "validate_llm_compressor_plan",
        lambda _plan: None,
    )

    def missing_import(
        name: str,
    ) -> object:
        raise ModuleNotFoundError(
            f"No module named '{name}'",
            name=name,
        )

    monkeypatch.setattr(
        recipes,
        "import_module",
        missing_import,
    )

    with pytest.raises(
        RuntimeError,
        match="required native recipe dependency",
    ):
        build_llm_compressor_gptq_w4a16_recipe(_make_plan())


def test_native_preset_validation_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recipes,
        "validate_llm_compressor_plan",
        lambda _plan: None,
    )

    def bad_preset(
        _name: str,
        targets: list[str],
    ) -> object:
        return _FakeScheme(
            targets=targets,
            weights=_FakeWeights(
                num_bits=8,
            ),
        )

    monkeypatch.setattr(
        recipes,
        "_load_gptq_w4a16_recipe_components",
        lambda: (
            bad_preset,
            lambda **kwargs: _FakeModifier(**kwargs),
            lambda modifier: _FakeRecipe(modifier),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="weight bit width",
    ):
        build_llm_compressor_gptq_w4a16_recipe(_make_plan())
