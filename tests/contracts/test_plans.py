"""Tests for resolved quantization plan contracts."""

import pytest
from huyawo_quant.contracts import (
    HardwareTarget,
    ModelIdentity,
    QuantPlan,
    QuantRecipe,
    RuntimeTarget,
)
from pydantic import ValidationError


def make_model() -> ModelIdentity:
    return ModelIdentity(
        model_id="organization/model",
        requested_revision="main",
        resolved_revision="resolved-model-revision",
    )


def make_recipe() -> QuantRecipe:
    return QuantRecipe(
        algorithm="gptq",
        scheme="w4a16",
        weight_granularity="group",
        weight_group_size=128,
        weight_symmetric=True,
        activation_mode="none",
        ignored_modules=("lm_head",),
    )


def make_hardware() -> HardwareTarget:
    return HardwareTarget(
        device_name="NVIDIA RTX A4500",
        device_count=1,
        memory_bytes_per_device=21_474_836_480,
        compute_capability_major=8,
        compute_capability_minor=6,
    )


def make_runtime() -> RuntimeTarget:
    return RuntimeTarget(
        runtime="vllm",
        version="test-version",
    )


def make_plan() -> QuantPlan:
    return QuantPlan(
        model=make_model(),
        recipe=make_recipe(),
        hardware=make_hardware(),
        runtime=make_runtime(),
        backend="llm_compressor",
        backend_version="test-version",
        resolved_modules=(
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
        ),
    )


def test_quant_plan_preserves_nested_contracts() -> None:
    model = make_model()
    recipe = make_recipe()
    hardware = make_hardware()
    runtime = make_runtime()

    plan = QuantPlan(
        model=model,
        recipe=recipe,
        hardware=hardware,
        runtime=runtime,
        backend="llm_compressor",
        backend_version="test-version",
        resolved_modules=("model.layers.0.self_attn.q_proj",),
    )

    assert plan.model is model
    assert plan.recipe is recipe
    assert plan.hardware is hardware
    assert plan.runtime is runtime


def test_quant_plan_serializes_expected_fields() -> None:
    plan = make_plan()

    assert plan.model_dump(mode="json") == {
        "model": {
            "source": "huggingface",
            "model_id": "organization/model",
            "requested_revision": "main",
            "resolved_revision": "resolved-model-revision",
        },
        "recipe": {
            "algorithm": "gptq",
            "scheme": "w4a16",
            "weight_granularity": "group",
            "weight_group_size": 128,
            "weight_symmetric": True,
            "activation_mode": "none",
            "ignored_modules": ["lm_head"],
        },
        "hardware": {
            "vendor": "nvidia",
            "accelerator": "cuda",
            "device_name": "NVIDIA RTX A4500",
            "device_count": 1,
            "memory_bytes_per_device": 21_474_836_480,
            "compute_capability_major": 8,
            "compute_capability_minor": 6,
        },
        "runtime": {
            "runtime": "vllm",
            "version": "test-version",
        },
        "backend": "llm_compressor",
        "backend_version": "test-version",
        "resolved_modules": [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
        ],
    }


@pytest.mark.parametrize("backend", ["llm_compressor", "torchao"])
def test_quant_plan_accepts_tier_one_backends(backend: str) -> None:
    payload = make_plan().model_dump()
    payload["backend"] = backend

    plan = QuantPlan.model_validate(payload)

    assert plan.backend == backend


def test_quant_plan_rejects_unknown_backend() -> None:
    payload = make_plan().model_dump()
    payload["backend"] = "bitsandbytes"

    with pytest.raises(ValidationError):
        QuantPlan.model_validate(payload)


@pytest.mark.parametrize("backend_version", ["", "   "])
def test_quant_plan_rejects_empty_backend_version(
    backend_version: str,
) -> None:
    with pytest.raises(ValidationError):
        QuantPlan(
            model=make_model(),
            recipe=make_recipe(),
            hardware=make_hardware(),
            runtime=make_runtime(),
            backend="llm_compressor",
            backend_version=backend_version,
            resolved_modules=("model.layers.0.self_attn.q_proj",),
        )


def test_quant_plan_rejects_empty_resolved_modules() -> None:
    with pytest.raises(ValidationError):
        QuantPlan(
            model=make_model(),
            recipe=make_recipe(),
            hardware=make_hardware(),
            runtime=make_runtime(),
            backend="llm_compressor",
            backend_version="test-version",
            resolved_modules=(),
        )


@pytest.mark.parametrize("module_name", ["", "   "])
def test_quant_plan_rejects_empty_resolved_module_name(
    module_name: str,
) -> None:
    with pytest.raises(ValidationError):
        QuantPlan(
            model=make_model(),
            recipe=make_recipe(),
            hardware=make_hardware(),
            runtime=make_runtime(),
            backend="llm_compressor",
            backend_version="test-version",
            resolved_modules=(module_name,),
        )


def test_quant_plan_uses_strict_tuple_typing() -> None:
    payload = make_plan().model_dump()
    payload["resolved_modules"] = ["model.layers.0.self_attn.q_proj"]

    with pytest.raises(ValidationError):
        QuantPlan.model_validate(payload)


def test_quant_plan_rejects_extra_fields() -> None:
    payload = make_plan().model_dump()
    payload["backend_config"] = {"example": True}

    with pytest.raises(ValidationError):
        QuantPlan.model_validate(payload)


def test_quant_plan_is_immutable() -> None:
    plan = make_plan()

    with pytest.raises(ValidationError):
        plan.__setattr__("backend", "torchao")
