"""Tests for the LLM Compressor backend adapter."""

from __future__ import annotations

from pathlib import Path

import huyawo_quant.backends.llm_compressor as backend
import pytest
from huyawo_quant.backends import (
    build_llm_compressor_oneshot_kwargs,
    execute_llm_compressor_oneshot,
    get_llm_compressor_backend_metadata,
    get_llm_compressor_version,
    llm_compressor_scheme_min_compute_capability,
    translate_llm_compressor_failure,
    validate_llm_compressor_plan,
)
from huyawo_quant.contracts import (
    HardwareTarget,
    ModelIdentity,
    QuantPlan,
    QuantRecipe,
    RuntimeTarget,
)


def _make_recipe(
    *,
    algorithm: str = "rtn",
    scheme: str = "w4a16",
) -> QuantRecipe:
    if scheme in {
        "w4a16",
        "w8a16",
    }:
        return QuantRecipe(
            algorithm=algorithm,
            scheme=scheme,
            weight_granularity="group",
            weight_group_size=128,
            weight_symmetric=True,
            activation_mode="none",
            ignored_modules=("lm_head",),
        )

    return QuantRecipe(
        algorithm=algorithm,
        scheme=scheme,
        weight_granularity="channel",
        weight_group_size=None,
        weight_symmetric=True,
        activation_mode="dynamic",
        ignored_modules=("lm_head",),
    )


def _make_plan(
    *,
    backend_name: str = "llm_compressor",
    backend_version: str = "0.14.0",
    algorithm: str = "rtn",
    scheme: str = "w4a16",
    compute_capability: tuple[int, int] = (8, 6),
) -> QuantPlan:
    return QuantPlan(
        model=ModelIdentity(
            model_id="org/model",
            requested_revision="main",
            resolved_revision="a" * 40,
        ),
        recipe=_make_recipe(
            algorithm=algorithm,
            scheme=scheme,
        ),
        hardware=HardwareTarget(
            device_name="NVIDIA RTX A5000",
            device_count=1,
            memory_bytes_per_device=(24_564 * 1024 * 1024),
            compute_capability_major=(compute_capability[0]),
            compute_capability_minor=(compute_capability[1]),
        ),
        runtime=RuntimeTarget(
            runtime="vllm",
            version="0.0.0-test",
        ),
        backend=backend_name,
        backend_version=backend_version,
        resolved_modules=("model.layers.0.self_attn.q_proj",),
    )


def test_get_llm_compressor_version_returns_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend,
        "distribution_version",
        lambda _name: "0.14.0",
    )

    assert get_llm_compressor_version() == "0.14.0"


def test_get_llm_compressor_version_returns_none_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> str:
        raise backend.PackageNotFoundError

    monkeypatch.setattr(
        backend,
        "distribution_version",
        missing,
    )

    assert get_llm_compressor_version() is None


def test_backend_metadata_is_huyawo_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "llmcompressor": "0.14.0",
        "compressed-tensors": "0.19.0",
    }

    monkeypatch.setattr(
        backend,
        "distribution_version",
        versions.__getitem__,
    )

    assert get_llm_compressor_backend_metadata() == {
        "backend": "llm_compressor",
        "backend_distribution": "llmcompressor",
        "backend_version": "0.14.0",
        "artifact_format": "compressed-tensors",
        "compressed_tensors_version": "0.19.0",
    }


def test_scheme_minimum_compute_capability_facts() -> None:
    assert llm_compressor_scheme_min_compute_capability() == {
        "w4a16": (7, 5),
        "w8a16": (7, 5),
        "w8a8_int8": (7, 5),
        "w8a8_fp8": (8, 9),
    }


def test_validate_plan_accepts_w4a16_on_a5000(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend,
        "get_llm_compressor_version",
        lambda: "0.14.0",
    )

    validate_llm_compressor_plan(_make_plan())


def test_validate_plan_rejects_wrong_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend,
        "get_llm_compressor_version",
        lambda: "0.14.0",
    )

    with pytest.raises(
        ValueError,
        match="backend must be llm_compressor",
    ):
        validate_llm_compressor_plan(
            _make_plan(
                backend_name="torchao",
            )
        )


def test_validate_plan_rejects_unsupported_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend,
        "get_llm_compressor_version",
        lambda: "0.15.0",
    )

    with pytest.raises(
        RuntimeError,
        match="unsupported installed LLM Compressor version",
    ):
        validate_llm_compressor_plan(
            _make_plan(
                backend_version="0.15.0",
            )
        )


def test_validate_plan_rejects_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend,
        "get_llm_compressor_version",
        lambda: "0.14.0",
    )

    with pytest.raises(
        ValueError,
        match="backend_version",
    ):
        validate_llm_compressor_plan(
            _make_plan(
                backend_version="0.13.0",
            )
        )


def test_validate_plan_rejects_fp8_on_compute_capability_86(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend,
        "get_llm_compressor_version",
        lambda: "0.14.0",
    )

    with pytest.raises(
        ValueError,
        match="required=8.9",
    ):
        validate_llm_compressor_plan(
            _make_plan(
                scheme="w8a8_fp8",
                compute_capability=(8, 6),
            )
        )


def test_validate_plan_accepts_fp8_on_compute_capability_89(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend,
        "get_llm_compressor_version",
        lambda: "0.14.0",
    )

    validate_llm_compressor_plan(
        _make_plan(
            scheme="w8a8_fp8",
            compute_capability=(8, 9),
        )
    )


def test_build_oneshot_kwargs_maps_plan_identity_and_execution_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        backend,
        "get_llm_compressor_version",
        lambda: "0.14.0",
    )

    native_recipe = object()
    dataset = object()
    output_dir = tmp_path / "artifact"

    kwargs = build_llm_compressor_oneshot_kwargs(
        plan=_make_plan(),
        native_recipe=native_recipe,
        dataset=dataset,
        output_dir=output_dir,
        save_compressed=True,
        num_calibration_samples=128,
        max_seq_length=2048,
    )

    assert kwargs == {
        "model": "org/model",
        "model_revision": "a" * 40,
        "recipe": native_recipe,
        "dataset": dataset,
        "save_compressed": True,
        "output_dir": str(output_dir),
        "num_calibration_samples": 128,
        "max_seq_length": 2048,
    }


def test_build_oneshot_kwargs_rejects_missing_native_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend,
        "get_llm_compressor_version",
        lambda: "0.14.0",
    )

    with pytest.raises(
        ValueError,
        match="native_recipe",
    ):
        build_llm_compressor_oneshot_kwargs(
            plan=_make_plan(),
            native_recipe=None,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("num_calibration_samples", 0),
        ("max_seq_length", 0),
    ),
)
def test_build_oneshot_kwargs_rejects_nonpositive_execution_limits(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
) -> None:
    monkeypatch.setattr(
        backend,
        "get_llm_compressor_version",
        lambda: "0.14.0",
    )

    arguments: dict[str, object] = {
        "plan": _make_plan(),
        "native_recipe": object(),
        field: value,
    }

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        build_llm_compressor_oneshot_kwargs(
            **arguments,
        )


def test_load_oneshot_runner_rejects_unsupported_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend,
        "get_llm_compressor_version",
        lambda: "0.15.0",
    )

    with pytest.raises(
        RuntimeError,
        match="unsupported installed LLM Compressor version",
    ):
        backend._load_oneshot_runner()


def test_execute_oneshot_uses_injected_runner() -> None:
    observed: dict[str, object] = {}

    def runner(**kwargs: object) -> object:
        observed.update(kwargs)
        return "result"

    result = execute_llm_compressor_oneshot(
        kwargs={
            "model": "org/model",
            "recipe": "native",
        },
        runner=runner,
    )

    assert result == "result"
    assert observed == {
        "model": "org/model",
        "recipe": "native",
    }


def test_execute_oneshot_uses_dynamic_runner_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def runner(**kwargs: object) -> object:
        return kwargs["model"]

    monkeypatch.setattr(
        backend,
        "_load_oneshot_runner",
        lambda: runner,
    )

    assert (
        execute_llm_compressor_oneshot(
            kwargs={"model": "org/model"},
        )
        == "org/model"
    )


@pytest.mark.parametrize(
    ("error", "category"),
    (
        (
            ValueError("bad recipe"),
            "llm_compressor_validation_error",
        ),
        (
            RuntimeError("backend failed"),
            "llm_compressor_runtime_error",
        ),
        (
            OSError("save failed"),
            "llm_compressor_io_error",
        ),
        (
            Exception("unexpected"),
            "llm_compressor_error",
        ),
    ),
)
def test_translate_failure_maps_backend_exception(
    error: Exception,
    category: str,
) -> None:
    failure = translate_llm_compressor_failure(
        error,
        stage="quantization",
    )

    assert failure.stage == "quantization"
    assert failure.category == category
    assert type(error).__name__ in failure.message
    assert str(error) in failure.message
