from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import version as distribution_version

import pytest
from huyawo_quant.baseline import BaselineDtype
from huyawo_quant.contracts import (
    HardwareTarget,
    ModelIdentity,
    RuntimeTarget,
    TokenizerIdentity,
)
from huyawo_quant.performance.manifest import (
    build_performance_authoritative_run_manifest,
)
from huyawo_quant.performance.protocol import (
    PRIMARY_WORKLOAD_ID,
    build_generation_kwargs,
    build_primary_workload_profile,
)

_MODEL_IDENTITY = ModelIdentity(
    source="huggingface",
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    requested_revision="7ae557604adf67be50417f59c2c2f167def9a775",
    resolved_revision="7ae557604adf67be50417f59c2c2f167def9a775",
)

_TOKENIZER_IDENTITY = TokenizerIdentity(
    source="huggingface",
    tokenizer_id="Qwen/Qwen2.5-0.5B-Instruct",
    requested_revision="7ae557604adf67be50417f59c2c2f167def9a775",
    resolved_revision="7ae557604adf67be50417f59c2c2f167def9a775",
)

_HARDWARE_TARGET = HardwareTarget(
    device_name="NVIDIA RTX A4500",
    device_count=1,
    memory_bytes_per_device=20470 * 1024 * 1024,
    compute_capability_major=8,
    compute_capability_minor=6,
)

_RUNTIME_TARGET = RuntimeTarget(
    runtime="transformers",
    version="5.17.0",
)

_CREATED_AT = datetime(2026, 10, 1, 15, 30, tzinfo=UTC)
_REPOSITORY_COMMIT = "a" * 40
_ENVIRONMENT_FINGERPRINT_SHA256 = "b" * 64
_PROMPT_TOKEN_IDS_SHA256 = "c" * 64
_PROMPT_FIXTURE_ID = "synthetic_interactive_prompt_v1"


def _build_manifest(
    *,
    run_id: str = "m6-test-run",
    created_at: datetime = _CREATED_AT,
    repository_commit: str = _REPOSITORY_COMMIT,
    baseline_dtype: BaselineDtype = "bfloat16",
    environment_fingerprint_sha256: str = (_ENVIRONMENT_FINGERPRINT_SHA256),
    prompt_fixture_id: str = _PROMPT_FIXTURE_ID,
    prompt_token_ids_sha256: str = _PROMPT_TOKEN_IDS_SHA256,
    runtime_target: RuntimeTarget = _RUNTIME_TARGET,
    hardware_target: HardwareTarget = _HARDWARE_TARGET,
) -> dict[str, object]:
    return build_performance_authoritative_run_manifest(
        _MODEL_IDENTITY,
        _TOKENIZER_IDENTITY,
        hardware_target,
        runtime_target,
        run_id=run_id,
        created_at=created_at,
        repository_commit=repository_commit,
        baseline_dtype=baseline_dtype,
        environment_fingerprint_sha256=(environment_fingerprint_sha256),
        prompt_fixture_id=prompt_fixture_id,
        prompt_token_ids_sha256=prompt_token_ids_sha256,
    )


def test_authoritative_run_manifest_freezes_protocol_values() -> None:
    manifest = _build_manifest()
    workload = build_primary_workload_profile()

    assert manifest["schema_version"] == "1"
    assert manifest["run_id"] == "m6-test-run"
    assert manifest["run_kind"] == "baseline"
    assert manifest["purpose"] == "performance_baseline"
    assert manifest["authoritative"] is True
    assert manifest["quant_version"] == distribution_version("huyawo-quant")
    assert manifest["created_at"] == "2026-10-01T15:30:00+00:00"
    assert manifest["repository_commit"] == _REPOSITORY_COMMIT

    assert manifest["model_identity"] == _MODEL_IDENTITY.model_dump(mode="json")
    assert manifest["tokenizer_identity"] == (_TOKENIZER_IDENTITY.model_dump(mode="json"))

    assert manifest["baseline_dtype"] == "bfloat16"
    assert manifest["runtime_target"] == _RUNTIME_TARGET.model_dump(mode="json")
    assert manifest["hardware_target"] == _HARDWARE_TARGET.model_dump(mode="json")
    assert manifest["environment_fingerprint_sha256"] == _ENVIRONMENT_FINGERPRINT_SHA256

    assert manifest["workload_id"] == PRIMARY_WORKLOAD_ID
    assert manifest["workload_profile"] == workload.model_dump(mode="json")
    assert manifest["warmup_runs"] == 3
    assert manifest["measured_runs"] == 10

    assert manifest["prompt_fixture"] == {
        "fixture_id": _PROMPT_FIXTURE_ID,
        "add_special_tokens": False,
        "token_count": 256,
        "token_ids_sha256": _PROMPT_TOKEN_IDS_SHA256,
    }

    assert manifest["generation_configuration"] == {
        "model_eval": True,
        "torch_inference_mode": True,
        "kwargs": build_generation_kwargs(),
        "custom_stopping_criteria": False,
        "seed": 42,
        "required_generated_token_count": 64,
    }


def test_authoritative_run_manifest_metric_definitions_are_exact() -> None:
    manifest = _build_manifest()

    assert manifest["metric_definitions"] == [
        {
            "metric_id": "generation/e2e_latency_seconds",
            "unit": "seconds",
            "direction": "lower_is_better",
            "aggregation": "median",
            "expected_observation_count": 10,
        },
        {
            "metric_id": "generation/ttft_seconds",
            "unit": "seconds",
            "direction": "lower_is_better",
            "aggregation": "median",
            "expected_observation_count": 10,
        },
        {
            "metric_id": "generation/inter_token_latency_seconds",
            "unit": "seconds",
            "direction": "lower_is_better",
            "aggregation": "median",
            "expected_observation_count": 10,
        },
        {
            "metric_id": ("generation/output_throughput_tokens_per_second"),
            "unit": "tokens/second",
            "direction": "higher_is_better",
            "aggregation": "median",
            "expected_observation_count": 10,
        },
        {
            "metric_id": "memory/torch_allocated_vram_bytes",
            "unit": "bytes",
            "direction": "lower_is_better",
            "aggregation": "last",
            "expected_observation_count": 1,
        },
        {
            "metric_id": "memory/torch_peak_allocated_vram_bytes",
            "unit": "bytes",
            "direction": "lower_is_better",
            "aggregation": "max",
            "expected_observation_count": 10,
        },
    ]


def test_authoritative_run_manifest_is_json_compatible() -> None:
    manifest = _build_manifest()

    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    )
    decoded = json.loads(encoded)

    assert decoded == manifest


@pytest.mark.parametrize("run_id", ["", "   "])
def test_authoritative_run_manifest_rejects_empty_run_id(
    run_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="run_id must contain a non-whitespace character",
    ):
        _build_manifest(run_id=run_id)


def test_authoritative_run_manifest_rejects_naive_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="created_at must be timezone-aware",
    ):
        _build_manifest(created_at=datetime(2026, 10, 1, 15, 30))


def test_authoritative_run_manifest_rejects_invalid_commit() -> None:
    with pytest.raises(
        ValueError,
        match=("repository_commit must contain exactly 40 lowercase hexadecimal characters"),
    ):
        _build_manifest(repository_commit="abc123")


def test_authoritative_run_manifest_rejects_invalid_environment_hash() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "environment_fingerprint_sha256 must contain exactly 64 "
            "lowercase hexadecimal characters"
        ),
    ):
        _build_manifest(environment_fingerprint_sha256="abc123")


def test_authoritative_run_manifest_rejects_invalid_prompt_hash() -> None:
    with pytest.raises(
        ValueError,
        match=("prompt_token_ids_sha256 must contain exactly 64 lowercase hexadecimal characters"),
    ):
        _build_manifest(prompt_token_ids_sha256="abc123")


def test_authoritative_run_manifest_rejects_float16() -> None:
    with pytest.raises(
        ValueError,
        match=("authoritative Milestone 6 baseline_dtype must be bfloat16"),
    ):
        _build_manifest(baseline_dtype="float16")


def test_authoritative_run_manifest_rejects_wrong_runtime() -> None:
    runtime_target = RuntimeTarget(
        runtime="vllm",
        version="0.1.0",
    )

    with pytest.raises(
        ValueError,
        match=("authoritative Milestone 6 runtime must be transformers"),
    ):
        _build_manifest(runtime_target=runtime_target)


def test_authoritative_run_manifest_rejects_wrong_runtime_version() -> None:
    runtime_target = RuntimeTarget(
        runtime="transformers",
        version="5.16.0",
    )

    with pytest.raises(
        ValueError,
        match=("authoritative Milestone 6 transformers version must be 5.17.0"),
    ):
        _build_manifest(runtime_target=runtime_target)


def test_authoritative_run_manifest_rejects_multiple_devices() -> None:
    hardware_target = HardwareTarget(
        device_name="NVIDIA RTX A4500",
        device_count=2,
        memory_bytes_per_device=20470 * 1024 * 1024,
        compute_capability_major=8,
        compute_capability_minor=6,
    )

    with pytest.raises(
        ValueError,
        match=("authoritative Milestone 6 hardware target must contain exactly one device"),
    ):
        _build_manifest(hardware_target=hardware_target)
