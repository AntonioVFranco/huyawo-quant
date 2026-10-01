from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import version as distribution_version

from huyawo_quant.baseline import BaselineDtype
from huyawo_quant.contracts import (
    HardwareTarget,
    ModelIdentity,
    RuntimeTarget,
    TokenizerIdentity,
)
from huyawo_quant.performance.protocol import (
    PRIMARY_WORKLOAD_ID,
    build_generation_kwargs,
    build_primary_workload_profile,
)

_PERFORMANCE_SEED = 42
_TRANSFORMERS_RUNTIME_VERSION = "5.17.0"


def _require_nonblank_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    if not value or value.isspace():
        raise ValueError(f"{name} must contain a non-whitespace character")

    return value


def _require_lower_hex(
    value: object,
    *,
    name: str,
    length: int,
) -> str:
    text = _require_nonblank_string(value, name=name)

    if len(text) != length:
        raise ValueError(f"{name} must contain exactly {length} lowercase hexadecimal characters")

    if any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must contain only lowercase hexadecimal characters")

    return text


def _build_metric_definitions(
    *,
    measured_runs: int,
) -> list[dict[str, object]]:
    return [
        {
            "metric_id": "generation/e2e_latency_seconds",
            "unit": "seconds",
            "direction": "lower_is_better",
            "aggregation": "median",
            "expected_observation_count": measured_runs,
        },
        {
            "metric_id": "generation/ttft_seconds",
            "unit": "seconds",
            "direction": "lower_is_better",
            "aggregation": "median",
            "expected_observation_count": measured_runs,
        },
        {
            "metric_id": "generation/inter_token_latency_seconds",
            "unit": "seconds",
            "direction": "lower_is_better",
            "aggregation": "median",
            "expected_observation_count": measured_runs,
        },
        {
            "metric_id": ("generation/output_throughput_tokens_per_second"),
            "unit": "tokens/second",
            "direction": "higher_is_better",
            "aggregation": "median",
            "expected_observation_count": measured_runs,
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
            "expected_observation_count": measured_runs,
        },
    ]


def build_performance_authoritative_run_manifest(
    model_identity: ModelIdentity,
    tokenizer_identity: TokenizerIdentity,
    hardware_target: HardwareTarget,
    runtime_target: RuntimeTarget,
    *,
    run_id: str,
    created_at: datetime,
    repository_commit: str,
    baseline_dtype: BaselineDtype,
    environment_fingerprint_sha256: str,
    prompt_fixture_id: str,
    prompt_token_ids_sha256: str,
) -> dict[str, object]:
    """Build the JSON-compatible authoritative performance run manifest."""
    if not isinstance(model_identity, ModelIdentity):
        raise TypeError("model_identity must be a ModelIdentity")

    if not isinstance(tokenizer_identity, TokenizerIdentity):
        raise TypeError("tokenizer_identity must be a TokenizerIdentity")

    if not isinstance(hardware_target, HardwareTarget):
        raise TypeError("hardware_target must be a HardwareTarget")

    if not isinstance(runtime_target, RuntimeTarget):
        raise TypeError("runtime_target must be a RuntimeTarget")

    normalized_run_id = _require_nonblank_string(
        run_id,
        name="run_id",
    )

    normalized_prompt_fixture_id = _require_nonblank_string(
        prompt_fixture_id,
        name="prompt_fixture_id",
    )

    normalized_repository_commit = _require_lower_hex(
        repository_commit,
        name="repository_commit",
        length=40,
    )

    normalized_environment_fingerprint_sha256 = _require_lower_hex(
        environment_fingerprint_sha256,
        name="environment_fingerprint_sha256",
        length=64,
    )

    normalized_prompt_token_ids_sha256 = _require_lower_hex(
        prompt_token_ids_sha256,
        name="prompt_token_ids_sha256",
        length=64,
    )

    if not isinstance(created_at, datetime):
        raise TypeError("created_at must be a datetime")

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")

    if baseline_dtype != "bfloat16":
        raise ValueError("authoritative Milestone 6 baseline_dtype must be bfloat16")

    if runtime_target.runtime != "transformers":
        raise ValueError("authoritative Milestone 6 runtime must be transformers")

    if runtime_target.version != _TRANSFORMERS_RUNTIME_VERSION:
        raise ValueError("authoritative Milestone 6 transformers version must be 5.17.0")

    if hardware_target.device_count != 1:
        raise ValueError(
            "authoritative Milestone 6 hardware target must contain exactly one device"
        )

    workload_profile = build_primary_workload_profile()
    generation_kwargs = build_generation_kwargs()

    return {
        "schema_version": "1",
        "run_id": normalized_run_id,
        "run_kind": "baseline",
        "purpose": "performance_baseline",
        "authoritative": True,
        "quant_version": distribution_version("huyawo-quant"),
        "created_at": created_at.astimezone(UTC).isoformat(),
        "repository_commit": normalized_repository_commit,
        "model_identity": model_identity.model_dump(mode="json"),
        "tokenizer_identity": tokenizer_identity.model_dump(mode="json"),
        "baseline_dtype": baseline_dtype,
        "runtime_target": runtime_target.model_dump(mode="json"),
        "hardware_target": hardware_target.model_dump(mode="json"),
        "environment_fingerprint_sha256": (normalized_environment_fingerprint_sha256),
        "workload_id": PRIMARY_WORKLOAD_ID,
        "workload_profile": workload_profile.model_dump(mode="json"),
        "prompt_fixture": {
            "fixture_id": normalized_prompt_fixture_id,
            "add_special_tokens": False,
            "token_count": workload_profile.prompt_tokens,
            "token_ids_sha256": normalized_prompt_token_ids_sha256,
        },
        "generation_configuration": {
            "model_eval": True,
            "torch_inference_mode": True,
            "kwargs": generation_kwargs,
            "custom_stopping_criteria": False,
            "seed": _PERFORMANCE_SEED,
            "required_generated_token_count": (workload_profile.output_tokens),
        },
        "warmup_runs": workload_profile.warmup_runs,
        "measured_runs": workload_profile.measured_runs,
        "metric_definitions": _build_metric_definitions(
            measured_runs=workload_profile.measured_runs
        ),
    }
