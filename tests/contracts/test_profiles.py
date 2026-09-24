"""Tests for workload and evaluation profile contracts."""

import pytest
from huyawo_quant.contracts import EvaluationProfile, WorkloadProfile
from pydantic import ValidationError


def test_workload_profile_serializes_expected_fields() -> None:
    profile = WorkloadProfile(
        prompt_tokens=512,
        output_tokens=128,
        batch_size=1,
        concurrency=4,
        warmup_runs=2,
        measured_runs=10,
    )

    assert profile.model_dump(mode="json") == {
        "prompt_tokens": 512,
        "output_tokens": 128,
        "batch_size": 1,
        "concurrency": 4,
        "warmup_runs": 2,
        "measured_runs": 10,
    }


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("prompt_tokens", 0),
        ("output_tokens", 0),
        ("batch_size", 0),
        ("concurrency", 0),
        ("measured_runs", 0),
        ("prompt_tokens", -1),
        ("output_tokens", -1),
        ("batch_size", -1),
        ("concurrency", -1),
        ("measured_runs", -1),
        ("warmup_runs", -1),
    ],
)
def test_workload_profile_rejects_invalid_numeric_values(
    field_name: str,
    value: int,
) -> None:
    payload: dict[str, object] = {
        "prompt_tokens": 512,
        "output_tokens": 128,
        "batch_size": 1,
        "concurrency": 4,
        "warmup_runs": 2,
        "measured_runs": 10,
    }
    payload[field_name] = value

    with pytest.raises(ValidationError):
        WorkloadProfile.model_validate(payload)


def test_workload_profile_uses_strict_typing() -> None:
    with pytest.raises(ValidationError):
        WorkloadProfile.model_validate(
            {
                "prompt_tokens": "512",
                "output_tokens": 128,
                "batch_size": 1,
                "concurrency": 4,
                "warmup_runs": 2,
                "measured_runs": 10,
            }
        )


def test_workload_profile_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        WorkloadProfile.model_validate(
            {
                "prompt_tokens": 512,
                "output_tokens": 128,
                "batch_size": 1,
                "concurrency": 4,
                "warmup_runs": 2,
                "measured_runs": 10,
                "runtime_flags": ["--example"],
            }
        )


def test_workload_profile_is_immutable() -> None:
    profile = WorkloadProfile(
        prompt_tokens=512,
        output_tokens=128,
        batch_size=1,
        concurrency=4,
        warmup_runs=2,
        measured_runs=10,
    )

    with pytest.raises(ValidationError):
        profile.__setattr__("concurrency", 8)


def test_evaluation_profile_serializes_expected_fields() -> None:
    profile = EvaluationProfile(
        benchmark="lm-evaluation-harness",
        benchmark_version="0.4.9",
        tasks=("hellaswag", "arc_easy"),
        metrics=("acc", "acc_norm"),
        sample_limit=100,
        seed=42,
    )

    assert profile.model_dump(mode="json") == {
        "benchmark": "lm-evaluation-harness",
        "benchmark_version": "0.4.9",
        "tasks": ["hellaswag", "arc_easy"],
        "metrics": ["acc", "acc_norm"],
        "sample_limit": 100,
        "seed": 42,
    }


def test_evaluation_profile_accepts_unlimited_samples() -> None:
    profile = EvaluationProfile(
        benchmark="lm-evaluation-harness",
        benchmark_version="0.4.9",
        tasks=("hellaswag",),
        metrics=("acc_norm",),
        seed=42,
    )

    assert profile.sample_limit is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("benchmark", ""),
        ("benchmark", "   "),
        ("benchmark_version", ""),
        ("benchmark_version", "   "),
    ],
)
def test_evaluation_profile_rejects_empty_identity_fields(
    field_name: str,
    value: str,
) -> None:
    payload: dict[str, object] = {
        "benchmark": "lm-evaluation-harness",
        "benchmark_version": "0.4.9",
        "tasks": ("hellaswag",),
        "metrics": ("acc_norm",),
        "sample_limit": 100,
        "seed": 42,
    }
    payload[field_name] = value

    with pytest.raises(ValidationError):
        EvaluationProfile.model_validate(payload)


@pytest.mark.parametrize(
    ("tasks", "metrics"),
    [
        ((), ("acc_norm",)),
        (("hellaswag",), ()),
        (("",), ("acc_norm",)),
        (("   ",), ("acc_norm",)),
        (("hellaswag",), ("",)),
        (("hellaswag",), ("   ",)),
    ],
)
def test_evaluation_profile_rejects_invalid_task_or_metric_collections(
    tasks: tuple[str, ...],
    metrics: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError):
        EvaluationProfile(
            benchmark="lm-evaluation-harness",
            benchmark_version="0.4.9",
            tasks=tasks,
            metrics=metrics,
            sample_limit=100,
            seed=42,
        )


@pytest.mark.parametrize("sample_limit", [0, -1])
def test_evaluation_profile_rejects_invalid_sample_limit(sample_limit: int) -> None:
    with pytest.raises(ValidationError):
        EvaluationProfile(
            benchmark="lm-evaluation-harness",
            benchmark_version="0.4.9",
            tasks=("hellaswag",),
            metrics=("acc_norm",),
            sample_limit=sample_limit,
            seed=42,
        )


def test_evaluation_profile_rejects_negative_seed() -> None:
    with pytest.raises(ValidationError):
        EvaluationProfile(
            benchmark="lm-evaluation-harness",
            benchmark_version="0.4.9",
            tasks=("hellaswag",),
            metrics=("acc_norm",),
            seed=-1,
        )


def test_evaluation_profile_uses_strict_tuple_typing() -> None:
    with pytest.raises(ValidationError):
        EvaluationProfile.model_validate(
            {
                "benchmark": "lm-evaluation-harness",
                "benchmark_version": "0.4.9",
                "tasks": ["hellaswag"],
                "metrics": ("acc_norm",),
                "sample_limit": 100,
                "seed": 42,
            }
        )


def test_evaluation_profile_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EvaluationProfile.model_validate(
            {
                "benchmark": "lm-evaluation-harness",
                "benchmark_version": "0.4.9",
                "tasks": ("hellaswag",),
                "metrics": ("acc_norm",),
                "sample_limit": 100,
                "seed": 42,
                "result": 0.75,
            }
        )


def test_evaluation_profile_is_immutable() -> None:
    profile = EvaluationProfile(
        benchmark="lm-evaluation-harness",
        benchmark_version="0.4.9",
        tasks=("hellaswag",),
        metrics=("acc_norm",),
        seed=42,
    )

    with pytest.raises(ValidationError):
        profile.__setattr__("seed", 43)
