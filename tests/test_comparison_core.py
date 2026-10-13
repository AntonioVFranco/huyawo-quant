"""CPU-only synthetic tests for comparison semantics."""

from collections.abc import Mapping
from typing import cast

import pytest
from huyawo_quant.comparison import compare_benchmarks
from huyawo_quant.contracts import (
    BenchmarkMetric,
    BenchmarkResult,
    EvaluationProfile,
    HardwareTarget,
    ModelIdentity,
    RuntimeTarget,
    TokenizerIdentity,
    WorkloadProfile,
)


def _model() -> ModelIdentity:
    return ModelIdentity(
        source="huggingface",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        requested_revision="main",
        resolved_revision="7ae557604adf67be50417f59c2c2f167def9a775",
    )


def _tokenizer() -> TokenizerIdentity:
    return TokenizerIdentity(
        source="huggingface",
        tokenizer_id="Qwen/Qwen2.5-0.5B-Instruct",
        requested_revision="main",
        resolved_revision="7ae557604adf67be50417f59c2c2f167def9a775",
    )


def _hardware(device_name: str = "NVIDIA RTX A4500") -> HardwareTarget:
    return HardwareTarget(
        vendor="nvidia",
        accelerator="cuda",
        device_name=device_name,
        device_count=1,
        memory_bytes_per_device=20 * 1024**3,
        compute_capability_major=8,
        compute_capability_minor=6,
    )


def _runtime(runtime: str = "transformers") -> RuntimeTarget:
    if runtime == "transformers":
        return RuntimeTarget(runtime="transformers", version="5.17.0")

    return RuntimeTarget(runtime="vllm", version="0.30.0")


def _performance(
    observations: tuple[float, ...] = (1.0, 2.0),
    *,
    unit: str = "seconds",
    measured_runs: int = 2,
) -> BenchmarkResult:
    return BenchmarkResult(
        result_kind="performance",
        workload_profile=WorkloadProfile(
            prompt_tokens=256,
            output_tokens=64,
            batch_size=1,
            concurrency=1,
            warmup_runs=3,
            measured_runs=measured_runs,
        ),
        metrics=(
            BenchmarkMetric(
                name="e2e_latency_seconds",
                scope="generation",
                unit=unit,
                direction="lower_is_better",
                observations=observations,
                aggregation="mean",
                aggregate=sum(observations) / len(observations),
            ),
        ),
    )


def _quality(value: float) -> BenchmarkResult:
    return BenchmarkResult(
        result_kind="quality",
        evaluation_profile=EvaluationProfile(
            benchmark="lm-evaluation-harness",
            benchmark_version="0.4.13",
            tasks=("hellaswag",),
            metrics=("acc",),
            sample_limit=None,
            seed=42,
        ),
        metrics=(
            BenchmarkMetric(
                name="acc",
                scope="hellaswag",
                unit="ratio",
                direction="higher_is_better",
                observations=(value,),
                aggregation="mean",
                aggregate=value,
            ),
        ),
    )


def _bindings(kind: str) -> dict[str, str]:
    if kind == "quality":
        return {
            "dataset_id": "Rowan/hellaswag",
            "dataset_revision": "a" * 40,
            "dataset_split": "validation",
            "task_config_sha256": "b" * 64,
            "fewshot_count": "0",
            "scoring_semantics": "multiple_choice_acc_v1",
        }

    return {
        "workload_id": "interactive_256_64_b1_c1_v1",
        "prompt_token_ids_sha256": "a" * 64,
        "generation_configuration_sha256": "b" * 64,
        "generation_seed": "42",
        "timing_semantics": "cuda_synchronized_generation_v1",
        "memory_semantics": "torch_cuda_allocated_v1",
    }


def _compare(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    *,
    candidate_runtime: RuntimeTarget | None = None,
    candidate_hardware: HardwareTarget | None = None,
    baseline_bindings: Mapping[str, str] | None = None,
    candidate_bindings: Mapping[str, str] | None = None,
) -> dict[str, object]:
    defaults = _bindings(baseline.result_kind)

    return compare_benchmarks(
        baseline=baseline,
        candidate=candidate,
        baseline_model=_model(),
        candidate_model=_model(),
        baseline_tokenizer=_tokenizer(),
        candidate_tokenizer=_tokenizer(),
        baseline_hardware=_hardware(),
        candidate_hardware=(candidate_hardware if candidate_hardware is not None else _hardware()),
        baseline_runtime=_runtime(),
        candidate_runtime=(candidate_runtime if candidate_runtime is not None else _runtime()),
        baseline_bindings=(baseline_bindings if baseline_bindings is not None else defaults),
        candidate_bindings=(candidate_bindings if candidate_bindings is not None else defaults),
    )


def _reasons(result: dict[str, object]) -> tuple[str, ...]:
    return cast(tuple[str, ...], result["reason_codes"])


def _deltas(
    result: dict[str, object],
) -> tuple[dict[str, object], ...]:
    return cast(tuple[dict[str, object], ...], result["metric_deltas"])


def test_matching_performance_produces_signed_deltas() -> None:
    result = _compare(
        _performance((1.0, 2.0)),
        _performance((0.7, 0.9)),
    )

    assert result["status"] == "COMPARABLE"
    assert _reasons(result) == ()

    delta = _deltas(result)[0]

    assert delta["baseline_aggregate"] == 1.5
    assert delta["candidate_aggregate"] == pytest.approx(0.8)
    assert delta["absolute_delta"] == pytest.approx(-0.7)
    assert delta["relative_delta_percent"] == pytest.approx(-46.666666666666664)
    assert delta["observation_count"] == 2


def test_matching_quality_produces_accuracy_delta() -> None:
    result = _compare(_quality(0.4), _quality(0.5))

    assert result["status"] == "COMPARABLE"
    assert _deltas(result)[0]["absolute_delta"] == pytest.approx(0.1)
    assert _deltas(result)[0]["relative_delta_percent"] == pytest.approx(25.0)


def test_zero_baseline_disables_relative_delta() -> None:
    result = _compare(
        _performance((0.0, 0.0)),
        _performance((1.0, 1.0)),
    )

    assert result["status"] == "COMPARABLE"
    assert _deltas(result)[0]["absolute_delta"] == 1.0
    assert _deltas(result)[0]["relative_delta_percent"] is None


def test_transformers_vs_vllm_is_blocked() -> None:
    result = _compare(
        _performance(),
        _performance(),
        candidate_runtime=_runtime("vllm"),
    )

    assert result["status"] == "BLOCKED"
    assert "RUNTIME_TARGET_MISMATCH" in _reasons(result)
    assert _deltas(result) == ()


def test_hardware_mismatch_is_blocked() -> None:
    result = _compare(
        _performance(),
        _performance(),
        candidate_hardware=_hardware("NVIDIA RTX A5000"),
    )

    assert result["status"] == "BLOCKED"
    assert "HARDWARE_TARGET_MISMATCH" in _reasons(result)


def test_missing_prompt_binding_is_blocked() -> None:
    bindings = _bindings("performance")
    del bindings["prompt_token_ids_sha256"]

    result = _compare(
        _performance(),
        _performance(),
        candidate_bindings=bindings,
    )

    assert result["status"] == "BLOCKED"
    assert "PROTOCOL_BINDING_MISSING:prompt_token_ids_sha256" in _reasons(result)
    assert _deltas(result) == ()


def test_different_prompt_binding_is_blocked() -> None:
    bindings = _bindings("performance")
    bindings["prompt_token_ids_sha256"] = "c" * 64

    result = _compare(
        _performance(),
        _performance(),
        candidate_bindings=bindings,
    )

    assert "PROTOCOL_BINDING_MISMATCH:prompt_token_ids_sha256" in _reasons(result)


def test_workload_mismatch_is_blocked() -> None:
    result = _compare(
        _performance(),
        _performance(measured_runs=3),
    )

    assert "WORKLOAD_PROFILE_MISMATCH" in _reasons(result)


def test_metric_unit_mismatch_is_blocked() -> None:
    result = _compare(
        _performance(),
        _performance(unit="milliseconds"),
    )

    assert "METRIC_UNIT_MISMATCH:generation:e2e_latency_seconds" in _reasons(result)


def test_metric_trial_count_mismatch_is_blocked() -> None:
    result = _compare(
        _performance((1.0, 2.0)),
        _performance((1.0,)),
    )

    assert "METRIC_TRIAL_COUNT_MISMATCH:generation:e2e_latency_seconds" in _reasons(result)


def test_quality_dataset_revision_mismatch_is_blocked() -> None:
    bindings = _bindings("quality")
    bindings["dataset_revision"] = "c" * 40

    result = _compare(
        _quality(0.4),
        _quality(0.5),
        candidate_bindings=bindings,
    )

    assert "PROTOCOL_BINDING_MISMATCH:dataset_revision" in _reasons(result)


def test_different_benchmark_kinds_are_blocked() -> None:
    result = _compare(_performance(), _quality(0.5))

    assert result["status"] == "BLOCKED"
    assert "BENCHMARK_KIND_MISMATCH" in _reasons(result)
    assert _deltas(result) == ()

def test_cross_gpu_quality_matching_synthetic_bindings_remains_blocked() -> None:
    result = _compare(
        _quality(0.4),
        _quality(0.5),
        candidate_hardware=_hardware("NVIDIA RTX A5000"),
    )

    assert result["status"] == "BLOCKED"
    assert _reasons(result) == ("HARDWARE_TARGET_MISMATCH",)
    assert _deltas(result) == ()


def test_cross_gpu_quality_spoofed_attestation_cannot_bypass_guard() -> None:
    bindings = _bindings("quality")
    bindings.update(
        {
            "source_attestation": "CERTIFIED",
            "historical_scoring_source": "VERIFIED",
            "historical_environment": "VERIFIED",
            "cross_hardware_quality_approved": "true",
        }
    )

    result = _compare(
        _quality(0.4),
        _quality(0.5),
        candidate_hardware=_hardware("NVIDIA RTX A5000"),
        baseline_bindings=bindings,
        candidate_bindings=dict(bindings),
    )

    assert result["status"] == "BLOCKED"
    assert "HARDWARE_TARGET_MISMATCH" in _reasons(result)
    assert _deltas(result) == ()


def test_cross_gpu_quality_missing_required_provenance_is_blocked() -> None:
    bindings = _bindings("quality")
    del bindings["task_config_sha256"]
    del bindings["scoring_semantics"]

    result = _compare(
        _quality(0.4),
        _quality(0.5),
        candidate_hardware=_hardware("NVIDIA RTX A5000"),
        baseline_bindings=bindings,
        candidate_bindings=dict(bindings),
    )

    assert result["status"] == "BLOCKED"
    assert _reasons(result) == (
        "HARDWARE_TARGET_MISMATCH",
        "PROTOCOL_BINDING_MISSING:scoring_semantics",
        "PROTOCOL_BINDING_MISSING:task_config_sha256",
    )
    assert _deltas(result) == ()


def test_cross_gpu_quality_runtime_mismatch_is_also_blocked() -> None:
    result = _compare(
        _quality(0.4),
        _quality(0.5),
        candidate_hardware=_hardware("NVIDIA RTX A5000"),
        candidate_runtime=_runtime("vllm"),
    )

    assert result["status"] == "BLOCKED"
    assert "HARDWARE_TARGET_MISMATCH" in _reasons(result)
    assert "RUNTIME_TARGET_MISMATCH" in _reasons(result)
    assert _deltas(result) == ()


def test_cross_gpu_performance_ignores_spoofed_quality_approval() -> None:
    bindings = _bindings("performance")
    bindings["cross_hardware_quality_approved"] = "true"

    result = _compare(
        _performance(),
        _performance(),
        candidate_hardware=_hardware("NVIDIA RTX A5000"),
        baseline_bindings=bindings,
        candidate_bindings=dict(bindings),
    )

    assert result["status"] == "BLOCKED"
    assert "HARDWARE_TARGET_MISMATCH" in _reasons(result)
    assert _deltas(result) == ()


def test_blocked_cross_gpu_quality_does_not_mutate_source_contracts() -> None:
    baseline = _quality(0.4)
    candidate = _quality(0.5)

    before_baseline = baseline.model_dump(mode="json")
    before_candidate = candidate.model_dump(mode="json")

    result = _compare(
        baseline,
        candidate,
        candidate_hardware=_hardware("NVIDIA RTX A5000"),
    )

    assert result["status"] == "BLOCKED"
    assert _deltas(result) == ()
    assert baseline.model_dump(mode="json") == before_baseline
    assert candidate.model_dump(mode="json") == before_candidate
