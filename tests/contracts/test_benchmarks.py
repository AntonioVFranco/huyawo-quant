"""Tests for benchmark result contracts."""

import math

import pytest
from huyawo_quant.contracts import (
    BenchmarkMetric,
    BenchmarkResult,
    EvaluationProfile,
    WorkloadProfile,
)
from pydantic import ValidationError


def make_evaluation_profile() -> EvaluationProfile:
    return EvaluationProfile(
        benchmark="lm-evaluation-harness",
        benchmark_version="test-version",
        tasks=("hellaswag",),
        metrics=("accuracy",),
        sample_limit=100,
        seed=42,
    )


def make_workload_profile() -> WorkloadProfile:
    return WorkloadProfile(
        prompt_tokens=512,
        output_tokens=128,
        batch_size=1,
        concurrency=1,
        warmup_runs=2,
        measured_runs=5,
    )


def make_metric(
    *,
    name: str = "accuracy",
    scope: str | None = "hellaswag",
    unit: str = "ratio",
    direction: str = "higher_is_better",
    observations: tuple[int | float, ...] = (0.70, 0.72, 0.71),
    aggregation: str = "mean",
    aggregate: int | float = 0.71,
) -> BenchmarkMetric:
    payload: dict[str, object] = {
        "name": name,
        "scope": scope,
        "unit": unit,
        "direction": direction,
        "observations": observations,
        "aggregation": aggregation,
        "aggregate": aggregate,
    }

    return BenchmarkMetric.model_validate(payload)


def make_quality_result() -> BenchmarkResult:
    return BenchmarkResult(
        result_kind="quality",
        evaluation_profile=make_evaluation_profile(),
        metrics=(make_metric(),),
    )


def test_benchmark_metric_serializes_expected_fields() -> None:
    metric = make_metric()

    assert metric.model_dump(mode="json") == {
        "name": "accuracy",
        "scope": "hellaswag",
        "unit": "ratio",
        "direction": "higher_is_better",
        "observations": [0.70, 0.72, 0.71],
        "aggregation": "mean",
        "aggregate": 0.71,
    }


def test_quality_benchmark_result_serializes_expected_fields() -> None:
    result = make_quality_result()

    assert result.model_dump(mode="json") == {
        "result_kind": "quality",
        "evaluation_profile": {
            "benchmark": "lm-evaluation-harness",
            "benchmark_version": "test-version",
            "tasks": ["hellaswag"],
            "metrics": ["accuracy"],
            "sample_limit": 100,
            "seed": 42,
        },
        "workload_profile": None,
        "metrics": [
            {
                "name": "accuracy",
                "scope": "hellaswag",
                "unit": "ratio",
                "direction": "higher_is_better",
                "observations": [0.70, 0.72, 0.71],
                "aggregation": "mean",
                "aggregate": 0.71,
            }
        ],
    }


def test_performance_result_accepts_workload_profile() -> None:
    result = BenchmarkResult(
        result_kind="performance",
        workload_profile=make_workload_profile(),
        metrics=(
            make_metric(
                name="generation_throughput",
                scope=None,
                unit="tokens_per_second",
                direction="higher_is_better",
                observations=(100.0, 110.0, 120.0),
                aggregation="mean",
                aggregate=110.0,
            ),
        ),
    )

    assert result.workload_profile == make_workload_profile()
    assert result.evaluation_profile is None


@pytest.mark.parametrize(
    "direction",
    [
        "higher_is_better",
        "lower_is_better",
        "neutral",
    ],
)
def test_benchmark_metric_accepts_directions(direction: str) -> None:
    payload = make_metric().model_dump()
    payload["direction"] = direction

    metric = BenchmarkMetric.model_validate(payload)

    assert metric.direction == direction


@pytest.mark.parametrize(
    ("aggregation", "observations", "aggregate"),
    [
        ("mean", (1.0, 2.0, 3.0), 2.0),
        ("median", (1.0, 3.0, 2.0), 2.0),
        ("min", (1.0, 2.0, 3.0), 1.0),
        ("max", (1.0, 2.0, 3.0), 3.0),
        ("sum", (1.0, 2.0, 3.0), 6.0),
        ("last", (1.0, 2.0, 3.0), 3.0),
    ],
)
def test_benchmark_metric_accepts_supported_aggregations(
    aggregation: str,
    observations: tuple[float, ...],
    aggregate: float,
) -> None:
    metric = make_metric(
        observations=observations,
        aggregation=aggregation,
        aggregate=aggregate,
    )

    assert metric.aggregation == aggregation
    assert metric.aggregate == aggregate


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", ""),
        ("name", "   "),
        ("unit", ""),
        ("unit", "   "),
        ("scope", ""),
        ("scope", "   "),
    ],
)
def test_benchmark_metric_rejects_empty_text_fields(
    field_name: str,
    value: str,
) -> None:
    payload = make_metric().model_dump()
    payload[field_name] = value

    with pytest.raises(ValidationError):
        BenchmarkMetric.model_validate(payload)


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_benchmark_metric_rejects_non_finite_observations(
    value: float,
) -> None:
    payload = make_metric().model_dump()
    payload["observations"] = (0.5, value)

    with pytest.raises(ValidationError):
        BenchmarkMetric.model_validate(payload)


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_benchmark_metric_rejects_non_finite_aggregate(
    value: float,
) -> None:
    payload = make_metric().model_dump()
    payload["aggregate"] = value

    with pytest.raises(ValidationError):
        BenchmarkMetric.model_validate(payload)


def test_benchmark_metric_rejects_empty_observations() -> None:
    payload = make_metric().model_dump()
    payload["observations"] = ()

    with pytest.raises(ValidationError):
        BenchmarkMetric.model_validate(payload)


def test_benchmark_metric_uses_strict_tuple_typing() -> None:
    payload = make_metric().model_dump()
    payload["observations"] = [0.70, 0.72, 0.71]

    with pytest.raises(ValidationError):
        BenchmarkMetric.model_validate(payload)


def test_benchmark_metric_rejects_inconsistent_aggregate() -> None:
    payload = make_metric().model_dump()
    payload["aggregate"] = 0.99

    with pytest.raises(ValidationError):
        BenchmarkMetric.model_validate(payload)


def test_quality_result_requires_evaluation_profile() -> None:
    with pytest.raises(ValidationError):
        BenchmarkResult(
            result_kind="quality",
            metrics=(make_metric(),),
        )


def test_quality_result_rejects_workload_profile() -> None:
    with pytest.raises(ValidationError):
        BenchmarkResult(
            result_kind="quality",
            evaluation_profile=make_evaluation_profile(),
            workload_profile=make_workload_profile(),
            metrics=(make_metric(),),
        )


def test_performance_result_requires_workload_profile() -> None:
    metric = make_metric(
        name="latency",
        scope=None,
        unit="seconds",
        direction="lower_is_better",
    )

    with pytest.raises(ValidationError):
        BenchmarkResult(
            result_kind="performance",
            metrics=(metric,),
        )


def test_performance_result_rejects_evaluation_profile() -> None:
    metric = make_metric(
        name="latency",
        scope=None,
        unit="seconds",
        direction="lower_is_better",
    )

    with pytest.raises(ValidationError):
        BenchmarkResult(
            result_kind="performance",
            evaluation_profile=make_evaluation_profile(),
            workload_profile=make_workload_profile(),
            metrics=(metric,),
        )


def test_benchmark_result_rejects_duplicate_metric_identities() -> None:
    with pytest.raises(ValidationError):
        BenchmarkResult(
            result_kind="quality",
            evaluation_profile=make_evaluation_profile(),
            metrics=(
                make_metric(),
                make_metric(
                    observations=(0.73,),
                    aggregation="last",
                    aggregate=0.73,
                ),
            ),
        )


def test_benchmark_result_preserves_protocol_defined_metric_order() -> None:
    metrics = (
        make_metric(name="z_metric", scope="hellaswag"),
        make_metric(name="a_metric", scope="hellaswag"),
    )

    result = BenchmarkResult(
        result_kind="quality", evaluation_profile=make_evaluation_profile(), metrics=metrics
    )

    assert result.metrics == metrics

    python_round_trip = BenchmarkResult.model_validate(result.model_dump(mode="python"))
    assert python_round_trip == result

    json_round_trip = BenchmarkResult.model_validate_json(result.model_dump_json())
    assert json_round_trip == result


def test_benchmark_result_accepts_sorted_metrics() -> None:
    result = BenchmarkResult(
        result_kind="quality",
        evaluation_profile=make_evaluation_profile(),
        metrics=(
            make_metric(
                name="accuracy",
                scope="hellaswag",
            ),
            make_metric(
                name="normalized_accuracy",
                scope="hellaswag",
            ),
        ),
    )

    assert len(result.metrics) == 2


def test_benchmark_result_rejects_empty_metrics() -> None:
    with pytest.raises(ValidationError):
        BenchmarkResult(
            result_kind="quality",
            evaluation_profile=make_evaluation_profile(),
            metrics=(),
        )


def test_benchmark_result_uses_strict_tuple_typing() -> None:
    payload: dict[str, object] = make_quality_result().model_dump()
    payload["metrics"] = [make_metric()]

    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)


def test_benchmark_metric_rejects_extra_fields() -> None:
    payload = make_metric().model_dump()
    payload["provider_metadata"] = {"device": "gpu"}

    with pytest.raises(ValidationError):
        BenchmarkMetric.model_validate(payload)


def test_benchmark_result_rejects_extra_fields() -> None:
    payload = make_quality_result().model_dump()
    payload["qualification_state"] = "ELIGIBLE"

    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)


def test_benchmark_metric_is_immutable() -> None:
    metric = make_metric()

    with pytest.raises(ValidationError):
        metric.__setattr__("aggregate", 0.80)


def test_benchmark_result_is_immutable() -> None:
    result = make_quality_result()

    with pytest.raises(ValidationError):
        result.__setattr__("result_kind", "performance")
