"""Pure CPU comparison of existing benchmark contracts."""

from collections.abc import Mapping
from math import isfinite
from re import fullmatch

from huyawo_quant.contracts import (
    BenchmarkMetric,
    BenchmarkResult,
    HardwareTarget,
    ModelIdentity,
    RuntimeTarget,
    TokenizerIdentity,
)

_QUALITY_BINDINGS = (
    "dataset_id",
    "dataset_revision",
    "dataset_split",
    "task_config_sha256",
    "fewshot_count",
    "scoring_semantics",
)

_PERFORMANCE_BINDINGS = (
    "workload_id",
    "prompt_token_ids_sha256",
    "generation_configuration_sha256",
    "generation_seed",
    "timing_semantics",
    "memory_semantics",
)


def _metric_key(metric: BenchmarkMetric) -> tuple[str, str]:
    return (metric.scope or "", metric.name)


def _blocked(reasons: set[str]) -> dict[str, object]:
    return {
        "status": "BLOCKED",
        "reason_codes": tuple(sorted(reasons)),
        "metric_deltas": (),
    }


def _check_bindings(
    baseline: Mapping[str, str],
    candidate: Mapping[str, str],
    required: tuple[str, ...],
    reasons: set[str],
) -> None:
    for field in required:
        left = baseline.get(field)
        right = candidate.get(field)

        if (
            not isinstance(left, str)
            or not left.strip()
            or not isinstance(right, str)
            or not right.strip()
        ):
            reasons.add(f"PROTOCOL_BINDING_MISSING:{field}")
            continue

        if field.endswith("_sha256") and (
            fullmatch(r"[0-9a-f]{64}", left) is None or fullmatch(r"[0-9a-f]{64}", right) is None
        ):
            reasons.add(f"PROTOCOL_BINDING_INVALID:{field}")
            continue

        if field == "dataset_revision" and (
            fullmatch(r"[0-9a-f]{40}", left) is None or fullmatch(r"[0-9a-f]{40}", right) is None
        ):
            reasons.add(f"PROTOCOL_BINDING_INVALID:{field}")
            continue

        if field in {"fewshot_count", "generation_seed"} and (
            not left.isdecimal() or not right.isdecimal()
        ):
            reasons.add(f"PROTOCOL_BINDING_INVALID:{field}")
            continue

        if left != right:
            reasons.add(f"PROTOCOL_BINDING_MISMATCH:{field}")


def compare_benchmarks(
    *,
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    baseline_model: ModelIdentity,
    candidate_model: ModelIdentity,
    baseline_tokenizer: TokenizerIdentity,
    candidate_tokenizer: TokenizerIdentity,
    baseline_hardware: HardwareTarget,
    candidate_hardware: HardwareTarget,
    baseline_runtime: RuntimeTarget,
    candidate_runtime: RuntimeTarget,
    baseline_bindings: Mapping[str, str],
    candidate_bindings: Mapping[str, str],
) -> dict[str, object]:
    """Return derived deltas only for strictly comparable evidence.

    Bindings must originate from independently verified run metadata.
    This result is not an EvidenceBundle or QualificationResult.
    """
    expected_types = (
        (baseline, BenchmarkResult),
        (candidate, BenchmarkResult),
        (baseline_model, ModelIdentity),
        (candidate_model, ModelIdentity),
        (baseline_tokenizer, TokenizerIdentity),
        (candidate_tokenizer, TokenizerIdentity),
        (baseline_hardware, HardwareTarget),
        (candidate_hardware, HardwareTarget),
        (baseline_runtime, RuntimeTarget),
        (candidate_runtime, RuntimeTarget),
    )

    for value, expected_type in expected_types:
        if not isinstance(value, expected_type):
            raise TypeError(f"Expected {expected_type.__name__} contract instance")

    if not isinstance(baseline_bindings, Mapping):
        raise TypeError("baseline_bindings must be a mapping")

    if not isinstance(candidate_bindings, Mapping):
        raise TypeError("candidate_bindings must be a mapping")

    reasons: set[str] = set()

    if (
        baseline_model.source,
        baseline_model.model_id,
        baseline_model.resolved_revision,
    ) != (
        candidate_model.source,
        candidate_model.model_id,
        candidate_model.resolved_revision,
    ):
        reasons.add("MODEL_IDENTITY_MISMATCH")

    if (
        baseline_tokenizer.source,
        baseline_tokenizer.tokenizer_id,
        baseline_tokenizer.resolved_revision,
    ) != (
        candidate_tokenizer.source,
        candidate_tokenizer.tokenizer_id,
        candidate_tokenizer.resolved_revision,
    ):
        reasons.add("TOKENIZER_IDENTITY_MISMATCH")

    if baseline_hardware != candidate_hardware:
        reasons.add("HARDWARE_TARGET_MISMATCH")

    if baseline_runtime != candidate_runtime:
        reasons.add("RUNTIME_TARGET_MISMATCH")

    if baseline.result_kind != candidate.result_kind:
        reasons.add("BENCHMARK_KIND_MISMATCH")

    elif baseline.result_kind == "quality":
        if baseline.evaluation_profile != candidate.evaluation_profile:
            reasons.add("EVALUATION_PROFILE_MISMATCH")

        _check_bindings(
            baseline_bindings,
            candidate_bindings,
            _QUALITY_BINDINGS,
            reasons,
        )

    else:
        if baseline.workload_profile != candidate.workload_profile:
            reasons.add("WORKLOAD_PROFILE_MISMATCH")

        _check_bindings(
            baseline_bindings,
            candidate_bindings,
            _PERFORMANCE_BINDINGS,
            reasons,
        )

    baseline_metrics = {_metric_key(metric): metric for metric in baseline.metrics}
    candidate_metrics = {_metric_key(metric): metric for metric in candidate.metrics}

    if baseline_metrics.keys() != candidate_metrics.keys():
        reasons.add("METRIC_SET_MISMATCH")

    else:
        for key in sorted(baseline_metrics):
            left = baseline_metrics[key]
            right = candidate_metrics[key]
            label = f"{key[0]}:{key[1]}"

            if left.unit != right.unit:
                reasons.add(f"METRIC_UNIT_MISMATCH:{label}")

            if left.direction != right.direction:
                reasons.add(f"METRIC_DIRECTION_MISMATCH:{label}")

            if left.aggregation != right.aggregation:
                reasons.add(f"METRIC_AGGREGATION_MISMATCH:{label}")

            if len(left.observations) != len(right.observations):
                reasons.add(f"METRIC_TRIAL_COUNT_MISMATCH:{label}")

    if reasons:
        return _blocked(reasons)

    deltas: list[dict[str, object]] = []

    for key in sorted(baseline_metrics):
        left = baseline_metrics[key]
        right = candidate_metrics[key]

        absolute_delta = right.aggregate - left.aggregate

        try:
            finite_delta = isfinite(float(absolute_delta))

            relative_delta = (
                None
                if left.aggregate == 0
                else 100.0 * float(absolute_delta) / abs(float(left.aggregate))
            )

            finite_relative = relative_delta is None or isfinite(relative_delta)

        except (OverflowError, ZeroDivisionError):
            finite_delta = False
            finite_relative = False
            relative_delta = None

        if not finite_delta or not finite_relative:
            reasons.add(f"NUMERIC_DELTA_INVALID:{key[0]}:{key[1]}")
            continue

        deltas.append(
            {
                "scope": left.scope,
                "name": left.name,
                "unit": left.unit,
                "direction": left.direction,
                "aggregation": left.aggregation,
                "observation_count": len(left.observations),
                "baseline_aggregate": left.aggregate,
                "candidate_aggregate": right.aggregate,
                "absolute_delta": absolute_delta,
                "relative_delta_percent": relative_delta,
            }
        )

    if reasons:
        return _blocked(reasons)

    return {
        "status": "COMPARABLE",
        "reason_codes": (),
        "metric_deltas": tuple(deltas),
    }
