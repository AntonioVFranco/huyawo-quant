from __future__ import annotations

from statistics import median

from huyawo_quant.contracts import BenchmarkMetric, BenchmarkResult, WorkloadProfile

from .protocol import build_primary_workload_profile


def _require_measured_observation_count(
    *,
    metric_identity: str,
    observations: tuple[float, ...] | tuple[int, ...],
    expected_count: int,
) -> None:
    if len(observations) != expected_count:
        raise ValueError(
            f"{metric_identity} must contain exactly {expected_count} measured observations"
        )


def build_performance_benchmark_result(
    *,
    workload_profile: WorkloadProfile,
    e2e_latency_seconds: tuple[float, ...],
    ttft_seconds: tuple[float, ...],
    inter_token_latency_seconds: tuple[float, ...],
    output_throughput_tokens_per_second: tuple[float, ...],
    torch_allocated_vram_bytes: int,
    torch_peak_allocated_vram_bytes: tuple[int, ...],
) -> BenchmarkResult:
    expected_workload = build_primary_workload_profile()

    if workload_profile != expected_workload:
        raise ValueError(
            "workload_profile must equal the frozen interactive_256_64_b1_c1_v1 workload profile"
        )

    measured_runs = workload_profile.measured_runs

    measured_observations = (
        (
            "generation/e2e_latency_seconds",
            e2e_latency_seconds,
        ),
        (
            "generation/ttft_seconds",
            ttft_seconds,
        ),
        (
            "generation/inter_token_latency_seconds",
            inter_token_latency_seconds,
        ),
        (
            "generation/output_throughput_tokens_per_second",
            output_throughput_tokens_per_second,
        ),
        (
            "memory/torch_peak_allocated_vram_bytes",
            torch_peak_allocated_vram_bytes,
        ),
    )

    for metric_identity, observations in measured_observations:
        _require_measured_observation_count(
            metric_identity=metric_identity,
            observations=observations,
            expected_count=measured_runs,
        )

    metrics = (
        BenchmarkMetric(
            scope="generation",
            name="e2e_latency_seconds",
            unit="seconds",
            direction="lower_is_better",
            observations=e2e_latency_seconds,
            aggregation="median",
            aggregate=float(median(e2e_latency_seconds)),
        ),
        BenchmarkMetric(
            scope="generation",
            name="ttft_seconds",
            unit="seconds",
            direction="lower_is_better",
            observations=ttft_seconds,
            aggregation="median",
            aggregate=float(median(ttft_seconds)),
        ),
        BenchmarkMetric(
            scope="generation",
            name="inter_token_latency_seconds",
            unit="seconds",
            direction="lower_is_better",
            observations=inter_token_latency_seconds,
            aggregation="median",
            aggregate=float(median(inter_token_latency_seconds)),
        ),
        BenchmarkMetric(
            scope="generation",
            name="output_throughput_tokens_per_second",
            unit="tokens/second",
            direction="higher_is_better",
            observations=output_throughput_tokens_per_second,
            aggregation="median",
            aggregate=float(median(output_throughput_tokens_per_second)),
        ),
        BenchmarkMetric(
            scope="memory",
            name="torch_allocated_vram_bytes",
            unit="bytes",
            direction="lower_is_better",
            observations=(torch_allocated_vram_bytes,),
            aggregation="last",
            aggregate=torch_allocated_vram_bytes,
        ),
        BenchmarkMetric(
            scope="memory",
            name="torch_peak_allocated_vram_bytes",
            unit="bytes",
            direction="lower_is_better",
            observations=torch_peak_allocated_vram_bytes,
            aggregation="max",
            aggregate=max(torch_peak_allocated_vram_bytes),
        ),
    )

    return BenchmarkResult(
        result_kind="performance",
        evaluation_profile=None,
        workload_profile=workload_profile,
        metrics=metrics,
    )
