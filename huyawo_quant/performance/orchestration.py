from __future__ import annotations

from dataclasses import dataclass

from huyawo_quant.contracts import BenchmarkResult
from huyawo_quant.performance.execution import (
    GenerationModel,
    PerformanceTrialMeasurement,
    read_torch_allocated_vram_bytes,
    run_measured_generation_trial,
)
from huyawo_quant.performance.projection import build_performance_benchmark_result
from huyawo_quant.performance.protocol import build_primary_workload_profile


@dataclass(frozen=True, slots=True)
class PerformanceOrchestrationResult:
    torch_allocated_vram_bytes: int
    measured_trials: tuple[PerformanceTrialMeasurement, ...]
    benchmark_result: BenchmarkResult


def run_performance_orchestration(
    model: GenerationModel,
    *,
    prompt_token_ids: tuple[int, ...],
    cuda_device_index: int,
) -> PerformanceOrchestrationResult:
    workload = build_primary_workload_profile()

    torch_allocated_vram_bytes = read_torch_allocated_vram_bytes(
        cuda_device_index=cuda_device_index,
    )

    for _ in range(workload.warmup_runs):
        run_measured_generation_trial(
            model,
            prompt_token_ids=prompt_token_ids,
            cuda_device_index=cuda_device_index,
        )

    measured_trials_list: list[PerformanceTrialMeasurement] = []

    for _ in range(workload.measured_runs):
        measured_trials_list.append(
            run_measured_generation_trial(
                model,
                prompt_token_ids=prompt_token_ids,
                cuda_device_index=cuda_device_index,
            )
        )

    measured_trials = tuple(measured_trials_list)

    e2e_latency_seconds = tuple(trial.e2e_latency_seconds for trial in measured_trials)
    ttft_seconds = tuple(trial.ttft_seconds for trial in measured_trials)
    inter_token_latency_seconds = tuple(
        trial.median_inter_token_latency_seconds for trial in measured_trials
    )
    output_throughput_tokens_per_second = tuple(
        trial.output_token_count / trial.e2e_latency_seconds for trial in measured_trials
    )
    torch_peak_allocated_vram_bytes = tuple(
        trial.torch_peak_allocated_vram_bytes for trial in measured_trials
    )

    benchmark_result = build_performance_benchmark_result(
        workload_profile=workload,
        e2e_latency_seconds=e2e_latency_seconds,
        ttft_seconds=ttft_seconds,
        inter_token_latency_seconds=inter_token_latency_seconds,
        output_throughput_tokens_per_second=output_throughput_tokens_per_second,
        torch_allocated_vram_bytes=torch_allocated_vram_bytes,
        torch_peak_allocated_vram_bytes=torch_peak_allocated_vram_bytes,
    )

    return PerformanceOrchestrationResult(
        torch_allocated_vram_bytes=torch_allocated_vram_bytes,
        measured_trials=measured_trials,
        benchmark_result=benchmark_result,
    )
