from __future__ import annotations

from statistics import median

import pytest
from huyawo_quant.contracts import BenchmarkResult
from huyawo_quant.performance.projection import (
    build_performance_benchmark_result,
)
from huyawo_quant.performance.protocol import (
    build_primary_workload_profile,
)

E2E_LATENCY_SECONDS = (
    1.00,
    1.10,
    0.90,
    1.20,
    1.05,
    1.15,
    0.95,
    1.25,
    1.30,
    1.08,
)

TTFT_SECONDS = (
    0.20,
    0.22,
    0.19,
    0.21,
    0.23,
    0.24,
    0.18,
    0.25,
    0.205,
    0.215,
)

INTER_TOKEN_LATENCY_SECONDS = (
    0.019,
    0.020,
    0.018,
    0.021,
    0.022,
    0.0195,
    0.0205,
    0.0185,
    0.0215,
    0.0198,
)

OUTPUT_THROUGHPUT_TOKENS_PER_SECOND = (
    63.0,
    64.0,
    62.0,
    65.0,
    61.0,
    66.0,
    64.5,
    63.5,
    65.5,
    62.5,
)

TORCH_ALLOCATED_VRAM_BYTES = 1_000_000_000

TORCH_PEAK_ALLOCATED_VRAM_BYTES = (
    1_200_000_000,
    1_210_000_000,
    1_190_000_000,
    1_230_000_000,
    1_220_000_000,
    1_205_000_000,
    1_215_000_000,
    1_225_000_000,
    1_235_000_000,
    1_218_000_000,
)


def build_result() -> BenchmarkResult:
    workload = build_primary_workload_profile()

    return build_performance_benchmark_result(
        workload_profile=workload,
        e2e_latency_seconds=E2E_LATENCY_SECONDS,
        ttft_seconds=TTFT_SECONDS,
        inter_token_latency_seconds=INTER_TOKEN_LATENCY_SECONDS,
        output_throughput_tokens_per_second=(OUTPUT_THROUGHPUT_TOKENS_PER_SECOND),
        torch_allocated_vram_bytes=TORCH_ALLOCATED_VRAM_BYTES,
        torch_peak_allocated_vram_bytes=(TORCH_PEAK_ALLOCATED_VRAM_BYTES),
    )


def test_projection_preserves_rfc0018_metric_order() -> None:
    result = build_result()

    identities = tuple(f"{metric.scope}/{metric.name}" for metric in result.metrics)

    assert identities == (
        "generation/e2e_latency_seconds",
        "generation/ttft_seconds",
        "generation/inter_token_latency_seconds",
        "generation/output_throughput_tokens_per_second",
        "memory/torch_allocated_vram_bytes",
        "memory/torch_peak_allocated_vram_bytes",
    )


def test_projection_preserves_trial_observations_and_aggregates() -> None:
    result = build_result()
    metrics = result.metrics

    assert metrics[0].observations == E2E_LATENCY_SECONDS
    assert metrics[0].aggregate == median(E2E_LATENCY_SECONDS)

    assert metrics[1].observations == TTFT_SECONDS
    assert metrics[1].aggregate == median(TTFT_SECONDS)

    assert metrics[2].observations == INTER_TOKEN_LATENCY_SECONDS
    assert metrics[2].aggregate == median(INTER_TOKEN_LATENCY_SECONDS)

    assert metrics[3].observations == OUTPUT_THROUGHPUT_TOKENS_PER_SECOND
    assert metrics[3].aggregate == median(OUTPUT_THROUGHPUT_TOKENS_PER_SECOND)

    assert metrics[4].observations == (TORCH_ALLOCATED_VRAM_BYTES,)
    assert metrics[4].aggregate == TORCH_ALLOCATED_VRAM_BYTES

    assert metrics[5].observations == TORCH_PEAK_ALLOCATED_VRAM_BYTES
    assert metrics[5].aggregate == max(TORCH_PEAK_ALLOCATED_VRAM_BYTES)


def test_projection_uses_performance_profile_coupling() -> None:
    result = build_result()

    assert result.result_kind == "performance"
    assert result.evaluation_profile is None
    assert result.workload_profile == build_primary_workload_profile()


def test_projection_round_trips_without_mutation() -> None:
    result = build_result()

    python_round_trip = BenchmarkResult.model_validate(result.model_dump(mode="python"))
    json_round_trip = BenchmarkResult.model_validate_json(result.model_dump_json())

    assert python_round_trip == result
    assert json_round_trip == result


def test_projection_rejects_wrong_measured_observation_count() -> None:
    workload = build_primary_workload_profile()

    with pytest.raises(
        ValueError,
        match=("generation/e2e_latency_seconds must contain exactly 10 measured observations"),
    ):
        build_performance_benchmark_result(
            workload_profile=workload,
            e2e_latency_seconds=E2E_LATENCY_SECONDS[:-1],
            ttft_seconds=TTFT_SECONDS,
            inter_token_latency_seconds=INTER_TOKEN_LATENCY_SECONDS,
            output_throughput_tokens_per_second=(OUTPUT_THROUGHPUT_TOKENS_PER_SECOND),
            torch_allocated_vram_bytes=TORCH_ALLOCATED_VRAM_BYTES,
            torch_peak_allocated_vram_bytes=(TORCH_PEAK_ALLOCATED_VRAM_BYTES),
        )


def test_projection_rejects_noncanonical_workload() -> None:
    workload = build_primary_workload_profile()
    noncanonical_workload = workload.model_copy(update={"measured_runs": 9})

    with pytest.raises(
        ValueError,
        match=(
            "workload_profile must equal the frozen interactive_256_64_b1_c1_v1 workload profile"
        ),
    ):
        build_performance_benchmark_result(
            workload_profile=noncanonical_workload,
            e2e_latency_seconds=E2E_LATENCY_SECONDS,
            ttft_seconds=TTFT_SECONDS,
            inter_token_latency_seconds=INTER_TOKEN_LATENCY_SECONDS,
            output_throughput_tokens_per_second=(OUTPUT_THROUGHPUT_TOKENS_PER_SECOND),
            torch_allocated_vram_bytes=TORCH_ALLOCATED_VRAM_BYTES,
            torch_peak_allocated_vram_bytes=(TORCH_PEAK_ALLOCATED_VRAM_BYTES),
        )
