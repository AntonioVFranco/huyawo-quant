from __future__ import annotations

from typing import Any

import pytest
from huyawo_quant.performance import orchestration
from huyawo_quant.performance.execution import PerformanceTrialMeasurement

PROMPT_TOKEN_IDS = tuple(range(256))


def _measurement(index: int) -> PerformanceTrialMeasurement:
    generation_start_ns = 1_000_000_000 + (index * 10_000_000_000)
    first_token_delay_ns = 100_000_000 + (index * 1_000_000)
    inter_token_interval_ns = 10_000_000 + (index * 1_000)

    generated_token_timestamps_ns = tuple(
        generation_start_ns + first_token_delay_ns + (token_index * inter_token_interval_ns)
        for token_index in range(64)
    )

    generation_end_ns = generated_token_timestamps_ns[-1] + 200_000_000

    inter_token_intervals_ns = tuple(
        current - previous
        for previous, current in zip(
            generated_token_timestamps_ns,
            generated_token_timestamps_ns[1:],
            strict=False,
        )
    )

    return PerformanceTrialMeasurement(
        generation_start_ns=generation_start_ns,
        generation_end_ns=generation_end_ns,
        generated_token_timestamps_ns=generated_token_timestamps_ns,
        inter_token_intervals_ns=inter_token_intervals_ns,
        torch_peak_allocated_vram_bytes=1_000_000 + index,
        output_token_count=64,
    )


def test_orchestration_preserves_protocol_order_and_projection_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = object()
    resident_vram_bytes = 777_000_000

    warmup_trials = tuple(_measurement(index) for index in range(3))
    measured_trials = tuple(_measurement(index) for index in range(3, 13))
    all_trials = warmup_trials + measured_trials

    events: list[str] = []
    trial_index = 0
    captured_projection_kwargs: dict[str, Any] = {}

    original_projection = orchestration.build_performance_benchmark_result

    def fake_resident_reader(*, cuda_device_index: int) -> int:
        events.append("resident")
        assert cuda_device_index == 0
        return resident_vram_bytes

    def fake_trial_runner(
        model: object,
        *,
        prompt_token_ids: tuple[int, ...],
        cuda_device_index: int,
    ) -> PerformanceTrialMeasurement:
        nonlocal trial_index

        assert model is fake_model
        assert prompt_token_ids == PROMPT_TOKEN_IDS
        assert cuda_device_index == 0

        events.append(f"trial:{trial_index}")

        measurement = all_trials[trial_index]
        trial_index += 1
        return measurement

    def recording_projection(**kwargs: Any) -> Any:
        events.append("projection")
        captured_projection_kwargs.update(kwargs)
        return original_projection(**kwargs)

    monkeypatch.setattr(
        orchestration,
        "read_torch_allocated_vram_bytes",
        fake_resident_reader,
    )
    monkeypatch.setattr(
        orchestration,
        "run_measured_generation_trial",
        fake_trial_runner,
    )
    monkeypatch.setattr(
        orchestration,
        "build_performance_benchmark_result",
        recording_projection,
    )

    result = orchestration.run_performance_orchestration(
        fake_model,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        cuda_device_index=0,
    )

    assert events == [
        "resident",
        *(f"trial:{index}" for index in range(13)),
        "projection",
    ]

    assert trial_index == 13
    assert result.torch_allocated_vram_bytes == resident_vram_bytes
    assert result.measured_trials == measured_trials

    assert captured_projection_kwargs["workload_profile"] == (
        orchestration.build_primary_workload_profile()
    )
    assert captured_projection_kwargs["e2e_latency_seconds"] == tuple(
        trial.e2e_latency_seconds for trial in measured_trials
    )
    assert captured_projection_kwargs["ttft_seconds"] == tuple(
        trial.ttft_seconds for trial in measured_trials
    )
    assert captured_projection_kwargs["inter_token_latency_seconds"] == tuple(
        trial.median_inter_token_latency_seconds for trial in measured_trials
    )
    assert captured_projection_kwargs["output_throughput_tokens_per_second"] == tuple(
        trial.output_token_count / trial.e2e_latency_seconds for trial in measured_trials
    )
    assert captured_projection_kwargs["torch_allocated_vram_bytes"] == resident_vram_bytes
    assert captured_projection_kwargs["torch_peak_allocated_vram_bytes"] == tuple(
        trial.torch_peak_allocated_vram_bytes for trial in measured_trials
    )

    metric_by_name = {metric.name: metric for metric in result.benchmark_result.metrics}

    assert metric_by_name["e2e_latency_seconds"].observations == tuple(
        trial.e2e_latency_seconds for trial in measured_trials
    )
    assert metric_by_name["ttft_seconds"].observations == tuple(
        trial.ttft_seconds for trial in measured_trials
    )
    assert metric_by_name["inter_token_latency_seconds"].observations == tuple(
        trial.median_inter_token_latency_seconds for trial in measured_trials
    )
    assert metric_by_name["output_throughput_tokens_per_second"].observations == tuple(
        trial.output_token_count / trial.e2e_latency_seconds for trial in measured_trials
    )
    assert metric_by_name["torch_allocated_vram_bytes"].observations == (resident_vram_bytes,)
    assert metric_by_name["torch_peak_allocated_vram_bytes"].observations == tuple(
        trial.torch_peak_allocated_vram_bytes for trial in measured_trials
    )


def test_orchestration_propagates_first_trial_failure_without_retry_or_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = object()

    events: list[str] = []
    trial_calls = 0
    projection_calls = 0

    def fake_resident_reader(*, cuda_device_index: int) -> int:
        events.append("resident")
        assert cuda_device_index == 0
        return 123

    def failing_trial_runner(
        model: object,
        *,
        prompt_token_ids: tuple[int, ...],
        cuda_device_index: int,
    ) -> PerformanceTrialMeasurement:
        nonlocal trial_calls

        assert model is fake_model
        assert prompt_token_ids == PROMPT_TOKEN_IDS
        assert cuda_device_index == 0

        events.append(f"trial:{trial_calls}")

        current_call = trial_calls
        trial_calls += 1

        if current_call == 4:
            raise RuntimeError("synthetic measured trial failure")

        return _measurement(current_call)

    def unexpected_projection(**kwargs: Any) -> Any:
        nonlocal projection_calls
        projection_calls += 1
        pytest.fail(f"projection must not run after failure: {kwargs}")

    monkeypatch.setattr(
        orchestration,
        "read_torch_allocated_vram_bytes",
        fake_resident_reader,
    )
    monkeypatch.setattr(
        orchestration,
        "run_measured_generation_trial",
        failing_trial_runner,
    )
    monkeypatch.setattr(
        orchestration,
        "build_performance_benchmark_result",
        unexpected_projection,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic measured trial failure",
    ):
        orchestration.run_performance_orchestration(
            fake_model,
            prompt_token_ids=PROMPT_TOKEN_IDS,
            cuda_device_index=0,
        )

    assert events == [
        "resident",
        "trial:0",
        "trial:1",
        "trial:2",
        "trial:3",
        "trial:4",
    ]
    assert trial_calls == 5
    assert projection_calls == 0
