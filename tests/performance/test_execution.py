from __future__ import annotations

from typing import Any

import pytest
import torch
from huyawo_quant.performance import execution
from huyawo_quant.performance.execution import (
    read_torch_allocated_vram_bytes,
    run_measured_generation_trial,
)
from huyawo_quant.performance.timing import TokenTimingStreamer

_REAL_TENSOR = torch.tensor


class _FakeGenerationModel:
    training = False

    def __init__(self, *, output_width: int = 320) -> None:
        self.output_width = output_width
        self.generation_kwargs: dict[str, Any] | None = None

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        streamer: TokenTimingStreamer,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.generation_kwargs = dict(kwargs)

        streamer.put(input_ids.cpu())

        for token_id in range(256, 320):
            streamer.put(
                _REAL_TENSOR(
                    [token_id],
                    dtype=torch.long,
                )
            )

        streamer.end()

        return _REAL_TENSOR(
            [list(range(self.output_width))],
            dtype=torch.long,
        )


def _patch_cuda_for_trial(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    peak_bytes: int = 123456,
) -> None:
    monkeypatch.setattr(
        execution.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        execution.torch.cuda,
        "device_count",
        lambda: 1,
    )

    def synchronize(device: object = None) -> None:
        events.append(f"sync:{device}")

    def reset_peak_memory_stats(device: object = None) -> None:
        events.append(f"reset:{device}")

    def max_memory_allocated(device: object = None) -> int:
        events.append(f"peak:{device}")
        return peak_bytes

    monkeypatch.setattr(
        execution.torch.cuda,
        "synchronize",
        synchronize,
    )
    monkeypatch.setattr(
        execution.torch.cuda,
        "reset_peak_memory_stats",
        reset_peak_memory_stats,
    )
    monkeypatch.setattr(
        execution.torch.cuda,
        "max_memory_allocated",
        max_memory_allocated,
    )


def test_measured_trial_preserves_protocol_order_and_raw_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    _patch_cuda_for_trial(monkeypatch, events)

    def fake_tensor(
        data: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        events.append(f"tensor:{device}")
        return _REAL_TENSOR(data, dtype=dtype)

    monkeypatch.setattr(
        execution.torch,
        "tensor",
        fake_tensor,
    )

    clock_values = iter(
        (
            1_000_000,
            *(1_100_000 + index * 10_000 for index in range(64)),
            2_000_000,
        )
    )

    def clock_ns() -> int:
        value = next(clock_values)
        events.append(f"clock:{value}")
        return value

    model = _FakeGenerationModel()

    measurement = run_measured_generation_trial(
        model,
        prompt_token_ids=tuple(range(256)),
        cuda_device_index=0,
        clock_ns=clock_ns,
    )

    assert events[:6] == [
        "tensor:cuda:0",
        "sync:cuda:0",
        "reset:cuda:0",
        "sync:cuda:0",
        "clock:1000000",
        "clock:1100000",
    ]
    assert events[-3:] == [
        "sync:cuda:0",
        "clock:2000000",
        "peak:cuda:0",
    ]

    assert model.generation_kwargs == {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "min_new_tokens": 64,
        "max_new_tokens": 64,
    }

    assert measurement.generation_start_ns == 1_000_000
    assert measurement.generation_end_ns == 2_000_000
    assert measurement.output_token_count == 64
    assert measurement.torch_peak_allocated_vram_bytes == 123456

    assert measurement.generated_token_timestamps_ns == tuple(
        1_100_000 + index * 10_000 for index in range(64)
    )
    assert measurement.inter_token_intervals_ns == (10_000,) * 63

    assert measurement.e2e_latency_ns == 1_000_000
    assert measurement.ttft_ns == 100_000
    assert measurement.median_inter_token_latency_ns == 10_000.0

    assert measurement.e2e_latency_seconds == pytest.approx(0.001)
    assert measurement.ttft_seconds == pytest.approx(0.0001)
    assert measurement.median_inter_token_latency_seconds == pytest.approx(0.00001)


def test_measured_trial_binds_start_before_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    _patch_cuda_for_trial(monkeypatch, events)

    def fake_tensor(
        data: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return _REAL_TENSOR(data, dtype=dtype)

    monkeypatch.setattr(
        execution.torch,
        "tensor",
        fake_tensor,
    )

    clock_values = iter(
        (
            10,
            *(20 + index for index in range(64)),
            100,
        )
    )

    class BindingProbeModel(_FakeGenerationModel):
        def generate(
            self,
            *,
            input_ids: torch.Tensor,
            streamer: TokenTimingStreamer,
            **kwargs: Any,
        ) -> torch.Tensor:
            assert streamer.generation_start_ns == 10
            return super().generate(
                input_ids=input_ids,
                streamer=streamer,
                **kwargs,
            )

    run_measured_generation_trial(
        BindingProbeModel(),
        prompt_token_ids=tuple(range(256)),
        cuda_device_index=0,
        clock_ns=lambda: next(clock_values),
    )


def test_measured_trial_rejects_training_model_before_cuda_access() -> None:
    class TrainingModel:
        training = True

    with pytest.raises(
        ValueError,
        match="model must be in evaluation mode",
    ):
        run_measured_generation_trial(
            TrainingModel(),
            prompt_token_ids=tuple(range(256)),
            cuda_device_index=0,
        )


def test_measured_trial_rejects_wrong_prompt_length_before_cuda_access() -> None:
    with pytest.raises(
        ValueError,
        match="must contain exactly 256 tokens",
    ):
        run_measured_generation_trial(
            _FakeGenerationModel(),
            prompt_token_ids=tuple(range(255)),
            cuda_device_index=0,
        )


def test_measured_trial_rejects_wrong_generated_output_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    _patch_cuda_for_trial(monkeypatch, events)

    def fake_tensor(
        data: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return _REAL_TENSOR(data, dtype=dtype)

    monkeypatch.setattr(
        execution.torch,
        "tensor",
        fake_tensor,
    )

    clock_values = iter(
        (
            10,
            *(20 + index for index in range(64)),
            100,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="did not produce exactly the frozen output token count",
    ):
        run_measured_generation_trial(
            _FakeGenerationModel(output_width=319),
            prompt_token_ids=tuple(range(256)),
            cuda_device_index=0,
            clock_ns=lambda: next(clock_values),
        )


def test_read_torch_allocated_vram_bytes_uses_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_devices: list[str] = []

    monkeypatch.setattr(
        execution.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        execution.torch.cuda,
        "device_count",
        lambda: 1,
    )

    def memory_allocated(device: object = None) -> int:
        seen_devices.append(str(device))
        return 4096

    monkeypatch.setattr(
        execution.torch.cuda,
        "memory_allocated",
        memory_allocated,
    )

    assert read_torch_allocated_vram_bytes(cuda_device_index=0) == 4096
    assert seen_devices == ["cuda:0"]
