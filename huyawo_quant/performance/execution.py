from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter_ns
from typing import Any, Protocol

import torch

from huyawo_quant.performance.protocol import (
    build_generation_kwargs,
    build_primary_workload_profile,
)
from huyawo_quant.performance.timing import TokenTimingStreamer

ClockNs = Callable[[], int]


class GenerationModel(Protocol):
    training: bool

    def generate(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class PerformanceTrialMeasurement:
    generation_start_ns: int
    generation_end_ns: int
    generated_token_timestamps_ns: tuple[int, ...]
    inter_token_intervals_ns: tuple[int, ...]
    torch_peak_allocated_vram_bytes: int
    output_token_count: int

    def __post_init__(self) -> None:
        if self.generation_start_ns < 0:
            raise ValueError("generation_start_ns must be non-negative")

        if self.generation_end_ns <= self.generation_start_ns:
            raise ValueError("generation_end_ns must be greater than generation_start_ns")

        if self.output_token_count <= 0:
            raise ValueError("output_token_count must be positive")

        if self.torch_peak_allocated_vram_bytes < 0:
            raise ValueError("torch_peak_allocated_vram_bytes must be non-negative")

        if len(self.generated_token_timestamps_ns) != self.output_token_count:
            raise ValueError("generated_token_timestamps_ns length must equal output_token_count")

        expected_interval_count = self.output_token_count - 1
        if len(self.inter_token_intervals_ns) != expected_interval_count:
            raise ValueError("inter_token_intervals_ns length must equal output_token_count - 1")

        if any(
            timestamp < self.generation_start_ns for timestamp in self.generated_token_timestamps_ns
        ):
            raise ValueError("Generated-token timestamps must not precede generation start")

        if any(
            timestamp > self.generation_end_ns for timestamp in self.generated_token_timestamps_ns
        ):
            raise ValueError("Generated-token timestamps must not exceed generation end")

        expected_intervals = tuple(
            current - previous
            for previous, current in zip(
                self.generated_token_timestamps_ns,
                self.generated_token_timestamps_ns[1:],
                strict=False,
            )
        )

        if self.inter_token_intervals_ns != expected_intervals:
            raise ValueError("inter_token_intervals_ns must match consecutive timestamp deltas")

    @property
    def e2e_latency_ns(self) -> int:
        return self.generation_end_ns - self.generation_start_ns

    @property
    def ttft_ns(self) -> int:
        return self.generated_token_timestamps_ns[0] - self.generation_start_ns

    @property
    def median_inter_token_latency_ns(self) -> float:
        return float(median(self.inter_token_intervals_ns))

    @property
    def e2e_latency_seconds(self) -> float:
        return self.e2e_latency_ns / 1_000_000_000

    @property
    def ttft_seconds(self) -> float:
        return self.ttft_ns / 1_000_000_000

    @property
    def median_inter_token_latency_seconds(self) -> float:
        return self.median_inter_token_latency_ns / 1_000_000_000


def _require_cuda_device(cuda_device_index: int) -> torch.device:
    if isinstance(cuda_device_index, bool) or not isinstance(
        cuda_device_index,
        int,
    ):
        raise TypeError("cuda_device_index must be an integer")

    if cuda_device_index < 0:
        raise ValueError("cuda_device_index must be non-negative")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for performance measurement")

    device_count = torch.cuda.device_count()
    if cuda_device_index >= device_count:
        raise ValueError("cuda_device_index is outside the available CUDA device range")

    return torch.device("cuda", cuda_device_index)


def _require_prompt_token_ids(
    prompt_token_ids: tuple[int, ...],
    *,
    expected_count: int,
) -> None:
    if not isinstance(prompt_token_ids, tuple):
        raise TypeError("prompt_token_ids must be a tuple")

    if len(prompt_token_ids) != expected_count:
        raise ValueError(f"prompt_token_ids must contain exactly {expected_count} tokens")

    for token_id in prompt_token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("prompt_token_ids must contain integers")

        if token_id < 0:
            raise ValueError("prompt_token_ids must be non-negative")


def _read_clock_ns(clock_ns: ClockNs, *, name: str) -> int:
    value = clock_ns()

    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} clock value must be an integer")

    if value < 0:
        raise RuntimeError(f"{name} clock value must be non-negative")

    return value


def read_torch_allocated_vram_bytes(
    *,
    cuda_device_index: int,
) -> int:
    device = _require_cuda_device(cuda_device_index)
    value = torch.cuda.memory_allocated(device)

    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError("torch.cuda.memory_allocated must return an integer")

    if value < 0:
        raise RuntimeError("torch.cuda.memory_allocated returned a negative value")

    return value


def run_measured_generation_trial(
    model: GenerationModel,
    *,
    prompt_token_ids: tuple[int, ...],
    cuda_device_index: int,
    clock_ns: ClockNs = perf_counter_ns,
) -> PerformanceTrialMeasurement:
    if model.training is not False:
        raise ValueError("model must be in evaluation mode before performance measurement")

    workload = build_primary_workload_profile()

    _require_prompt_token_ids(
        prompt_token_ids,
        expected_count=workload.prompt_tokens,
    )

    device = _require_cuda_device(cuda_device_index)

    input_ids = torch.tensor(
        (prompt_token_ids,),
        dtype=torch.long,
        device=device,
    )

    expected_input_shape = (workload.batch_size, workload.prompt_tokens)
    if tuple(input_ids.shape) != expected_input_shape:
        raise RuntimeError("Performance input tensor has an unexpected shape")

    streamer = TokenTimingStreamer(
        generation_start_ns=None,
        expected_prompt_tokens=workload.prompt_tokens,
        expected_generated_tokens=workload.output_tokens,
        clock_ns=clock_ns,
    )

    generation_kwargs = build_generation_kwargs()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    generation_start_ns = _read_clock_ns(
        clock_ns,
        name="generation_start_ns",
    )
    streamer.bind_generation_start_ns(generation_start_ns)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            streamer=streamer,
            **generation_kwargs,
        )

    torch.cuda.synchronize(device)

    generation_end_ns = _read_clock_ns(
        clock_ns,
        name="generation_end_ns",
    )

    peak_allocated_vram_bytes = torch.cuda.max_memory_allocated(device)

    if isinstance(
        peak_allocated_vram_bytes,
        bool,
    ) or not isinstance(
        peak_allocated_vram_bytes,
        int,
    ):
        raise RuntimeError("torch.cuda.max_memory_allocated must return an integer")

    if peak_allocated_vram_bytes < 0:
        raise RuntimeError("torch.cuda.max_memory_allocated returned a negative value")

    if not isinstance(generated, torch.Tensor):
        raise RuntimeError(
            "model.generate must return a torch.Tensor for the frozen performance configuration"
        )

    expected_output_shape = (
        workload.batch_size,
        workload.prompt_tokens + workload.output_tokens,
    )

    if tuple(generated.shape) != expected_output_shape:
        raise RuntimeError("model.generate did not produce exactly the frozen output token count")

    if not torch.equal(
        generated[:, : workload.prompt_tokens],
        input_ids,
    ):
        raise RuntimeError("model.generate output does not preserve the frozen prompt prefix")

    if not streamer.prompt_handoff_seen:
        raise RuntimeError("Token timing streamer did not observe the prompt handoff")

    if not streamer.ended:
        raise RuntimeError("Token timing streamer did not observe generation end")

    if streamer.generated_token_count != workload.output_tokens:
        raise RuntimeError("Token timing streamer did not observe the frozen output token count")

    if generation_end_ns < streamer.generated_token_timestamps_ns[-1]:
        raise RuntimeError("generation_end_ns precedes the final generated-token timestamp")

    return PerformanceTrialMeasurement(
        generation_start_ns=generation_start_ns,
        generation_end_ns=generation_end_ns,
        generated_token_timestamps_ns=(streamer.generated_token_timestamps_ns),
        inter_token_intervals_ns=streamer.inter_token_intervals_ns,
        torch_peak_allocated_vram_bytes=peak_allocated_vram_bytes,
        output_token_count=workload.output_tokens,
    )
