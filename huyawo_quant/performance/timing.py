from __future__ import annotations

from collections.abc import Callable
from time import perf_counter_ns
from typing import Any

import torch
from transformers.generation.streamers import BaseStreamer

ClockNs = Callable[[], int]


def _require_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


class TokenTimingStreamer(BaseStreamer):
    """Collect generated-token wall-clock timestamps without decoding tokens."""

    def __init__(
        self,
        *,
        generation_start_ns: int,
        expected_prompt_tokens: int,
        expected_generated_tokens: int,
        clock_ns: ClockNs = perf_counter_ns,
    ) -> None:
        self._generation_start_ns = _require_non_negative_int(
            generation_start_ns,
            name="generation_start_ns",
        )
        self._expected_prompt_tokens = _require_positive_int(
            expected_prompt_tokens,
            name="expected_prompt_tokens",
        )
        self._expected_generated_tokens = _require_positive_int(
            expected_generated_tokens,
            name="expected_generated_tokens",
        )

        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")

        self._clock_ns = clock_ns
        self._prompt_handoff_seen = False
        self._ended = False
        self._generated_token_timestamps_ns: list[int] = []

    @property
    def generation_start_ns(self) -> int:
        return self._generation_start_ns

    @property
    def prompt_handoff_seen(self) -> bool:
        return self._prompt_handoff_seen

    @property
    def ended(self) -> bool:
        return self._ended

    @property
    def generated_token_count(self) -> int:
        return len(self._generated_token_timestamps_ns)

    @property
    def generated_token_timestamps_ns(self) -> tuple[int, ...]:
        return tuple(self._generated_token_timestamps_ns)

    @property
    def ttft_ns(self) -> int:
        if not self._generated_token_timestamps_ns:
            raise RuntimeError("TTFT is unavailable before the first generated-token callback")

        return self._generated_token_timestamps_ns[0] - self._generation_start_ns

    @property
    def inter_token_intervals_ns(self) -> tuple[int, ...]:
        timestamps = self._generated_token_timestamps_ns
        return tuple(
            current - previous
            for previous, current in zip(
                timestamps,
                timestamps[1:],
                strict=False,
            )
        )

    def put(self, value: Any) -> None:
        if self._ended:
            raise RuntimeError("Cannot accept token callbacks after end()")

        if not self._prompt_handoff_seen:
            if not isinstance(value, torch.Tensor):
                raise TypeError("Streamer callbacks must provide torch.Tensor values")

            self._validate_prompt_handoff(value)
            self._prompt_handoff_seen = True
            return

        timestamp_ns = self._read_clock_ns()

        if not isinstance(value, torch.Tensor):
            raise TypeError("Streamer callbacks must provide torch.Tensor values")

        self._validate_generated_token_event(value)

        if self.generated_token_count >= self._expected_generated_tokens:
            raise RuntimeError("Received more generated-token callbacks than expected")

        self._generated_token_timestamps_ns.append(timestamp_ns)

    def end(self) -> None:
        if self._ended:
            raise RuntimeError("end() may only be called once")

        self._ended = True

        if not self._prompt_handoff_seen:
            raise RuntimeError("Generation ended before the initial prompt handoff")

        if self.generated_token_count != self._expected_generated_tokens:
            raise RuntimeError(
                "Expected "
                f"{self._expected_generated_tokens} generated-token callbacks, "
                f"observed {self.generated_token_count}"
            )

    def _validate_prompt_handoff(self, value: torch.Tensor) -> None:
        expected_shape = (1, self._expected_prompt_tokens)

        if tuple(value.shape) != expected_shape:
            raise ValueError(
                "Initial prompt handoff must have shape "
                f"{expected_shape}, observed {tuple(value.shape)}"
            )

        self._validate_token_tensor(value)

    def _validate_generated_token_event(
        self,
        value: torch.Tensor,
    ) -> None:
        expected_shape = (1,)

        if tuple(value.shape) != expected_shape:
            raise ValueError(
                "Generated-token callback must have shape "
                f"{expected_shape}, observed {tuple(value.shape)}"
            )

        self._validate_token_tensor(value)

    @staticmethod
    def _validate_token_tensor(value: torch.Tensor) -> None:
        if value.device.type != "cpu":
            raise ValueError("Streamer callback tensors must be on CPU")

        if value.dtype != torch.long:
            raise ValueError("Streamer callback tensors must use torch.long")

    def _read_clock_ns(self) -> int:
        value = self._clock_ns()

        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError("clock_ns must return an integer nanosecond timestamp")

        if value < self._generation_start_ns:
            raise RuntimeError("Generated-token timestamp precedes generation start")

        if self._generated_token_timestamps_ns and value < self._generated_token_timestamps_ns[-1]:
            raise RuntimeError("Generated-token timestamps must be monotonic")

        return value
