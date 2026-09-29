from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from huyawo_quant.performance.timing import TokenTimingStreamer
from transformers.generation.streamers import BaseStreamer


def _clock(values: list[int]) -> Iterator[int]:
    return iter(values)


def _prompt(token_count: int = 4) -> torch.Tensor:
    return torch.arange(
        token_count,
        dtype=torch.long,
    ).reshape(1, token_count)


def _token(token_id: int = 1) -> torch.Tensor:
    return torch.tensor([token_id], dtype=torch.long)


def test_token_timing_streamer_is_explicit_base_streamer_subclass() -> None:
    assert issubclass(TokenTimingStreamer, BaseStreamer)


def test_prompt_handoff_is_ignored_and_does_not_read_clock() -> None:
    def fail_if_called() -> int:
        raise AssertionError("clock must not be read for prompt handoff")

    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=1,
        clock_ns=fail_if_called,
    )

    streamer.put(_prompt())

    assert streamer.prompt_handoff_seen is True
    assert streamer.generated_token_count == 0
    assert streamer.generated_token_timestamps_ns == ()


def test_generated_token_timestamps_ttft_and_itl_are_exact() -> None:
    timestamps = _clock([110, 130, 160])

    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=3,
        clock_ns=timestamps.__next__,
    )

    streamer.put(_prompt())
    streamer.put(_token(10))
    streamer.put(_token(11))
    streamer.put(_token(12))
    streamer.end()

    assert streamer.ended is True
    assert streamer.generated_token_count == 3
    assert streamer.generated_token_timestamps_ns == (110, 130, 160)
    assert streamer.ttft_ns == 10
    assert streamer.inter_token_intervals_ns == (20, 30)


def test_ttft_is_unavailable_before_first_generated_token() -> None:
    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=1,
        clock_ns=_clock([110]).__next__,
    )

    streamer.put(_prompt())

    with pytest.raises(RuntimeError, match="TTFT is unavailable"):
        _ = streamer.ttft_ns


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (
            torch.tensor([1, 2, 3, 4], dtype=torch.long),
            ValueError,
            "Initial prompt handoff must have shape",
        ),
        (
            torch.zeros((1, 4), dtype=torch.float32),
            ValueError,
            "must use torch.long",
        ),
        (
            "not-a-tensor",
            TypeError,
            "must provide torch.Tensor",
        ),
    ],
)
def test_invalid_prompt_handoff_fails_closed(
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=1,
        clock_ns=_clock([110]).__next__,
    )

    with pytest.raises(error_type, match=message):
        streamer.put(value)


def test_invalid_generated_token_shape_reads_clock_before_rejection() -> None:
    timestamps = _clock([110])

    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=1,
        clock_ns=timestamps.__next__,
    )

    streamer.put(_prompt())

    with pytest.raises(
        ValueError,
        match="Generated-token callback must have shape",
    ):
        streamer.put(torch.tensor([[1]], dtype=torch.long))

    with pytest.raises(StopIteration):
        next(timestamps)

    assert streamer.generated_token_count == 0
    assert streamer.generated_token_timestamps_ns == ()


def test_invalid_generated_token_type_reads_clock_before_rejection() -> None:
    timestamps = _clock([110])

    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=1,
        clock_ns=timestamps.__next__,
    )

    streamer.put(_prompt())

    with pytest.raises(
        TypeError,
        match="Streamer callbacks must provide torch.Tensor values",
    ):
        streamer.put(object())

    with pytest.raises(StopIteration):
        next(timestamps)

    assert streamer.generated_token_count == 0
    assert streamer.generated_token_timestamps_ns == ()


def test_extra_generated_token_callback_reads_clock_before_rejection() -> None:
    timestamps = _clock([110, 120])

    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=1,
        clock_ns=timestamps.__next__,
    )

    streamer.put(_prompt())
    streamer.put(_token())

    with pytest.raises(
        RuntimeError,
        match="more generated-token callbacks than expected",
    ):
        streamer.put(_token(2))

    with pytest.raises(StopIteration):
        next(timestamps)

    assert streamer.generated_token_count == 1
    assert streamer.generated_token_timestamps_ns == (110,)


def test_end_requires_exact_generated_token_count() -> None:
    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=2,
        clock_ns=_clock([110]).__next__,
    )

    streamer.put(_prompt())
    streamer.put(_token())

    with pytest.raises(
        RuntimeError,
        match="Expected 2 generated-token callbacks, observed 1",
    ):
        streamer.end()

    assert streamer.ended is True


def test_end_before_prompt_handoff_fails_closed() -> None:
    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=1,
        clock_ns=_clock([110]).__next__,
    )

    with pytest.raises(
        RuntimeError,
        match="before the initial prompt handoff",
    ):
        streamer.end()

    assert streamer.ended is True


def test_duplicate_end_and_put_after_end_fail_closed() -> None:
    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=1,
        clock_ns=_clock([110]).__next__,
    )

    streamer.put(_prompt())
    streamer.put(_token())
    streamer.end()

    with pytest.raises(
        RuntimeError,
        match=r"end\(\) may only be called once",
    ):
        streamer.end()

    with pytest.raises(
        RuntimeError,
        match=r"after end\(\)",
    ):
        streamer.put(_token(2))


@pytest.mark.parametrize(
    "invalid_timestamp",
    [
        True,
        99.5,
    ],
)
def test_clock_must_return_integer_nanoseconds(
    invalid_timestamp: object,
) -> None:
    def invalid_clock() -> int:
        return invalid_timestamp  # type: ignore[return-value]

    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=1,
        clock_ns=invalid_clock,
    )

    streamer.put(_prompt())

    with pytest.raises(
        RuntimeError,
        match="must return an integer nanosecond timestamp",
    ):
        streamer.put(_token())


def test_timestamp_cannot_precede_generation_start() -> None:
    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=1,
        clock_ns=_clock([99]).__next__,
    )

    streamer.put(_prompt())

    with pytest.raises(
        RuntimeError,
        match="precedes generation start",
    ):
        streamer.put(_token())


def test_generated_token_timestamps_must_be_monotonic() -> None:
    timestamps = _clock([120, 119])

    streamer = TokenTimingStreamer(
        generation_start_ns=100,
        expected_prompt_tokens=4,
        expected_generated_tokens=2,
        clock_ns=timestamps.__next__,
    )

    streamer.put(_prompt())
    streamer.put(_token())

    with pytest.raises(
        RuntimeError,
        match="must be monotonic",
    ):
        streamer.put(_token(2))


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("generation_start_ns", -1),
        ("expected_prompt_tokens", 0),
        ("expected_generated_tokens", 0),
    ],
)
def test_invalid_constructor_integer_boundaries_fail_closed(
    keyword: str,
    value: int,
) -> None:
    kwargs = {
        "generation_start_ns": 100,
        "expected_prompt_tokens": 4,
        "expected_generated_tokens": 1,
        keyword: value,
    }

    with pytest.raises(ValueError):
        TokenTimingStreamer(**kwargs)
