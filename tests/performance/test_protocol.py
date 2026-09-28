"""Tests for the frozen Milestone 6 performance protocol."""

import json
from hashlib import sha256

import pytest
from huyawo_quant.performance import (
    OUTPUT_TOKEN_COUNT,
    PERFORMANCE_PROMPT_TEXT,
    PRIMARY_WORKLOAD_ID,
    PROMPT_TOKEN_COUNT,
    build_generation_kwargs,
    build_performance_prompt_token_fixture,
    build_primary_workload_profile,
)


class RecordingTokenizer:
    """Tokenizer double for deterministic prompt-fixture tests."""

    def __init__(self, encoded: object) -> None:
        self.encoded = encoded
        self.calls: list[tuple[str, bool]] = []

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> object:
        self.calls.append((text, add_special_tokens))
        return self.encoded


def test_primary_workload_identity_is_frozen() -> None:
    assert PRIMARY_WORKLOAD_ID == "interactive_256_64_b1_c1_v1"


def test_primary_workload_profile_matches_accepted_protocol() -> None:
    profile = build_primary_workload_profile()

    assert profile.model_dump(mode="json") == {
        "prompt_tokens": 256,
        "output_tokens": 64,
        "batch_size": 1,
        "concurrency": 1,
        "warmup_runs": 3,
        "measured_runs": 10,
    }


def test_prompt_and_output_token_counts_are_frozen() -> None:
    assert PROMPT_TOKEN_COUNT == 256
    assert OUTPUT_TOKEN_COUNT == 64


def test_generation_kwargs_match_accepted_protocol() -> None:
    assert build_generation_kwargs() == {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "min_new_tokens": 64,
        "max_new_tokens": 64,
    }


def test_generation_kwargs_are_fresh_per_call() -> None:
    first = build_generation_kwargs()
    second = build_generation_kwargs()

    first["max_new_tokens"] = 1

    assert second["max_new_tokens"] == 64


def test_prompt_fixture_uses_no_special_tokens_and_repeats_deterministically() -> None:
    tokenizer = RecordingTokenizer([11, 22, 33])

    token_ids, token_sha256 = build_performance_prompt_token_fixture(tokenizer)

    expected = tuple(([11, 22, 33] * 86)[:256])
    canonical_bytes = json.dumps(
        list(expected),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")

    assert tokenizer.calls == [(PERFORMANCE_PROMPT_TEXT, False)]
    assert token_ids == expected
    assert len(token_ids) == 256
    assert token_sha256 == sha256(canonical_bytes).hexdigest()


def test_prompt_fixture_truncates_long_token_sequences() -> None:
    tokenizer = RecordingTokenizer(list(range(300)))

    token_ids, _ = build_performance_prompt_token_fixture(tokenizer)

    assert token_ids == tuple(range(256))


def test_prompt_fixture_is_deterministic_across_calls() -> None:
    first_tokenizer = RecordingTokenizer([7, 8, 9, 10])
    second_tokenizer = RecordingTokenizer([7, 8, 9, 10])

    first = build_performance_prompt_token_fixture(first_tokenizer)
    second = build_performance_prompt_token_fixture(second_tokenizer)

    assert first == second


def test_prompt_fixture_rejects_empty_encoding() -> None:
    tokenizer = RecordingTokenizer([])

    with pytest.raises(
        ValueError,
        match="Synthetic performance prompt produced no token IDs",
    ):
        build_performance_prompt_token_fixture(tokenizer)


def test_prompt_fixture_rejects_non_list_encoding() -> None:
    tokenizer = RecordingTokenizer((1, 2, 3))

    with pytest.raises(
        TypeError,
        match="Tokenizer encode result must be a list of token IDs",
    ):
        build_performance_prompt_token_fixture(tokenizer)


@pytest.mark.parametrize(
    "encoded",
    [
        [True],
        [1.5],
        ["1"],
    ],
)
def test_prompt_fixture_rejects_non_integer_token_ids(encoded: list[object]) -> None:
    tokenizer = RecordingTokenizer(encoded)

    with pytest.raises(TypeError, match="Prompt token IDs must be integers"):
        build_performance_prompt_token_fixture(tokenizer)


def test_prompt_fixture_rejects_negative_token_ids() -> None:
    tokenizer = RecordingTokenizer([1, -1, 2])

    with pytest.raises(ValueError, match="Prompt token IDs must be non-negative"):
        build_performance_prompt_token_fixture(tokenizer)
