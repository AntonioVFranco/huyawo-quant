"""Frozen Milestone 6 performance protocol builders."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Protocol

from huyawo_quant.contracts import WorkloadProfile

PRIMARY_WORKLOAD_ID = "interactive_256_64_b1_c1_v1"
PROMPT_TOKEN_COUNT = 256
OUTPUT_TOKEN_COUNT = 64

PERFORMANCE_PROMPT_TEXT = (
    "Careful measurement makes model runtime behavior reproducible. "
    "Record the workload, hardware, runtime, generation settings, timing "
    "method, memory method, and raw observations before drawing conclusions."
)


class PromptTokenizer(Protocol):
    """Minimal tokenizer surface required to build the prompt fixture."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> object:
        """Encode text into token identifiers."""


def build_primary_workload_profile() -> WorkloadProfile:
    """Build the accepted primary Milestone 6 workload profile."""

    return WorkloadProfile(
        prompt_tokens=PROMPT_TOKEN_COUNT,
        output_tokens=OUTPUT_TOKEN_COUNT,
        batch_size=1,
        concurrency=1,
        warmup_runs=3,
        measured_runs=10,
    )


def build_performance_prompt_token_fixture(
    tokenizer: PromptTokenizer,
) -> tuple[tuple[int, ...], str]:
    """Build exactly 256 deterministic prompt token IDs and their SHA-256."""

    encoded = tokenizer.encode(
        PERFORMANCE_PROMPT_TEXT,
        add_special_tokens=False,
    )

    if not isinstance(encoded, list):
        raise TypeError("Tokenizer encode result must be a list of token IDs")

    if not encoded:
        raise ValueError("Synthetic performance prompt produced no token IDs")

    base_token_ids: list[int] = []

    for token_id in encoded:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("Prompt token IDs must be integers")
        if token_id < 0:
            raise ValueError("Prompt token IDs must be non-negative")
        base_token_ids.append(token_id)

    repeat_count = (PROMPT_TOKEN_COUNT + len(base_token_ids) - 1) // len(base_token_ids)

    token_ids = tuple((base_token_ids * repeat_count)[:PROMPT_TOKEN_COUNT])

    if len(token_ids) != PROMPT_TOKEN_COUNT:
        raise RuntimeError("Prompt fixture did not produce exactly 256 token IDs")

    canonical_bytes = json.dumps(
        list(token_ids),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")

    return token_ids, sha256(canonical_bytes).hexdigest()


def build_generation_kwargs() -> dict[str, bool | int]:
    """Build the accepted deterministic generation controls."""

    return {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "min_new_tokens": OUTPUT_TOKEN_COUNT,
        "max_new_tokens": OUTPUT_TOKEN_COUNT,
    }
