"""Performance baseline protocol surface."""

from huyawo_quant.performance.protocol import (
    OUTPUT_TOKEN_COUNT,
    PERFORMANCE_PROMPT_TEXT,
    PRIMARY_WORKLOAD_ID,
    PROMPT_TOKEN_COUNT,
    PromptTokenizer,
    build_generation_kwargs,
    build_performance_prompt_token_fixture,
    build_primary_workload_profile,
)

__all__ = [
    "OUTPUT_TOKEN_COUNT",
    "PERFORMANCE_PROMPT_TEXT",
    "PRIMARY_WORKLOAD_ID",
    "PROMPT_TOKEN_COUNT",
    "PromptTokenizer",
    "build_generation_kwargs",
    "build_performance_prompt_token_fixture",
    "build_primary_workload_profile",
]
