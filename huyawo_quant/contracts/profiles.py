"""Workload and evaluation profile contracts."""

from typing import Annotated

from pydantic import Field

from huyawo_quant.contracts._base import ContractModel

NonEmptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]


class WorkloadProfile(ContractModel):
    """Frozen systems workload shape."""

    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    warmup_runs: int = Field(ge=0)
    measured_runs: int = Field(gt=0)


class EvaluationProfile(ContractModel):
    """Frozen quality evaluation configuration."""

    benchmark: str = Field(min_length=1, pattern=r"\S")
    benchmark_version: str = Field(min_length=1, pattern=r"\S")
    tasks: tuple[NonEmptyString, ...] = Field(min_length=1)
    metrics: tuple[NonEmptyString, ...] = Field(min_length=1)
    sample_limit: int | None = Field(default=None, gt=0)
    seed: int = Field(ge=0)
