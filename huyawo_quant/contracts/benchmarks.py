"""Benchmark result contracts."""

from math import fsum, isclose, isfinite
from statistics import fmean, median
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from huyawo_quant.contracts._base import ContractModel
from huyawo_quant.contracts.profiles import EvaluationProfile, WorkloadProfile

NonEmptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]
NumericValue = int | float


class BenchmarkMetric(ContractModel):
    """One measured benchmark metric with compact trial-level evidence."""

    name: NonEmptyString
    scope: NonEmptyString | None = None
    unit: NonEmptyString
    direction: Literal["higher_is_better", "lower_is_better", "neutral"]
    observations: tuple[NumericValue, ...] = Field(min_length=1)
    aggregation: Literal["mean", "median", "min", "max", "sum", "last"]
    aggregate: NumericValue

    @field_validator("observations")
    @classmethod
    def validate_observations(
        cls,
        value: tuple[NumericValue, ...],
    ) -> tuple[NumericValue, ...]:
        for observation in value:
            if isinstance(observation, float) and not isfinite(observation):
                raise ValueError("observations must contain only finite numbers")

        return value

    @field_validator("aggregate")
    @classmethod
    def validate_aggregate_is_finite(
        cls,
        value: NumericValue,
    ) -> NumericValue:
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("aggregate must be finite")

        return value

    @model_validator(mode="after")
    def validate_aggregate_consistency(self) -> Self:
        values = self.observations

        if self.aggregation == "mean":
            expected = fmean(values)
        elif self.aggregation == "median":
            expected = median(values)
        elif self.aggregation == "min":
            expected = min(values)
        elif self.aggregation == "max":
            expected = max(values)
        elif self.aggregation == "sum":
            expected = fsum(values)
        else:
            expected = values[-1]

        if not isclose(
            float(self.aggregate),
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("aggregate must match the declared aggregation of observations")

        return self


class BenchmarkResult(ContractModel):
    """Measured benchmark results under one stable protocol profile."""

    result_kind: Literal["quality", "performance"]
    evaluation_profile: EvaluationProfile | None = None
    workload_profile: WorkloadProfile | None = None
    metrics: tuple[BenchmarkMetric, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result_protocol(self) -> Self:
        if self.result_kind == "quality":
            if self.evaluation_profile is None:
                raise ValueError("quality results require an evaluation_profile")
            if self.workload_profile is not None:
                raise ValueError("quality results must not include a workload_profile")
        else:
            if self.workload_profile is None:
                raise ValueError("performance results require a workload_profile")
            if self.evaluation_profile is not None:
                raise ValueError("performance results must not include an evaluation_profile")

        identities = tuple((metric.scope or "", metric.name) for metric in self.metrics)

        if len(identities) != len(set(identities)):
            raise ValueError("benchmark metric identities must be unique")

        if identities != tuple(sorted(identities)):
            raise ValueError(
                "benchmark metrics must be ordered lexicographically by scope and name"
            )

        return self
