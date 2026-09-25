"""Canonical evidence bundle contracts."""

from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from huyawo_quant.contracts._base import ContractModel
from huyawo_quant.contracts.artifacts import QuantArtifactIdentity
from huyawo_quant.contracts.benchmarks import BenchmarkResult
from huyawo_quant.contracts.identity import ModelIdentity, TokenizerIdentity
from huyawo_quant.contracts.plans import QuantPlan
from huyawo_quant.contracts.qualification import QualificationResult
from huyawo_quant.contracts.targets import HardwareTarget, RuntimeTarget

NonEmptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class EvidenceFileReference(ContractModel):
    """Content-addressed file inside one Evidence Bundle."""

    category: Literal[
        "environment_fingerprint",
        "calibration_manifest",
        "evaluation_manifest",
        "execution_log",
        "reproducibility_metadata",
        "backend_metadata",
        "runtime_validation",
        "comparison_evidence",
        "other",
    ]
    relative_path: str = Field(min_length=1, pattern=r"\S")
    sha256: Sha256Digest
    size_bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("relative_path must use POSIX separators")

        path = PurePosixPath(value)

        if path.is_absolute():
            raise ValueError("relative_path must not be absolute")

        if value in {".", ".."} or ".." in path.parts:
            raise ValueError("relative_path must remain within the Evidence Bundle root")

        if path.as_posix() != value:
            raise ValueError("relative_path must already be normalized")

        return value


class FailureRecord(ContractModel):
    """Typed failure evidence preserved by one run."""

    stage: Literal[
        "model_load",
        "planning",
        "calibration",
        "quantization",
        "artifact_save",
        "artifact_load",
        "runtime_validation",
        "quality_evaluation",
        "performance_benchmark",
        "comparison",
        "qualification",
        "export",
        "unknown",
    ]
    category: NonEmptyString
    message: NonEmptyString


class EvidenceBundle(ContractModel):
    """Canonical local evidence envelope for one run."""

    schema_version: NonEmptyString
    quant_version: NonEmptyString
    run_id: NonEmptyString
    run_kind: Literal["baseline", "candidate"]
    execution_status: Literal["COMPLETED", "FAILED", "BLOCKED"]
    started_at: datetime
    finished_at: datetime
    model: ModelIdentity
    tokenizer: TokenizerIdentity
    hardware: HardwareTarget
    runtime: RuntimeTarget
    plan: QuantPlan | None = None
    artifact: QuantArtifactIdentity | None = None
    benchmarks: tuple[BenchmarkResult, ...] = ()
    qualification: QualificationResult | None = None
    failures: tuple[FailureRecord, ...] = ()
    evidence_files: tuple[EvidenceFileReference, ...] = ()

    @field_validator("started_at", "finished_at")
    @classmethod
    def validate_timezone_aware_datetime(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Evidence Bundle timestamps must be timezone-aware")

        return value

    @model_validator(mode="after")
    def validate_bundle_consistency(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")

        if self.run_kind == "baseline":
            if self.plan is not None:
                raise ValueError("baseline runs must not include a plan")

            if self.artifact is not None:
                raise ValueError("baseline runs must not include a quantized artifact")

            if self.qualification is not None:
                raise ValueError("baseline runs must not include qualification")
        else:
            if self.plan is None:
                raise ValueError("candidate runs require a plan")

            if self.plan.model != self.model:
                raise ValueError("candidate plan model must match bundle model")

            if self.plan.hardware != self.hardware:
                raise ValueError("candidate plan hardware must match bundle hardware")

            if self.plan.runtime != self.runtime:
                raise ValueError("candidate plan runtime must match bundle runtime")

            if self.execution_status == "COMPLETED" and self.artifact is None:
                raise ValueError("completed candidate runs require an artifact")

        if self.qualification is not None:
            if self.artifact is None:
                raise ValueError("qualification requires a candidate artifact")

            if self.qualification.candidate_artifact != self.artifact:
                raise ValueError("qualification artifact must match bundle artifact")

            if self.qualification.state == "BLOCKED" and self.execution_status != "BLOCKED":
                raise ValueError("BLOCKED qualification requires BLOCKED execution status")

        if self.execution_status == "COMPLETED":
            if self.failures:
                raise ValueError("COMPLETED runs require an empty failures tuple")
        elif not self.failures:
            raise ValueError(f"{self.execution_status} runs require at least one failure")

        evidence_paths = tuple(reference.relative_path for reference in self.evidence_files)

        if len(evidence_paths) != len(set(evidence_paths)):
            raise ValueError("evidence file paths must be unique")

        if evidence_paths != tuple(sorted(evidence_paths)):
            raise ValueError("evidence files must be ordered lexicographically by relative_path")

        return self
