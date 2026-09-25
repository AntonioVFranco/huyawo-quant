"""Observed environment and NVIDIA hardware fingerprint contracts."""

from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from huyawo_quant.contracts._base import ContractModel

NonEmptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]
AbsolutePathString = Annotated[str, Field(min_length=1, pattern=r"\S")]


class PythonDistributionFingerprint(ContractModel):
    """Observed installation state of one Python distribution package."""

    name: NonEmptyString
    status: Literal["INSTALLED", "ABSENT"]
    version: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_distribution_state(self) -> Self:
        if self.status == "INSTALLED":
            if self.version is None:
                raise ValueError("INSTALLED distributions require a version")
        elif self.version is not None:
            raise ValueError("ABSENT distributions must not include a version")

        return self


class NvidiaGpuFingerprint(ContractModel):
    """Observed NVIDIA GPU identity and hardware facts."""

    observed_index: int = Field(ge=0)
    name: NonEmptyString
    uuid: NonEmptyString
    pci_bus_id: NonEmptyString
    memory_total_mib: int = Field(gt=0)
    driver_version: NonEmptyString
    compute_capability_major: int = Field(ge=0)
    compute_capability_minor: int = Field(ge=0)


class CudaToolkitFingerprint(ContractModel):
    """Observed CUDA toolkit identity."""

    nvcc_path: AbsolutePathString
    release: NonEmptyString
    version: NonEmptyString

    @field_validator("nvcc_path")
    @classmethod
    def validate_nvcc_path(cls, value: str) -> str:
        if not PurePosixPath(value).is_absolute():
            raise ValueError("nvcc_path must be absolute")

        return value


class EnvironmentFingerprint(ContractModel):
    """Observed software environment and NVIDIA hardware fingerprint."""

    captured_at: datetime
    project_root: AbsolutePathString
    working_directory: AbsolutePathString
    virtual_environment: AbsolutePathString
    python_executable: AbsolutePathString
    python_version: NonEmptyString
    python_implementation: NonEmptyString
    python_prefix: AbsolutePathString
    python_base_prefix: AbsolutePathString
    platform_system: NonEmptyString
    platform_release: NonEmptyString
    platform_machine: NonEmptyString
    platform_version: NonEmptyString
    cuda_visible_devices: str | None = None
    cuda_device_order: str | None = None
    python_distributions: tuple[PythonDistributionFingerprint, ...] = Field(min_length=1)
    nvidia_smi_path: AbsolutePathString
    nvidia_gpus: tuple[NvidiaGpuFingerprint, ...] = Field(min_length=1)
    cuda_toolkit: CudaToolkitFingerprint | None = None

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")

        return value

    @field_validator(
        "project_root",
        "working_directory",
        "virtual_environment",
        "python_executable",
        "python_prefix",
        "python_base_prefix",
        "nvidia_smi_path",
    )
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        if not PurePosixPath(value).is_absolute():
            raise ValueError("environment paths must be absolute")

        return value

    @model_validator(mode="after")
    def validate_deterministic_collections(self) -> Self:
        distribution_names = tuple(distribution.name for distribution in self.python_distributions)

        if len(distribution_names) != len(set(distribution_names)):
            raise ValueError("python distribution names must be unique")

        if distribution_names != tuple(sorted(distribution_names)):
            raise ValueError("python distributions must be ordered lexicographically by name")

        gpu_uuids = tuple(gpu.uuid for gpu in self.nvidia_gpus)

        if len(gpu_uuids) != len(set(gpu_uuids)):
            raise ValueError("NVIDIA GPU UUIDs must be unique")

        if gpu_uuids != tuple(sorted(gpu_uuids)):
            raise ValueError("NVIDIA GPUs must be ordered lexicographically by UUID")

        return self
