"""Hardware and runtime target contracts."""

from typing import Literal

from pydantic import Field

from huyawo_quant.contracts._base import ContractModel


class HardwareTarget(ContractModel):
    """Declared homogeneous hardware target."""

    vendor: Literal["nvidia"] = "nvidia"
    accelerator: Literal["cuda"] = "cuda"
    device_name: str = Field(min_length=1, pattern=r"\S")
    device_count: int = Field(gt=0)
    memory_bytes_per_device: int = Field(gt=0)
    compute_capability_major: int = Field(ge=0)
    compute_capability_minor: int = Field(ge=0)


class RuntimeTarget(ContractModel):
    """Declared runtime target."""

    runtime: Literal["vllm", "transformers"]
    version: str = Field(min_length=1, pattern=r"\S")
