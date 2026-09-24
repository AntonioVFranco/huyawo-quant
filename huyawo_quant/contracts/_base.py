"""Base models for Huyawo Quant contracts."""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Base class for immutable, strict Huyawo Quant contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )
