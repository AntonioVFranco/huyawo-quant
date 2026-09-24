"""Resolved quantization plan contracts."""

from typing import Annotated, Literal

from pydantic import Field

from huyawo_quant.contracts._base import ContractModel
from huyawo_quant.contracts.identity import ModelIdentity
from huyawo_quant.contracts.recipes import QuantRecipe
from huyawo_quant.contracts.targets import HardwareTarget, RuntimeTarget

NonEmptyModuleName = Annotated[str, Field(min_length=1, pattern=r"\S")]


class QuantPlan(ContractModel):
    """Resolved bridge from normalized quantization intent to one backend path."""

    model: ModelIdentity
    recipe: QuantRecipe
    hardware: HardwareTarget
    runtime: RuntimeTarget
    backend: Literal["llm_compressor", "torchao"]
    backend_version: str = Field(min_length=1, pattern=r"\S")
    resolved_modules: tuple[NonEmptyModuleName, ...] = Field(min_length=1)
