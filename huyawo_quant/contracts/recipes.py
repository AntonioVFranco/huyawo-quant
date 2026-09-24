"""Normalized quantization recipe contracts."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from huyawo_quant.contracts._base import ContractModel

NonEmptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]


class QuantRecipe(ContractModel):
    """Backend-neutral quantization intent."""

    algorithm: Literal["rtn", "gptq", "awq"]
    scheme: Literal["w4a16", "w8a16", "w8a8_int8", "w8a8_fp8"]
    weight_granularity: Literal["tensor", "channel", "group"]
    weight_group_size: int | None = Field(default=None, gt=0)
    weight_symmetric: bool
    activation_mode: Literal["none", "dynamic", "static"]
    ignored_modules: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_recipe_consistency(self) -> Self:
        if self.weight_granularity == "group":
            if self.weight_group_size is None:
                raise ValueError("weight_group_size is required when weight_granularity is group")
        elif self.weight_group_size is not None:
            raise ValueError("weight_group_size is only valid when weight_granularity is group")

        if self.scheme in {"w4a16", "w8a16"}:
            if self.activation_mode != "none":
                raise ValueError("weight-only schemes require activation_mode to be none")
        elif self.activation_mode == "none":
            raise ValueError(
                "weight-and-activation schemes require dynamic or static activation_mode"
            )

        return self
