"""Calibration contracts for Huyawo Quant."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from huyawo_quant.contracts._base import ContractModel
from huyawo_quant.contracts.identity import DatasetIdentity, TokenizerIdentity

NonEmptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]
NonNegativeIndex = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class CalibrationPreprocessing(ContractModel):
    """Deterministic preprocessing configuration for calibration samples."""

    source_column: NonEmptyString
    render_mode: Literal["raw_text", "chat_template"]
    add_generation_prompt: bool
    add_special_tokens: bool
    truncation: bool

    @model_validator(mode="after")
    def validate_render_mode(self) -> Self:
        if self.render_mode == "raw_text" and self.add_generation_prompt:
            raise ValueError("raw_text preprocessing cannot add a generation prompt")

        return self


class CalibrationManifest(ContractModel):
    """Frozen calibration protocol for one quantization candidate."""

    schema_version: NonEmptyString
    dataset: DatasetIdentity
    tokenizer: TokenizerIdentity
    preprocessing: CalibrationPreprocessing
    sample_count: PositiveInt
    sample_indices: tuple[NonNegativeIndex, ...] = Field(min_length=1)
    max_sequence_length: PositiveInt
    token_budget: PositiveInt | None = None
    seed: NonNegativeInt
    shuffle: bool

    @model_validator(mode="after")
    def validate_sample_selection(self) -> Self:
        if self.sample_count != len(self.sample_indices):
            raise ValueError("sample_count must equal len(sample_indices)")

        if len(self.sample_indices) != len(set(self.sample_indices)):
            raise ValueError("sample_indices must be unique")

        return self
