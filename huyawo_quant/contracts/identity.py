"""Identity contracts for models, tokenizers, and datasets."""

from typing import Annotated, Literal

from pydantic import Field

from huyawo_quant.contracts._base import ContractModel


class ModelIdentity(ContractModel):
    """Resolved identity of a model checkpoint."""

    source: Literal["huggingface"] = "huggingface"
    model_id: str = Field(min_length=1, pattern=r"\S")
    requested_revision: str = Field(min_length=1, pattern=r"\S")
    resolved_revision: str = Field(min_length=1, pattern=r"\S")


class TokenizerIdentity(ContractModel):
    """Resolved identity of a tokenizer."""

    source: Literal["huggingface"] = "huggingface"
    tokenizer_id: str = Field(min_length=1, pattern=r"\S")
    requested_revision: str = Field(min_length=1, pattern=r"\S")
    resolved_revision: str = Field(min_length=1, pattern=r"\S")


class DatasetIdentity(ContractModel):
    """Resolved identity of a dataset selection."""

    source: Literal["huggingface"] = "huggingface"
    dataset_id: str = Field(min_length=1, pattern=r"\S")
    config_name: Annotated[str, Field(min_length=1, pattern=r"\S")] | None = None
    split: str = Field(min_length=1, pattern=r"\S")
    requested_revision: str = Field(min_length=1, pattern=r"\S")
    resolved_revision: str = Field(min_length=1, pattern=r"\S")
