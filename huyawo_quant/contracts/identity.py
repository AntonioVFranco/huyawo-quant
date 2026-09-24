"""Identity contracts for models and tokenizers."""

from typing import Literal

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
