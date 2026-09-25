"""Quantized artifact identity contracts."""

from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from huyawo_quant.contracts._base import ContractModel

Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ArtifactFileIdentity(ContractModel):
    """Content identity of one file inside an artifact."""

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
            raise ValueError("relative_path must remain within the artifact root")

        if path.as_posix() != value:
            raise ValueError("relative_path must already be normalized")

        return value


class QuantArtifactIdentity(ContractModel):
    """Portable content identity of one persisted quantized artifact."""

    layout: Literal["huggingface_pretrained", "torch_state_dict"]
    serialization_format: str = Field(min_length=1, pattern=r"\S")
    files: tuple[ArtifactFileIdentity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_file_manifest(self) -> Self:
        paths = tuple(file.relative_path for file in self.files)

        if len(paths) != len(set(paths)):
            raise ValueError("artifact file paths must be unique")

        if paths != tuple(sorted(paths)):
            raise ValueError("artifact files must be ordered lexicographically by relative_path")

        return self
