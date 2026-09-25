"""Tests for quantized artifact identity contracts."""

import pytest
from huyawo_quant.contracts import ArtifactFileIdentity, QuantArtifactIdentity
from pydantic import ValidationError

SHA_A = "a" * 64
SHA_B = "b" * 64


def make_file(
    relative_path: str = "model.safetensors",
    sha256: str = SHA_A,
    size_bytes: int = 1024,
) -> ArtifactFileIdentity:
    return ArtifactFileIdentity(
        relative_path=relative_path,
        sha256=sha256,
        size_bytes=size_bytes,
    )


def make_artifact() -> QuantArtifactIdentity:
    return QuantArtifactIdentity(
        layout="huggingface_pretrained",
        serialization_format="compressed_tensors",
        files=(
            make_file(
                relative_path="config.json",
                sha256=SHA_A,
                size_bytes=512,
            ),
            make_file(
                relative_path="model.safetensors",
                sha256=SHA_B,
                size_bytes=4096,
            ),
        ),
    )


def test_artifact_file_identity_serializes_expected_fields() -> None:
    identity = make_file()

    assert identity.model_dump(mode="json") == {
        "relative_path": "model.safetensors",
        "sha256": SHA_A,
        "size_bytes": 1024,
    }


def test_quant_artifact_identity_serializes_expected_fields() -> None:
    identity = make_artifact()

    assert identity.model_dump(mode="json") == {
        "layout": "huggingface_pretrained",
        "serialization_format": "compressed_tensors",
        "files": [
            {
                "relative_path": "config.json",
                "sha256": SHA_A,
                "size_bytes": 512,
            },
            {
                "relative_path": "model.safetensors",
                "sha256": SHA_B,
                "size_bytes": 4096,
            },
        ],
    }


@pytest.mark.parametrize(
    "layout",
    ["huggingface_pretrained", "torch_state_dict"],
)
def test_quant_artifact_identity_accepts_initial_layouts(
    layout: str,
) -> None:
    payload: dict[str, object] = {
        "layout": layout,
        "serialization_format": "test-format",
        "files": (make_file(),),
    }

    identity = QuantArtifactIdentity.model_validate(payload)

    assert identity.layout == layout


def test_quant_artifact_identity_rejects_unknown_layout() -> None:
    payload = make_artifact().model_dump()
    payload["layout"] = "gguf"

    with pytest.raises(ValidationError):
        QuantArtifactIdentity.model_validate(payload)


@pytest.mark.parametrize("serialization_format", ["", "   "])
def test_quant_artifact_identity_rejects_empty_serialization_format(
    serialization_format: str,
) -> None:
    with pytest.raises(ValidationError):
        QuantArtifactIdentity(
            layout="huggingface_pretrained",
            serialization_format=serialization_format,
            files=(make_file(),),
        )


def test_quant_artifact_identity_rejects_empty_files() -> None:
    with pytest.raises(ValidationError):
        QuantArtifactIdentity(
            layout="huggingface_pretrained",
            serialization_format="compressed_tensors",
            files=(),
        )


@pytest.mark.parametrize(
    "sha256",
    [
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
    ],
)
def test_artifact_file_identity_rejects_invalid_sha256(
    sha256: str,
) -> None:
    with pytest.raises(ValidationError):
        make_file(sha256=sha256)


def test_artifact_file_identity_accepts_zero_size() -> None:
    identity = make_file(size_bytes=0)

    assert identity.size_bytes == 0


def test_artifact_file_identity_rejects_negative_size() -> None:
    with pytest.raises(ValidationError):
        make_file(size_bytes=-1)


@pytest.mark.parametrize(
    "relative_path",
    [
        "/model.safetensors",
        "../model.safetensors",
        "weights/../model.safetensors",
        "./model.safetensors",
        "weights//model.safetensors",
        "weights\\model.safetensors",
        "weights/",
        ".",
        "..",
    ],
)
def test_artifact_file_identity_rejects_non_normalized_or_unsafe_paths(
    relative_path: str,
) -> None:
    with pytest.raises(ValidationError):
        make_file(relative_path=relative_path)


def test_quant_artifact_identity_rejects_duplicate_paths() -> None:
    with pytest.raises(ValidationError):
        QuantArtifactIdentity(
            layout="huggingface_pretrained",
            serialization_format="compressed_tensors",
            files=(
                make_file(
                    relative_path="model.safetensors",
                    sha256=SHA_A,
                ),
                make_file(
                    relative_path="model.safetensors",
                    sha256=SHA_B,
                ),
            ),
        )


def test_quant_artifact_identity_rejects_unsorted_files() -> None:
    with pytest.raises(ValidationError):
        QuantArtifactIdentity(
            layout="huggingface_pretrained",
            serialization_format="compressed_tensors",
            files=(
                make_file(
                    relative_path="model.safetensors",
                    sha256=SHA_B,
                ),
                make_file(
                    relative_path="config.json",
                    sha256=SHA_A,
                ),
            ),
        )


def test_quant_artifact_identity_uses_strict_tuple_typing() -> None:
    payload: dict[str, object] = make_artifact().model_dump()
    payload["files"] = [make_file()]

    with pytest.raises(ValidationError):
        QuantArtifactIdentity.model_validate(payload)


def test_artifact_file_identity_rejects_extra_fields() -> None:
    payload: dict[str, object] = make_file().model_dump()
    payload["absolute_path"] = "/workspace/model.safetensors"

    with pytest.raises(ValidationError):
        ArtifactFileIdentity.model_validate(payload)


def test_quant_artifact_identity_rejects_extra_fields() -> None:
    payload: dict[str, object] = make_artifact().model_dump()
    payload["runtime_validated"] = True

    with pytest.raises(ValidationError):
        QuantArtifactIdentity.model_validate(payload)


def test_artifact_file_identity_is_immutable() -> None:
    identity = make_file()

    with pytest.raises(ValidationError):
        identity.__setattr__("size_bytes", 2048)


def test_quant_artifact_identity_is_immutable() -> None:
    identity = make_artifact()

    with pytest.raises(ValidationError):
        identity.__setattr__("serialization_format", "safetensors")
