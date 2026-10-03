from __future__ import annotations

import hashlib
import os
from pathlib import Path

import huyawo_quant.artifacts as artifacts_package
import pytest
from huyawo_quant.artifacts import (
    build_quant_artifact_identity,
    verify_quant_artifact_identity,
)
from huyawo_quant.contracts import (
    ArtifactFileIdentity,
    QuantArtifactIdentity,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_example_tree(root: Path) -> None:
    (root / "nested").mkdir()
    (root / "z.bin").write_bytes(b"z")
    (root / "nested" / "a.bin").write_bytes(b"alpha")
    (root / "empty.bin").write_bytes(b"")


def test_artifacts_package_exports_minimal_identity_surface() -> None:
    assert artifacts_package.__all__ == [
        "build_quant_artifact_identity",
        "verify_quant_artifact_identity",
    ]
    assert artifacts_package.build_quant_artifact_identity is build_quant_artifact_identity
    assert artifacts_package.verify_quant_artifact_identity is verify_quant_artifact_identity


def test_build_quant_artifact_identity_measures_recursive_tree_exactly(
    tmp_path: Path,
) -> None:
    _build_example_tree(tmp_path)

    identity = build_quant_artifact_identity(
        artifact_root=tmp_path,
        layout="huggingface_pretrained",
        serialization_format="safetensors",
    )

    assert identity == QuantArtifactIdentity(
        layout="huggingface_pretrained",
        serialization_format="safetensors",
        files=(
            ArtifactFileIdentity(
                relative_path="empty.bin",
                sha256=_sha256(b""),
                size_bytes=0,
            ),
            ArtifactFileIdentity(
                relative_path="nested/a.bin",
                sha256=_sha256(b"alpha"),
                size_bytes=5,
            ),
            ArtifactFileIdentity(
                relative_path="z.bin",
                sha256=_sha256(b"z"),
                size_bytes=1,
            ),
        ),
    )


def test_build_quant_artifact_identity_is_deterministic(
    tmp_path: Path,
) -> None:
    _build_example_tree(tmp_path)

    first = build_quant_artifact_identity(
        artifact_root=tmp_path,
        layout="huggingface_pretrained",
        serialization_format="safetensors",
    )
    second = build_quant_artifact_identity(
        artifact_root=tmp_path,
        layout="huggingface_pretrained",
        serialization_format="safetensors",
    )

    assert first == second
    assert tuple(file.relative_path for file in first.files) == (
        "empty.bin",
        "nested/a.bin",
        "z.bin",
    )


def test_build_quant_artifact_identity_rejects_empty_tree(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="must contain at least one regular file",
    ):
        build_quant_artifact_identity(
            artifact_root=tmp_path,
            layout="huggingface_pretrained",
            serialization_format="safetensors",
        )


def test_build_quant_artifact_identity_rejects_missing_root(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(
        FileNotFoundError,
        match="artifact root does not exist",
    ):
        build_quant_artifact_identity(
            artifact_root=missing,
            layout="huggingface_pretrained",
            serialization_format="safetensors",
        )


def test_build_quant_artifact_identity_rejects_file_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact.bin"
    root.write_bytes(b"artifact")

    with pytest.raises(
        ValueError,
        match="artifact root must be a directory",
    ):
        build_quant_artifact_identity(
            artifact_root=root,
            layout="huggingface_pretrained",
            serialization_format="safetensors",
        )


def test_build_quant_artifact_identity_rejects_root_symlink(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    (real_root / "model.bin").write_bytes(b"model")

    linked_root = tmp_path / "linked"
    linked_root.symlink_to(
        real_root,
        target_is_directory=True,
    )

    with pytest.raises(
        ValueError,
        match="artifact root must not be a symbolic link",
    ):
        build_quant_artifact_identity(
            artifact_root=linked_root,
            layout="huggingface_pretrained",
            serialization_format="safetensors",
        )


def test_build_quant_artifact_identity_rejects_nested_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")

    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    (artifact_root / "model.bin").write_bytes(b"model")
    (artifact_root / "linked.bin").symlink_to(target)

    with pytest.raises(
        ValueError,
        match="artifact tree must not contain symbolic links",
    ):
        build_quant_artifact_identity(
            artifact_root=artifact_root,
            layout="huggingface_pretrained",
            serialization_format="safetensors",
        )


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"),
    reason="FIFO creation is unavailable on this platform",
)
def test_build_quant_artifact_identity_rejects_non_regular_entry(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.bin").write_bytes(b"model")
    fifo = tmp_path / "artifact.fifo"
    os.mkfifo(fifo)

    with pytest.raises(
        ValueError,
        match="must contain only regular files and directories",
    ):
        build_quant_artifact_identity(
            artifact_root=tmp_path,
            layout="huggingface_pretrained",
            serialization_format="safetensors",
        )


def test_verify_quant_artifact_identity_accepts_exact_tree_without_mutation(
    tmp_path: Path,
) -> None:
    _build_example_tree(tmp_path)

    expected = build_quant_artifact_identity(
        artifact_root=tmp_path,
        layout="huggingface_pretrained",
        serialization_format="safetensors",
    )

    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    verify_quant_artifact_identity(
        artifact_root=tmp_path,
        expected=expected,
    )

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert after == before


def test_verify_quant_artifact_identity_detects_file_mutation(
    tmp_path: Path,
) -> None:
    _build_example_tree(tmp_path)

    expected = build_quant_artifact_identity(
        artifact_root=tmp_path,
        layout="huggingface_pretrained",
        serialization_format="safetensors",
    )

    (tmp_path / "nested" / "a.bin").write_bytes(b"changed")

    with pytest.raises(
        ValueError,
        match=r"changed=\['nested/a\.bin'\]",
    ):
        verify_quant_artifact_identity(
            artifact_root=tmp_path,
            expected=expected,
        )


def test_verify_quant_artifact_identity_detects_added_file(
    tmp_path: Path,
) -> None:
    _build_example_tree(tmp_path)

    expected = build_quant_artifact_identity(
        artifact_root=tmp_path,
        layout="huggingface_pretrained",
        serialization_format="safetensors",
    )

    (tmp_path / "added.bin").write_bytes(b"added")

    with pytest.raises(
        ValueError,
        match=r"extra=\['added\.bin'\]",
    ):
        verify_quant_artifact_identity(
            artifact_root=tmp_path,
            expected=expected,
        )


def test_verify_quant_artifact_identity_detects_removed_file(
    tmp_path: Path,
) -> None:
    _build_example_tree(tmp_path)

    expected = build_quant_artifact_identity(
        artifact_root=tmp_path,
        layout="huggingface_pretrained",
        serialization_format="safetensors",
    )

    (tmp_path / "z.bin").unlink()

    with pytest.raises(
        ValueError,
        match=r"missing=\['z\.bin'\]",
    ):
        verify_quant_artifact_identity(
            artifact_root=tmp_path,
            expected=expected,
        )


def test_verify_quant_artifact_identity_requires_contract_instance(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.bin").write_bytes(b"model")

    with pytest.raises(
        TypeError,
        match="expected must be a QuantArtifactIdentity",
    ):
        verify_quant_artifact_identity(
            artifact_root=tmp_path,
            expected={"not": "an identity"},  # type: ignore[arg-type]
        )


def test_metadata_is_bound_into_quant_artifact_identity(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.bin").write_bytes(b"model")

    huggingface = build_quant_artifact_identity(
        artifact_root=tmp_path,
        layout="huggingface_pretrained",
        serialization_format="safetensors",
    )

    state_dict = build_quant_artifact_identity(
        artifact_root=tmp_path,
        layout="torch_state_dict",
        serialization_format="pytorch",
    )

    assert huggingface.files == state_dict.files
    assert huggingface != state_dict
    assert huggingface.layout == "huggingface_pretrained"
    assert state_dict.layout == "torch_state_dict"
    assert huggingface.serialization_format == "safetensors"
    assert state_dict.serialization_format == "pytorch"
