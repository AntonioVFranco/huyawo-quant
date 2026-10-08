"""Tests for identity-verified local artifact restoration."""

from pathlib import Path

import huyawo_quant.artifacts.preservation as preservation
import pytest
from huyawo_quant.artifacts import (
    build_quant_artifact_identity,
    verify_quant_artifact_identity,
)
from huyawo_quant.contracts import QuantArtifactIdentity

FILES = {
    "chat_template.jinja": b"{{ messages }}",
    "config.json": b'{"model_type":"synthetic"}\n',
    "generation_config.json": b'{"max_new_tokens":8}\n',
    "model.safetensors": bytes(range(256)) * 16,
    "recipe.yaml": b"algorithm: synthetic\n",
    "tokenizer.json": b'{"synthetic":true}\n',
    "tokenizer_config.json": b'{"synthetic":true}\n',
}


def build_source(tmp_path: Path) -> tuple[Path, QuantArtifactIdentity]:
    source = tmp_path / "source"
    source.mkdir()

    for name, content in FILES.items():
        (source / name).write_bytes(content)

    expected = build_quant_artifact_identity(
        artifact_root=source,
        layout="huggingface_pretrained",
        serialization_format="safetensors",
    )

    return source, expected


def test_copy_restores_all_seven_files_exactly(tmp_path: Path) -> None:
    source, expected = build_source(tmp_path)
    destination = tmp_path / "restored"

    preservation.copy_verified_artifact_tree(
        source_root=source,
        destination_root=destination,
        expected=expected,
    )

    verify_quant_artifact_identity(
        artifact_root=destination,
        expected=expected,
    )

    assert len(expected.files) == 7

    for name, content in FILES.items():
        assert (source / name).read_bytes() == content
        assert (destination / name).read_bytes() == content


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    source, expected = build_source(tmp_path)
    destination = tmp_path / "restored"
    destination.mkdir()

    sentinel = destination / "existing.txt"
    sentinel.write_bytes(b"preserve")

    with pytest.raises(FileExistsError, match="already exists"):
        preservation.copy_verified_artifact_tree(
            source_root=source,
            destination_root=destination,
            expected=expected,
        )

    assert sentinel.read_bytes() == b"preserve"
    assert tuple(destination.iterdir()) == (sentinel,)


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra"])
def test_invalid_source_is_rejected_before_destination_creation(
    tmp_path: Path,
    mutation: str,
) -> None:
    source, expected = build_source(tmp_path)
    destination = tmp_path / "restored"

    if mutation == "changed":
        (source / "recipe.yaml").write_bytes(b"changed")
    elif mutation == "missing":
        (source / "tokenizer.json").unlink()
    else:
        (source / "unexpected.bin").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="identity does not match"):
        preservation.copy_verified_artifact_tree(
            source_root=source,
            destination_root=destination,
            expected=expected,
        )

    assert not destination.exists()


def test_symlink_is_rejected_before_copy(tmp_path: Path) -> None:
    source, expected = build_source(tmp_path)
    destination = tmp_path / "restored"

    (source / "linked.bin").symlink_to(source / "model.safetensors")

    with pytest.raises(ValueError, match="symbolic links"):
        preservation.copy_verified_artifact_tree(
            source_root=source,
            destination_root=destination,
            expected=expected,
        )

    assert not destination.exists()


def test_destination_inside_source_is_rejected(tmp_path: Path) -> None:
    source, expected = build_source(tmp_path)
    destination = source / "nested"

    with pytest.raises(ValueError, match="inside the source"):
        preservation.copy_verified_artifact_tree(
            source_root=source,
            destination_root=destination,
            expected=expected,
        )

    assert not destination.exists()


def test_symlink_destination_parent_is_rejected(tmp_path: Path) -> None:
    source, expected = build_source(tmp_path)

    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(tmp_path, target_is_directory=True)

    destination = linked_parent / "restored"

    with pytest.raises(ValueError, match="destination parent"):
        preservation.copy_verified_artifact_tree(
            source_root=source,
            destination_root=destination,
            expected=expected,
        )

    assert not (tmp_path / "restored").exists()


def test_failed_copy_preserves_incomplete_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, expected = build_source(tmp_path)
    destination = tmp_path / "restored"

    def fail_copy(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(
        "huyawo_quant.artifacts.preservation.shutil.copyfileobj",
        fail_copy,
    )

    with pytest.raises(OSError, match="synthetic copy failure"):
        preservation.copy_verified_artifact_tree(
            source_root=source,
            destination_root=destination,
            expected=expected,
        )

    assert destination.is_dir()

    with pytest.raises(ValueError, match="identity does not match"):
        verify_quant_artifact_identity(
            artifact_root=destination,
            expected=expected,
        )


def test_relative_destination_is_rejected(tmp_path: Path) -> None:
    source, expected = build_source(tmp_path)

    with pytest.raises(ValueError, match="must be absolute"):
        preservation.copy_verified_artifact_tree(
            source_root=source,
            destination_root="relative-restored",
            expected=expected,
        )


def test_source_ancestor_symlink_is_rejected_before_copy(
    tmp_path: Path,
) -> None:
    source, expected = build_source(tmp_path)

    alias = tmp_path / "source-parent-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)

    destination = tmp_path / "restored"

    with pytest.raises(
        ValueError,
        match="source_root must not traverse symbolic links",
    ):
        preservation.copy_verified_artifact_tree(
            source_root=alias / source.name,
            destination_root=destination,
            expected=expected,
        )

    assert not destination.exists()


def test_destination_ancestor_symlink_is_rejected_before_copy(
    tmp_path: Path,
) -> None:
    source, expected = build_source(tmp_path)

    real_parent = tmp_path / "real-destination-parent"
    nested_parent = real_parent / "nested"
    nested_parent.mkdir(parents=True)

    alias = tmp_path / "destination-parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    destination = alias / "nested" / "restored"

    with pytest.raises(
        ValueError,
        match="destination_root must not traverse symbolic links",
    ):
        preservation.copy_verified_artifact_tree(
            source_root=source,
            destination_root=destination,
            expected=expected,
        )

    assert not (nested_parent / "restored").exists()
