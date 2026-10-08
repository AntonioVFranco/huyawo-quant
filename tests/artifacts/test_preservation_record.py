"""Tests for private, create-once preservation observations."""

import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import IO

import pytest
from huyawo_quant.artifacts import build_quant_artifact_identity
from huyawo_quant.artifacts.preservation_record import (
    write_unverified_preservation_record,
)
from huyawo_quant.contracts import QuantArtifactIdentity


def make_identity(tmp_path: Path) -> QuantArtifactIdentity:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_bytes(b'{"synthetic":true}\n')

    return build_quant_artifact_identity(
        artifact_root=source,
        layout="huggingface_pretrained",
        serialization_format="safetensors",
    )


def make_private_destination(tmp_path: Path) -> Path:
    directory = tmp_path / ".private-artifacts"
    directory.mkdir(mode=0o700)
    return directory / "observation.json"


def test_private_observation_is_exclusive_and_unverified(
    tmp_path: Path,
) -> None:
    identity = make_identity(tmp_path)
    destination = make_private_destination(tmp_path)

    actual = write_unverified_preservation_record(
        destination_path=destination,
        run_id="synthetic-preservation-001",
        artifact_identity=identity,
    )

    assert actual == destination

    record = json.loads(destination.read_text(encoding="utf-8"))

    assert record["run_id"] == "synthetic-preservation-001"
    assert record["record_type"] == "artifact_preservation_observation"
    assert record["verification_result"] == "NOT_VERIFIED"
    assert record["operational_state"] == "RECOVERY_BLOCKED"
    assert record["restoration_tested"] is False
    assert record["preserved_at"] is None
    assert record["verified_at"] is None
    assert record["privacy_classification"] == "PRIVATE"
    assert record["planned_destination_class"] == "private_git_lfs"
    assert record["blocker_code"] == "PRIVATE_ARTIFACT_COPY_NOT_VERIFIED"

    assert record["artifact_layout"] == identity.layout
    assert record["serialization_format"] == identity.serialization_format
    assert record["artifact_total_bytes"] == sum(item.size_bytes for item in identity.files)

    reconstructed = QuantArtifactIdentity.model_validate_json(
        json.dumps(record["artifact_identity"])
    )
    assert reconstructed == identity

    recorded_at = datetime.fromisoformat(record["recorded_at"])
    assert recorded_at.utcoffset() is not None

    assert stat.S_IMODE(destination.stat().st_mode) & 0o077 == 0
    assert "access_token" not in record
    assert "secret_uri" not in record
    assert "signed_url" not in record


def test_existing_record_is_never_overwritten(tmp_path: Path) -> None:
    identity = make_identity(tmp_path)
    destination = make_private_destination(tmp_path)

    destination.write_bytes(b"existing-private-evidence")

    with pytest.raises(FileExistsError, match="already exists"):
        write_unverified_preservation_record(
            destination_path=destination,
            run_id="synthetic-001",
            artifact_identity=identity,
        )

    assert destination.read_bytes() == b"existing-private-evidence"


@pytest.mark.parametrize(
    "invalid_run_id",
    ["", "../unsafe", "nested/run", "contains space", "a" * 129],
)
def test_invalid_run_id_is_rejected(
    tmp_path: Path,
    invalid_run_id: str,
) -> None:
    identity = make_identity(tmp_path)
    destination = make_private_destination(tmp_path)

    with pytest.raises(ValueError, match="safe nonempty identifier"):
        write_unverified_preservation_record(
            destination_path=destination,
            run_id=invalid_run_id,
            artifact_identity=identity,
        )

    assert not destination.exists()


def test_relative_destination_is_rejected(tmp_path: Path) -> None:
    identity = make_identity(tmp_path)

    with pytest.raises(ValueError, match="must be absolute"):
        write_unverified_preservation_record(
            destination_path=".private-artifacts/observation.json",
            run_id="synthetic-001",
            artifact_identity=identity,
        )


def test_destination_outside_private_area_is_rejected(
    tmp_path: Path,
) -> None:
    identity = make_identity(tmp_path)

    with pytest.raises(ValueError, match="inside .private-artifacts"):
        write_unverified_preservation_record(
            destination_path=tmp_path / "public.json",
            run_id="synthetic-001",
            artifact_identity=identity,
        )


def test_traversal_is_rejected(tmp_path: Path) -> None:
    identity = make_identity(tmp_path)
    destination = make_private_destination(tmp_path)

    with pytest.raises(ValueError, match="must not contain traversal"):
        write_unverified_preservation_record(
            destination_path=destination.parent / ".." / "escaped.json",
            run_id="synthetic-001",
            artifact_identity=identity,
        )

    assert not (tmp_path / "escaped.json").exists()


def test_ancestor_symlink_is_rejected(tmp_path: Path) -> None:
    identity = make_identity(tmp_path)

    parent = tmp_path / "real"
    private = parent / ".private-artifacts"
    private.mkdir(parents=True)

    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic links"):
        write_unverified_preservation_record(
            destination_path=alias / ".private-artifacts" / "record.json",
            run_id="synthetic-001",
            artifact_identity=identity,
        )

    assert not (private / "record.json").exists()


def test_dangling_destination_symlink_is_rejected(
    tmp_path: Path,
) -> None:
    identity = make_identity(tmp_path)
    destination = make_private_destination(tmp_path)
    destination.symlink_to(tmp_path / "missing.json")

    with pytest.raises(FileExistsError, match="already exists"):
        write_unverified_preservation_record(
            destination_path=destination,
            run_id="synthetic-001",
            artifact_identity=identity,
        )

    assert destination.is_symlink()


def test_missing_parent_is_rejected(tmp_path: Path) -> None:
    identity = make_identity(tmp_path)
    destination = tmp_path / ".private-artifacts" / "missing" / "record.json"

    with pytest.raises(ValueError, match="existing directory"):
        write_unverified_preservation_record(
            destination_path=destination,
            run_id="synthetic-001",
            artifact_identity=identity,
        )

    assert not destination.exists()


@pytest.mark.parametrize("mode", [0o755, 0o777, 0o770])
def test_permissive_private_root_is_rejected(
    tmp_path: Path,
    mode: int,
) -> None:
    identity = make_identity(tmp_path)
    destination = make_private_destination(tmp_path)
    private_root = destination.parent

    private_root.chmod(mode)

    try:
        assert stat.S_IMODE(private_root.stat().st_mode) == mode

        with pytest.raises(PermissionError, match="owner-only"):
            write_unverified_preservation_record(
                destination_path=destination,
                run_id="synthetic-001",
                artifact_identity=identity,
            )

        assert not destination.exists()
    finally:
        private_root.chmod(0o700)


def test_permissive_nested_private_directory_is_rejected(
    tmp_path: Path,
) -> None:
    identity = make_identity(tmp_path)
    private_root = make_private_destination(tmp_path).parent
    nested = private_root / "nested"
    nested.mkdir()
    nested.chmod(0o755)

    destination = nested / "observation.json"

    try:
        with pytest.raises(PermissionError, match="owner-only"):
            write_unverified_preservation_record(
                destination_path=destination,
                run_id="synthetic-001",
                artifact_identity=identity,
            )

        assert not destination.exists()
    finally:
        nested.chmod(0o700)


def test_private_root_name_cannot_be_destination_file(
    tmp_path: Path,
) -> None:
    identity = make_identity(tmp_path)
    destination = tmp_path / ".private-artifacts"

    with pytest.raises(ValueError, match="inside .private-artifacts"):
        write_unverified_preservation_record(
            destination_path=destination,
            run_id="synthetic-001",
            artifact_identity=identity,
        )

    assert not destination.exists()


def assert_failed_record_remains_exclusive(
    destination: Path,
    identity: QuantArtifactIdentity,
) -> bytes:
    assert destination.is_file()
    assert not destination.is_symlink()
    assert stat.S_IMODE(destination.stat().st_mode) & 0o077 == 0

    persisted = destination.read_bytes()
    assert persisted

    with pytest.raises(FileExistsError, match="already exists"):
        write_unverified_preservation_record(
            destination_path=destination,
            run_id="synthetic-001",
            artifact_identity=identity,
        )

    assert destination.read_bytes() == persisted
    return persisted


def test_interrupted_write_retains_incomplete_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = make_identity(tmp_path)
    destination = make_private_destination(tmp_path)
    original_fdopen = os.fdopen

    class PartialWriter:
        def __init__(self, handle: IO[bytes]) -> None:
            self.handle = handle

        def write(self, payload: bytes) -> int:
            self.handle.write(payload[:16])
            self.handle.flush()
            raise OSError("synthetic partial write failure")

    @contextmanager
    def interrupted_fdopen(
        descriptor: int,
        mode: str,
    ) -> Iterator[PartialWriter]:
        with original_fdopen(descriptor, mode) as handle:
            yield PartialWriter(handle)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "fdopen", interrupted_fdopen)

        with pytest.raises(OSError, match="synthetic partial write failure"):
            write_unverified_preservation_record(
                destination_path=destination,
                run_id="synthetic-001",
                artifact_identity=identity,
            )

    persisted = assert_failed_record_remains_exclusive(
        destination,
        identity,
    )

    assert len(persisted) == 16

    with pytest.raises(json.JSONDecodeError):
        json.loads(persisted)


def test_fsync_failure_retains_unverified_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = make_identity(tmp_path)
    destination = make_private_destination(tmp_path)

    def failed_fsync(descriptor: int) -> None:
        raise OSError("synthetic fsync failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "fsync", failed_fsync)

        with pytest.raises(OSError, match="synthetic fsync failure"):
            write_unverified_preservation_record(
                destination_path=destination,
                run_id="synthetic-001",
                artifact_identity=identity,
            )

    persisted = assert_failed_record_remains_exclusive(
        destination,
        identity,
    )

    record = json.loads(persisted)

    assert record["verification_result"] == "NOT_VERIFIED"
    assert record["operational_state"] == "RECOVERY_BLOCKED"
    assert record["restoration_tested"] is False
    assert record["preserved_at"] is None
    assert record["verified_at"] is None


def test_readback_mismatch_retains_unverified_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = make_identity(tmp_path)
    destination = make_private_destination(tmp_path)
    original_read_bytes = Path.read_bytes

    def altered_readback(path: Path) -> bytes:
        if path == destination:
            return b"synthetic readback mismatch"
        return original_read_bytes(path)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "read_bytes", altered_readback)

        with pytest.raises(
            RuntimeError,
            match="persisted preservation record changed",
        ):
            write_unverified_preservation_record(
                destination_path=destination,
                run_id="synthetic-001",
                artifact_identity=identity,
            )

    persisted = assert_failed_record_remains_exclusive(
        destination,
        identity,
    )

    record = json.loads(persisted)

    assert record["verification_result"] == "NOT_VERIFIED"
    assert record["operational_state"] == "RECOVERY_BLOCKED"
    assert record["restoration_tested"] is False
    assert record["preserved_at"] is None
    assert record["verified_at"] is None
