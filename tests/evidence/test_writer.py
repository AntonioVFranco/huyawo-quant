"""Tests for Evidence Bundle materialization."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import huyawo_quant.evidence.writer as writer_module
import pytest
from huyawo_quant.contracts import (
    EvidenceBundle,
    EvidenceFileReference,
    HardwareTarget,
    ModelIdentity,
    RuntimeTarget,
    TokenizerIdentity,
)
from huyawo_quant.evidence import (
    build_evidence_file_reference,
    write_evidence_bundle,
)

STARTED_AT = datetime(
    2026,
    10,
    2,
    12,
    0,
    tzinfo=UTC,
)
FINISHED_AT = datetime(
    2026,
    10,
    2,
    12,
    1,
    tzinfo=UTC,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_source(
    root: Path,
    relative_path: str,
    data: bytes,
) -> Path:
    path = root / relative_path
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_bytes(data)
    return path


def _make_bundle(
    *,
    evidence_files: tuple[
        EvidenceFileReference,
        ...,
    ] = (),
) -> EvidenceBundle:
    return EvidenceBundle(
        schema_version="1",
        quant_version="0.0.0-test",
        run_id="m7-writer-test",
        run_kind="baseline",
        execution_status="COMPLETED",
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
        model=ModelIdentity(
            model_id="org/model",
            requested_revision="main",
            resolved_revision="a" * 40,
        ),
        tokenizer=TokenizerIdentity(
            tokenizer_id="org/model",
            requested_revision="main",
            resolved_revision="b" * 40,
        ),
        hardware=HardwareTarget(
            device_name="NVIDIA RTX A5000",
            device_count=1,
            memory_bytes_per_device=(24_564 * 1024 * 1024),
            compute_capability_major=8,
            compute_capability_minor=6,
        ),
        runtime=RuntimeTarget(
            runtime="vllm",
            version="0.0.0-test",
        ),
        evidence_files=evidence_files,
    )


def test_build_evidence_file_reference_measures_exact_bytes(
    tmp_path: Path,
) -> None:
    data = b"\x00exact-evidence\n"
    source = _write_source(
        tmp_path,
        "source.bin",
        data,
    )

    reference = build_evidence_file_reference(
        source_path=source,
        category="other",
        relative_path="raw/source.bin",
    )

    assert reference == EvidenceFileReference(
        category="other",
        relative_path="raw/source.bin",
        sha256=_sha256(data),
        size_bytes=len(data),
    )


def test_build_evidence_file_reference_rejects_manifest_path(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path,
        "source.bin",
        b"data",
    )

    with pytest.raises(
        ValueError,
        match="reserved",
    ):
        build_evidence_file_reference(
            source_path=source,
            category="other",
            relative_path="evidence_bundle.json",
        )


def test_write_evidence_bundle_materializes_nested_file_and_roundtrips(
    tmp_path: Path,
) -> None:
    data = b'{"fingerprint":"observed"}\n'
    source = _write_source(
        tmp_path,
        "inputs/fingerprint.json",
        data,
    )
    reference = build_evidence_file_reference(
        source_path=source,
        category="environment_fingerprint",
        relative_path="environment/fingerprint.json",
    )
    bundle = _make_bundle(
        evidence_files=(reference,),
    )
    destination = tmp_path / "bundle"

    result = write_evidence_bundle(
        bundle=bundle,
        destination_root=destination,
        source_files={
            reference.relative_path: source,
        },
    )

    persisted = destination / reference.relative_path
    manifest = destination / "evidence_bundle.json"

    assert result == destination
    assert persisted.read_bytes() == data
    assert persisted.stat().st_size == reference.size_bytes
    assert _sha256(persisted.read_bytes()) == reference.sha256
    assert EvidenceBundle.model_validate_json(manifest.read_bytes()) == bundle


def test_write_evidence_bundle_supports_zero_evidence_files(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle()
    destination = tmp_path / "bundle"

    write_evidence_bundle(
        bundle=bundle,
        destination_root=destination,
        source_files={},
    )

    assert tuple(destination.iterdir()) == (destination / "evidence_bundle.json",)
    assert (
        EvidenceBundle.model_validate_json((destination / "evidence_bundle.json").read_bytes())
        == bundle
    )


def test_write_evidence_bundle_rejects_missing_source_mapping(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path,
        "source.log",
        b"log",
    )
    reference = build_evidence_file_reference(
        source_path=source,
        category="execution_log",
        relative_path="logs/run.log",
    )
    bundle = _make_bundle(
        evidence_files=(reference,),
    )
    destination = tmp_path / "bundle"

    with pytest.raises(
        ValueError,
        match="missing=",
    ):
        write_evidence_bundle(
            bundle=bundle,
            destination_root=destination,
            source_files={},
        )

    assert not destination.exists()


def test_write_evidence_bundle_rejects_extra_source_mapping(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle()
    destination = tmp_path / "bundle"

    with pytest.raises(
        ValueError,
        match="extra=",
    ):
        write_evidence_bundle(
            bundle=bundle,
            destination_root=destination,
            source_files={
                "logs/extra.log": (tmp_path / "extra.log"),
            },
        )

    assert not destination.exists()


def test_write_evidence_bundle_rejects_source_size_mismatch(
    tmp_path: Path,
) -> None:
    data = b"abcd"
    source = _write_source(
        tmp_path,
        "source.bin",
        data,
    )
    reference = EvidenceFileReference(
        category="other",
        relative_path="raw/source.bin",
        sha256=_sha256(data),
        size_bytes=len(data) + 1,
    )
    bundle = _make_bundle(
        evidence_files=(reference,),
    )
    destination = tmp_path / "bundle"

    with pytest.raises(
        ValueError,
        match="size",
    ):
        write_evidence_bundle(
            bundle=bundle,
            destination_root=destination,
            source_files={
                reference.relative_path: source,
            },
        )

    assert not destination.exists()


def test_write_evidence_bundle_rejects_source_sha_mismatch(
    tmp_path: Path,
) -> None:
    data = b"abcd"
    source = _write_source(
        tmp_path,
        "source.bin",
        data,
    )
    reference = EvidenceFileReference(
        category="other",
        relative_path="raw/source.bin",
        sha256="0" * 64,
        size_bytes=len(data),
    )
    bundle = _make_bundle(
        evidence_files=(reference,),
    )
    destination = tmp_path / "bundle"

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):
        write_evidence_bundle(
            bundle=bundle,
            destination_root=destination,
            source_files={
                reference.relative_path: source,
            },
        )

    assert not destination.exists()


def test_write_evidence_bundle_rejects_symbolic_link_source(
    tmp_path: Path,
) -> None:
    real_source = _write_source(
        tmp_path,
        "real.bin",
        b"data",
    )
    symlink_source = tmp_path / "source-link.bin"
    symlink_source.symlink_to(real_source)

    reference = EvidenceFileReference(
        category="other",
        relative_path="raw/source.bin",
        sha256=_sha256(b"data"),
        size_bytes=4,
    )
    bundle = _make_bundle(
        evidence_files=(reference,),
    )
    destination = tmp_path / "bundle"

    with pytest.raises(
        ValueError,
        match="symbolic link",
    ):
        write_evidence_bundle(
            bundle=bundle,
            destination_root=destination,
            source_files={
                reference.relative_path: (symlink_source),
            },
        )

    assert not destination.exists()


def test_write_evidence_bundle_rejects_existing_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "bundle"
    destination.mkdir()

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        write_evidence_bundle(
            bundle=_make_bundle(),
            destination_root=destination,
            source_files={},
        )


def test_failed_staged_write_leaves_no_final_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"log"
    source = _write_source(
        tmp_path,
        "source.log",
        data,
    )
    reference = build_evidence_file_reference(
        source_path=source,
        category="execution_log",
        relative_path="logs/run.log",
    )
    bundle = _make_bundle(
        evidence_files=(reference,),
    )
    destination = tmp_path / "bundle"

    def fail_serialization(
        _bundle: EvidenceBundle,
    ) -> bytes:
        raise RuntimeError("forced serialization failure")

    monkeypatch.setattr(
        writer_module,
        "_serialize_bundle",
        fail_serialization,
    )

    with pytest.raises(
        RuntimeError,
        match="forced serialization failure",
    ):
        write_evidence_bundle(
            bundle=bundle,
            destination_root=destination,
            source_files={
                reference.relative_path: source,
            },
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".bundle.staging-*"))


def test_repeated_writes_to_distinct_destinations_are_byte_identical(
    tmp_path: Path,
) -> None:
    first_data = b"alpha\n"
    second_data = b"beta\n"

    first_source = _write_source(
        tmp_path,
        "inputs/alpha.log",
        first_data,
    )
    second_source = _write_source(
        tmp_path,
        "inputs/beta.json",
        second_data,
    )

    first_reference = build_evidence_file_reference(
        source_path=first_source,
        category="execution_log",
        relative_path="logs/alpha.log",
    )
    second_reference = build_evidence_file_reference(
        source_path=second_source,
        category="other",
        relative_path="raw/beta.json",
    )

    bundle = _make_bundle(
        evidence_files=(
            first_reference,
            second_reference,
        )
    )

    first_destination = tmp_path / "bundle-a"
    second_destination = tmp_path / "bundle-b"
    source_files = {
        first_reference.relative_path: (first_source),
        second_reference.relative_path: (second_source),
    }

    write_evidence_bundle(
        bundle=bundle,
        destination_root=first_destination,
        source_files=source_files,
    )
    write_evidence_bundle(
        bundle=bundle,
        destination_root=second_destination,
        source_files=source_files,
    )

    relative_paths = (
        "evidence_bundle.json",
        first_reference.relative_path,
        second_reference.relative_path,
    )

    for relative_path in relative_paths:
        assert (first_destination / relative_path).read_bytes() == (
            second_destination / relative_path
        ).read_bytes()


def test_write_evidence_bundle_rejects_manifest_self_reference(
    tmp_path: Path,
) -> None:
    data = b"not-a-manifest"
    source = _write_source(
        tmp_path,
        "source.bin",
        data,
    )
    reference = EvidenceFileReference(
        category="other",
        relative_path="evidence_bundle.json",
        sha256=_sha256(data),
        size_bytes=len(data),
    )
    bundle = _make_bundle(
        evidence_files=(reference,),
    )
    destination = tmp_path / "bundle"

    with pytest.raises(
        ValueError,
        match="reserved",
    ):
        write_evidence_bundle(
            bundle=bundle,
            destination_root=destination,
            source_files={
                reference.relative_path: source,
            },
        )

    assert not destination.exists()
