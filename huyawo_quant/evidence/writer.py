"""Evidence Bundle materialization."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from huyawo_quant.contracts import EvidenceBundle, EvidenceFileReference

MANIFEST_FILENAME = "evidence_bundle.json"


def _require_regular_nonsymlink_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"source evidence file must not be a symbolic link: {path}")

    if not path.exists():
        raise FileNotFoundError(f"source evidence file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"source evidence path must be a regular file: {path}")


def _measure_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)

    return size_bytes, digest.hexdigest()


def _target_path(root: Path, relative_path: str) -> Path:
    return root.joinpath(*PurePosixPath(relative_path).parts)


def _serialize_bundle(bundle: EvidenceBundle) -> bytes:
    payload = bundle.model_dump(mode="json")
    return (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _verify_file_against_reference(
    path: Path,
    reference: EvidenceFileReference,
) -> None:
    _require_regular_nonsymlink_file(path)
    size_bytes, sha256 = _measure_file(path)

    if size_bytes != reference.size_bytes:
        raise ValueError(f"evidence file size does not match reference: {reference.relative_path}")

    if sha256 != reference.sha256:
        raise ValueError(
            f"evidence file SHA-256 does not match reference: {reference.relative_path}"
        )


def build_evidence_file_reference(
    *,
    source_path: str | Path,
    category: str,
    relative_path: str,
) -> EvidenceFileReference:
    """Build one evidence reference from exact observed source bytes."""

    if relative_path == MANIFEST_FILENAME:
        raise ValueError(f"{MANIFEST_FILENAME} is reserved for the Evidence Bundle manifest")

    source = Path(source_path)
    _require_regular_nonsymlink_file(source)
    size_bytes, sha256 = _measure_file(source)

    payload: dict[str, object] = {
        "category": category,
        "relative_path": relative_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }
    return EvidenceFileReference.model_validate(payload)


def write_evidence_bundle(
    *,
    bundle: EvidenceBundle,
    destination_root: str | Path,
    source_files: Mapping[str, str | Path],
) -> Path:
    """Materialize one validated Evidence Bundle using create-once semantics."""

    destination = Path(destination_root)

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination bundle root already exists: {destination}")

    expected_paths = tuple(reference.relative_path for reference in bundle.evidence_files)
    expected_path_set = set(expected_paths)
    provided_path_set = set(source_files)

    if MANIFEST_FILENAME in expected_path_set:
        raise ValueError(f"{MANIFEST_FILENAME} is reserved for the Evidence Bundle manifest")

    missing_paths = sorted(expected_path_set - provided_path_set)
    extra_paths = sorted(provided_path_set - expected_path_set)

    if missing_paths or extra_paths:
        raise ValueError(
            "source file mapping must exactly match bundle evidence paths; "
            f"missing={missing_paths}, extra={extra_paths}"
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )

    try:
        for reference in bundle.evidence_files:
            source = Path(source_files[reference.relative_path])
            _verify_file_against_reference(
                source,
                reference,
            )

            target = _target_path(
                staging,
                reference.relative_path,
            )
            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with (
                source.open("rb") as source_handle,
                target.open("xb") as target_handle,
            ):
                shutil.copyfileobj(
                    source_handle,
                    target_handle,
                )

            _verify_file_against_reference(
                target,
                reference,
            )

        manifest_bytes = _serialize_bundle(bundle)
        manifest_path = staging / MANIFEST_FILENAME
        manifest_path.write_bytes(manifest_bytes)

        roundtripped_bundle = EvidenceBundle.model_validate_json(manifest_bytes)

        if roundtripped_bundle != bundle:
            raise RuntimeError("persisted Evidence Bundle manifest changed contract value")

        for reference in roundtripped_bundle.evidence_files:
            target = _target_path(
                staging,
                reference.relative_path,
            )
            _verify_file_against_reference(
                target,
                reference,
            )

        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"destination bundle root already exists: {destination}")

        staging.rename(destination)
    except Exception:
        shutil.rmtree(
            staging,
            ignore_errors=True,
        )
        raise

    return destination
