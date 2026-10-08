"""Create-once private preservation observations without success claims."""

import json
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path

from huyawo_quant.contracts import QuantArtifactIdentity

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def write_unverified_preservation_record(
    *,
    destination_path: str | Path,
    run_id: str,
    artifact_identity: QuantArtifactIdentity,
) -> Path:
    """Persist an unverified preservation observation without overwriting."""

    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id must be a safe nonempty identifier")

    if not isinstance(artifact_identity, QuantArtifactIdentity):
        raise TypeError("artifact_identity must be a QuantArtifactIdentity")

    destination = Path(destination_path)

    if not destination.is_absolute():
        raise ValueError("destination_path must be absolute")

    if ".." in destination.parts:
        raise ValueError("destination_path must not contain traversal")

    if ".private-artifacts" not in destination.parent.parts:
        raise ValueError("destination_path must be inside .private-artifacts")

    if destination.exists() or destination.is_symlink():
        raise FileExistsError("preservation record destination already exists")

    for component in (destination, *destination.parents):
        if component.is_symlink():
            raise ValueError("destination_path must not traverse symbolic links")

    if not destination.parent.is_dir():
        raise ValueError("destination parent must be an existing directory")

    private_parts = destination.parent.parts
    private_index = private_parts.index(".private-artifacts")
    private_root = Path(*private_parts[: private_index + 1])

    current = destination.parent

    while current != private_root.parent:
        metadata = current.stat(follow_symlinks=False)

        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("private record parent must be a directory")

        if metadata.st_uid != os.geteuid():
            raise PermissionError("private record directory must be owned by current user")

        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise PermissionError("private record directory permissions must be owner-only")

        current = current.parent

    record: dict[str, object] = {
        "record_type": "artifact_preservation_observation",
        "run_id": run_id,
        "artifact_identity": artifact_identity.model_dump(mode="json"),
        "artifact_layout": artifact_identity.layout,
        "serialization_format": artifact_identity.serialization_format,
        "artifact_total_bytes": sum(item.size_bytes for item in artifact_identity.files),
        "planned_destination_class": "private_git_lfs",
        "recorded_at": datetime.now(UTC).isoformat(),
        "preserved_at": None,
        "verified_at": None,
        "verification_result": "NOT_VERIFIED",
        "restoration_tested": False,
        "privacy_classification": "PRIVATE",
        "operational_state": "RECOVERY_BLOCKED",
        "blocker_code": "PRIVATE_ARTIFACT_COPY_NOT_VERIFIED",
    }

    payload = (json.dumps(record, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)

    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

    persisted = destination.read_bytes()

    if persisted != payload:
        raise RuntimeError("persisted preservation record changed")

    restored = json.loads(persisted)

    if restored != record:
        raise RuntimeError("preservation record JSON roundtrip failed")

    restored_identity = QuantArtifactIdentity.model_validate_json(
        json.dumps(restored["artifact_identity"])
    )

    if restored_identity != artifact_identity:
        raise RuntimeError("preservation record artifact identity changed")

    if stat.S_IMODE(destination.stat().st_mode) & 0o077:
        raise RuntimeError("preservation record permissions are not private")

    return destination
