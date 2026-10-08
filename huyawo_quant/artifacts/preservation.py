"""Local, identity-verified artifact restoration."""

import os
import shutil
from pathlib import Path

from huyawo_quant.artifacts.identity import verify_quant_artifact_identity
from huyawo_quant.contracts import QuantArtifactIdentity

_COPY_CHUNK_SIZE = 1024 * 1024


def _reject_unsafe_path_ancestry(path: Path, *, label: str) -> None:
    if ".." in path.parts:
        raise ValueError(f"{label} must not contain traversal")

    absolute_path = path.absolute()

    for component in (absolute_path, *absolute_path.parents):
        if component.is_symlink():
            raise ValueError(f"{label} must not traverse symbolic links: {component}")


def copy_verified_artifact_tree(
    *,
    source_root: str | Path,
    destination_root: str | Path,
    expected: QuantArtifactIdentity,
) -> None:
    """Copy one verified artifact into a new, exclusive destination.

    Failed copies leave their incomplete destination intact for inspection.
    A destination is accepted only after all identity checks succeed.
    """
    source = Path(source_root)
    destination = Path(destination_root)

    if not destination.is_absolute():
        raise ValueError("destination_root must be absolute")

    if ".." in destination.parts:
        raise ValueError("destination_root must not contain traversal")

    if destination.is_symlink() or destination.exists():
        raise FileExistsError("artifact destination already exists")

    parent = destination.parent

    if parent.is_symlink():
        raise ValueError("destination parent must not be a symbolic link")

    if not parent.is_dir():
        raise ValueError("destination parent must be an existing directory")

    _reject_unsafe_path_ancestry(source, label="source_root")
    _reject_unsafe_path_ancestry(destination, label="destination_root")

    source_real = source.resolve(strict=True)
    parent_real = parent.resolve(strict=True)

    if source_real == parent_real or source_real in parent_real.parents:
        raise ValueError("destination must not be inside the source artifact")

    verify_quant_artifact_identity(
        artifact_root=source,
        expected=expected,
    )

    destination.mkdir(mode=0o700, exist_ok=False)

    for file_identity in expected.files:
        relative_path = Path(file_identity.relative_path)
        source_file = source / relative_path
        destination_file = destination / relative_path

        destination_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if source_file.is_symlink():
            raise ValueError("source artifact contains a symbolic link")

        with source_file.open("rb") as reader, destination_file.open("xb") as writer:
            shutil.copyfileobj(
                reader,
                writer,
                length=_COPY_CHUNK_SIZE,
            )
            writer.flush()
            os.fsync(writer.fileno())

    verify_quant_artifact_identity(
        artifact_root=source,
        expected=expected,
    )

    verify_quant_artifact_identity(
        artifact_root=destination,
        expected=expected,
    )
