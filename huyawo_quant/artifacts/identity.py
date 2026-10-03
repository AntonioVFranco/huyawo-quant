from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal

from huyawo_quant.contracts import ArtifactFileIdentity, QuantArtifactIdentity

_ArtifactLayout = Literal["huggingface_pretrained", "torch_state_dict"]
_HASH_CHUNK_SIZE = 1024 * 1024


def _require_artifact_root(artifact_root: str | Path) -> Path:
    root = Path(artifact_root)

    if root.is_symlink():
        raise ValueError(f"artifact root must not be a symbolic link: {root}")

    if not root.exists():
        raise FileNotFoundError(f"artifact root does not exist: {root}")

    if not root.is_dir():
        raise ValueError(f"artifact root must be a directory: {root}")

    return root


def _discover_regular_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    pending = [root]

    while pending:
        directory = pending.pop()

        with os.scandir(directory) as entries:
            ordered_entries = sorted(
                entries,
                key=lambda entry: entry.name,
            )

            for entry in ordered_entries:
                path = Path(entry.path)

                if entry.is_symlink():
                    raise ValueError(f"artifact tree must not contain symbolic links: {path}")

                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                    continue

                if entry.is_file(follow_symlinks=False):
                    files.append(path)
                    continue

                raise ValueError(
                    f"artifact tree must contain only regular files and directories: {path}"
                )

    if not files:
        raise ValueError("artifact root must contain at least one regular file")

    return tuple(
        sorted(
            files,
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def _measure_regular_file(path: Path) -> tuple[int, str]:
    if path.is_symlink():
        raise ValueError(f"artifact file must not be a symbolic link: {path}")

    if not path.exists():
        raise FileNotFoundError(f"artifact file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"artifact path must be a regular file: {path}")

    digest = hashlib.sha256()
    size_bytes = 0

    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)

    return size_bytes, digest.hexdigest()


def build_quant_artifact_identity(
    *,
    artifact_root: str | Path,
    layout: _ArtifactLayout,
    serialization_format: str,
) -> QuantArtifactIdentity:
    """Build one deterministic identity from exact persisted artifact bytes."""

    root = _require_artifact_root(artifact_root)
    paths = _discover_regular_files(root)

    files: list[ArtifactFileIdentity] = []

    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        size_bytes, sha256 = _measure_regular_file(path)

        files.append(
            ArtifactFileIdentity(
                relative_path=relative_path,
                sha256=sha256,
                size_bytes=size_bytes,
            )
        )

    return QuantArtifactIdentity(
        layout=layout,
        serialization_format=serialization_format,
        files=tuple(files),
    )


def verify_quant_artifact_identity(
    *,
    artifact_root: str | Path,
    expected: QuantArtifactIdentity,
) -> None:
    """Require the current artifact tree to equal one expected identity."""

    if not isinstance(expected, QuantArtifactIdentity):
        raise TypeError("expected must be a QuantArtifactIdentity")

    observed = build_quant_artifact_identity(
        artifact_root=artifact_root,
        layout=expected.layout,
        serialization_format=expected.serialization_format,
    )

    if observed == expected:
        return

    expected_files = {file.relative_path: file for file in expected.files}

    observed_files = {file.relative_path: file for file in observed.files}

    missing_paths = sorted(set(expected_files) - set(observed_files))

    extra_paths = sorted(set(observed_files) - set(expected_files))

    changed_paths = sorted(
        path
        for path in set(expected_files) & set(observed_files)
        if expected_files[path] != observed_files[path]
    )

    raise ValueError(
        "artifact identity does not match expected identity; "
        f"missing={missing_paths}, "
        f"extra={extra_paths}, "
        f"changed={changed_paths}"
    )
