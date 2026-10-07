from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from huyawo_quant.artifacts.compatibility import validate_transformers_artifact_compatibility
from huyawo_quant.artifacts.identity import (
    build_quant_artifact_identity,
    verify_quant_artifact_identity,
)

PROJECT = Path("/workspace/huyawo-quant")
PRIMARY_VENV = PROJECT / ".venv"
EXPECTED_HEAD = "7037a65be17a4e608fc322a7011a8acdf2a061bd"
EXPECTED_BRANCH = "main"
EXPECTED_ORIGIN = "https://github.com/AntonioVFranco/huyawo-quant.git"
RUN_ID = "m11-qwen2.5-0.5b-instruct-gptq-w4a16-reproduction-20261006-001"
EXECUTION_ROOT = PROJECT / "evidence" / "milestone-11" / RUN_ID / "artifact-execution-20261007-001"
ARTIFACT_ROOT = PROJECT / "artifacts" / RUN_ID
EXPECTED_EXECUTION_SUMS_SHA256 = "eb6cb2ec4bc2c593081eb3b841f7627cbc0d7dbf9507788318295bc70bca60f6"
EXPECTED_ARTIFACT_LAYOUT = "huggingface_pretrained"
EXPECTED_ARTIFACT_SERIALIZATION_FORMAT = "safetensors"
EXPECTED_ARTIFACT_TOTAL_BYTES = 468416746
EXPECTED_ARTIFACT_FILES = {
    "chat_template.jinja": (2507, "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"),
    "config.json": (1604, "46c5d7b24545bda5dcad64d7f8c1769cebfa4d475293effe8bf4eab2603a3649"),
    "generation_config.json": (242, "e694a23c4adabaa1177508c9407bee789ff55f3d5908dfe6755d93a6af55f71d"),
    "model.safetensors": (456988896, "8d129d1885c5ef8a69c019946ff00d4e6dd443442e11676614928b4715834985"),
    "recipe.yaml": (912, "26343e1c65d13a3ed3b7dec02223796353845a4c18d285ad2c2db32b108d8983"),
    "tokenizer.json": (11421892, "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8"),
    "tokenizer_config.json": (693, "2e3bd3cb5055f7a903a54e7062b2fd14d21b70f4174c3ffcd9fc81ec50a25631"),
}
PROMPT = "Hello"
MAX_INPUT_TOKENS = 32
MAX_NEW_TOKENS = 8
SEED = 0
CUDA_DEVICE_INDEX = 0
LOCAL_RUNTIME_PREFIX = ".venv-m12-vllm-0.30.0/"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=PROJECT,
        text=True,
    ).strip()


def verify_repository_scope() -> None:
    if Path(sys.prefix).resolve() != PRIMARY_VENV.resolve():
        fail("primary virtual environment is not active")

    if git_output("rev-parse", "HEAD") != EXPECTED_HEAD:
        fail("unexpected repository HEAD")

    if git_output("branch", "--show-current") != EXPECTED_BRANCH:
        fail("unexpected repository branch")

    if git_output("remote", "get-url", "origin") != EXPECTED_ORIGIN:
        fail("unexpected repository origin")

    if subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=PROJECT,
        check=False,
    ).returncode != 0:
        fail("tracked worktree is not clean")

    if subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=PROJECT,
        check=False,
    ).returncode != 0:
        fail("staged index is not empty")

    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=PROJECT,
        text=True,
    ).splitlines()

    allowed_prefixes = (
        LOCAL_RUNTIME_PREFIX,
        f"artifacts/{RUN_ID}/",
        f"evidence/milestone-11/{RUN_ID}/artifact-execution-20261007-001/",
    )

    unexpected = sorted(
        path
        for path in untracked
        if not any(path.startswith(prefix) for prefix in allowed_prefixes)
    )

    if unexpected:
        fail(f"unexpected untracked paths: {unexpected!r}")


def verify_execution_evidence() -> None:
    if not EXECUTION_ROOT.is_dir() or EXECUTION_ROOT.is_symlink():
        fail("execution root is missing or invalid")

    expected_names = {
        "SHA256SUMS",
        "execution_preboundary.json",
        "execution_success.json",
        "quantization_attempt_started.json",
    }

    observed_names = {
        path.name
        for path in EXECUTION_ROOT.iterdir()
    }

    if observed_names != expected_names:
        fail(
            "execution evidence inventory mismatch: "
            f"{sorted(observed_names)!r}"
        )

    for path in EXECUTION_ROOT.iterdir():
        if path.is_symlink() or not path.is_file():
            fail(f"invalid execution evidence entry: {path}")

    manifest = EXECUTION_ROOT / "SHA256SUMS"

    if sha256_file(manifest) != EXPECTED_EXECUTION_SUMS_SHA256:
        fail("execution SHA256SUMS identity mismatch")

    subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=EXECUTION_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def inspect_artifact_tree() -> tuple[tuple[str, int, str], ...]:
    if not ARTIFACT_ROOT.is_dir() or ARTIFACT_ROOT.is_symlink():
        fail("artifact root is missing or invalid")

    observed: list[tuple[str, int, str]] = []
    total_bytes = 0

    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if path.is_symlink():
            fail(f"artifact symbolic link is not allowed: {path}")

        if path.is_dir():
            continue

        if not path.is_file():
            fail(f"artifact entry is not a regular file: {path}")

        relative = path.relative_to(ARTIFACT_ROOT).as_posix()
        size_bytes = path.stat().st_size
        digest = sha256_file(path)

        observed.append((relative, size_bytes, digest))
        total_bytes += size_bytes

    expected = tuple(
        sorted(
            (
                relative,
                size_bytes,
                digest,
            )
            for relative, (size_bytes, digest)
            in EXPECTED_ARTIFACT_FILES.items()
        )
    )

    result = tuple(observed)

    if result != expected:
        fail(
            "artifact tree identity mismatch: "
            f"observed={result!r}"
        )

    if total_bytes != EXPECTED_ARTIFACT_TOTAL_BYTES:
        fail(
            "artifact total byte mismatch: "
            f"{total_bytes} != {EXPECTED_ARTIFACT_TOTAL_BYTES}"
        )

    return result


def run_preflight() -> tuple[tuple[str, int, str], ...]:
    offline_controls = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }

    for name, value in offline_controls.items():
        os.environ[name] = value

    for name, value in offline_controls.items():
        if os.environ.get(name) != value:
            fail(
                "offline control mismatch: "
                f"{name}={os.environ.get(name)!r}"
            )

    verify_repository_scope()
    verify_execution_evidence()
    artifact_tree = inspect_artifact_tree()

    if len(artifact_tree) != len(EXPECTED_ARTIFACT_FILES):
        fail(
            "artifact file-count mismatch after preflight: "
            f"{len(artifact_tree)}"
        )

    return artifact_tree


from huyawo_quant.contracts import FailureRecord


def run_transformers_compatibility_once() -> FailureRecord | None:
    run_preflight()

    expected_identity = build_quant_artifact_identity(
        artifact_root=ARTIFACT_ROOT,
        layout=EXPECTED_ARTIFACT_LAYOUT,
        serialization_format=EXPECTED_ARTIFACT_SERIALIZATION_FORMAT,
    )

    verify_quant_artifact_identity(
        artifact_root=ARTIFACT_ROOT,
        expected=expected_identity,
    )

    failure = validate_transformers_artifact_compatibility(
        artifact_root=ARTIFACT_ROOT,
        expected=expected_identity,
        prompt=PROMPT,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=SEED,
        cuda_device_index=CUDA_DEVICE_INDEX,
    )

    return failure


def main() -> int:
    print("TRANSFORMERS_WRAPPER_CALL_NEXT=true", flush=True)

    result = run_transformers_compatibility_once()

    if result is not None:
        if not isinstance(result, FailureRecord):
            print("ERROR: invalid compatibility result type", flush=True)
            return 3

        print("TRANSFORMERS_ARTIFACT_VALIDATION=FAIL", flush=True)
        print("FAILURE_RECORD=" + result.model_dump_json(), flush=True)
        return 2

    print("TRANSFORMERS_ARTIFACT_VALIDATION=PASS", flush=True)
    print("FAILURE_RECORD_PRESENT=false", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
