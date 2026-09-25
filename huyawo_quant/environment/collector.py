"""Read-only environment and NVIDIA hardware fingerprint capture."""

import csv
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from io import StringIO
from pathlib import Path

from huyawo_quant.contracts.environment import (
    CudaToolkitFingerprint,
    EnvironmentFingerprint,
    NvidiaGpuFingerprint,
    PythonDistributionFingerprint,
)

RELEVANT_DISTRIBUTIONS = (
    "absl-py",
    "accelerate",
    "aiohappyeyeballs",
    "aiohttp",
    "aiosignal",
    "attrs",
    "bitsandbytes",
    "chardet",
    "charset-normalizer",
    "cloudpickle",
    "colorama",
    "dataproperty",
    "datasets",
    "defusedxml",
    "dill",
    "evaluate",
    "frozenlist",
    "fsspec",
    "gptqmodel",
    "huggingface-hub",
    "joblib",
    "llmcompressor",
    "lm-eval",
    "lxml",
    "mbstrdecoder",
    "more-itertools",
    "multidict",
    "multiprocess",
    "mypy",
    "narwhals",
    "nltk",
    "pandas",
    "pathvalidate",
    "peft",
    "portalocker",
    "propcache",
    "psutil",
    "pyarrow",
    "pydantic",
    "pytablewriter",
    "pytest",
    "python-dateutil",
    "pytz",
    "requests",
    "rouge-score",
    "ruff",
    "sacrebleu",
    "scikit-learn",
    "scipy",
    "six",
    "sqlitedict",
    "tabledata",
    "tabulate",
    "tcolorpy",
    "threadpoolctl",
    "tokenizers",
    "torch",
    "torchao",
    "transformers",
    "typepy",
    "urllib3",
    "vllm",
    "word2number",
    "xxhash",
    "yarl",
)


def _run_command(arguments: tuple[str, ...]) -> str:
    completed = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"command failed with status {completed.returncode}: {' '.join(arguments)}: {detail}"
        )

    return completed.stdout


def _parse_csv_rows(
    output: str,
    *,
    expected_columns: int,
) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []

    for raw_row in csv.reader(StringIO(output)):
        row = tuple(value.strip() for value in raw_row)

        if not any(row):
            continue

        if len(row) != expected_columns:
            raise ValueError(
                "unexpected NVIDIA query column count: "
                f"expected {expected_columns}, received {len(row)}"
            )

        rows.append(row)

    return tuple(rows)


def _parse_compute_capability(
    value: str,
) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", value)

    if match is None:
        raise ValueError(f"invalid NVIDIA compute capability: {value}")

    return int(match.group(1)), int(match.group(2))


def _parse_gpu_fingerprints(
    gpu_output: str,
    compute_output: str,
) -> tuple[NvidiaGpuFingerprint, ...]:
    gpu_rows = _parse_csv_rows(
        gpu_output,
        expected_columns=6,
    )
    compute_rows = _parse_csv_rows(
        compute_output,
        expected_columns=1,
    )

    if not gpu_rows:
        raise ValueError("nvidia-smi returned no NVIDIA GPUs")

    if len(gpu_rows) != len(compute_rows):
        raise ValueError(
            "NVIDIA GPU query and compute-capability query returned different row counts"
        )

    fingerprints: list[NvidiaGpuFingerprint] = []

    for gpu_row, compute_row in zip(
        gpu_rows,
        compute_rows,
        strict=True,
    ):
        (
            observed_index,
            name,
            uuid,
            pci_bus_id,
            memory_total_mib,
            driver_version,
        ) = gpu_row

        compute_major, compute_minor = _parse_compute_capability(compute_row[0])

        fingerprints.append(
            NvidiaGpuFingerprint(
                observed_index=int(observed_index),
                name=name,
                uuid=uuid,
                pci_bus_id=pci_bus_id,
                memory_total_mib=int(memory_total_mib),
                driver_version=driver_version,
                compute_capability_major=compute_major,
                compute_capability_minor=compute_minor,
            )
        )

    return tuple(
        sorted(
            fingerprints,
            key=lambda fingerprint: fingerprint.uuid,
        )
    )


def _capture_nvidia_gpus(
    nvidia_smi_path: str,
) -> tuple[NvidiaGpuFingerprint, ...]:
    gpu_output = _run_command(
        (
            nvidia_smi_path,
            "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    compute_output = _run_command(
        (
            nvidia_smi_path,
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
        )
    )

    return _parse_gpu_fingerprints(
        gpu_output,
        compute_output,
    )


def _parse_nvcc_version(
    output: str,
) -> tuple[str, str]:
    match = re.search(
        r"release\s+([0-9]+(?:\.[0-9]+)+),\s+"
        r"V([0-9]+(?:\.[0-9]+)+)",
        output,
    )

    if match is None:
        raise ValueError("nvcc output does not contain a parseable CUDA release and version")

    return match.group(1), match.group(2)


def _capture_cuda_toolkit() -> CudaToolkitFingerprint | None:
    nvcc_path = shutil.which("nvcc")

    if nvcc_path is None:
        return None

    output = _run_command((nvcc_path, "--version"))
    release, version = _parse_nvcc_version(output)

    return CudaToolkitFingerprint(
        nvcc_path=nvcc_path,
        release=release,
        version=version,
    )


def _capture_python_distributions(
    names: tuple[str, ...] = RELEVANT_DISTRIBUTIONS,
) -> tuple[PythonDistributionFingerprint, ...]:
    fingerprints: list[PythonDistributionFingerprint] = []

    for name in sorted(names):
        try:
            installed_version = distribution_version(name)
        except PackageNotFoundError:
            fingerprints.append(
                PythonDistributionFingerprint(
                    name=name,
                    status="ABSENT",
                )
            )
        else:
            fingerprints.append(
                PythonDistributionFingerprint(
                    name=name,
                    status="INSTALLED",
                    version=installed_version,
                )
            )

    return tuple(fingerprints)


def _resolve_project_root() -> Path:
    current = Path.cwd().resolve()

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "huyawo_quant").is_dir():
            return candidate

    raise RuntimeError(
        "could not resolve the Huyawo Quant project root from the current working directory"
    )


def _validate_execution_context(
    project_root: Path,
) -> tuple[Path, Path, Path]:
    working_directory = Path.cwd().resolve()

    if not working_directory.is_relative_to(project_root):
        raise RuntimeError("working directory is outside the Huyawo Quant project root")

    virtual_environment_value = os.environ.get("VIRTUAL_ENV")

    if virtual_environment_value is None:
        raise RuntimeError("VIRTUAL_ENV is not set for EnvironmentFingerprint capture")

    virtual_environment = Path(virtual_environment_value).resolve()
    expected_virtual_environment = (project_root / ".venv").resolve()

    if virtual_environment != expected_virtual_environment:
        raise RuntimeError("EnvironmentFingerprint capture requires the project-local .venv")

    python_executable = Path(os.path.abspath(sys.executable))

    if not python_executable.is_relative_to(virtual_environment):
        raise RuntimeError("Python executable is outside the active project virtual environment")

    return (
        working_directory,
        virtual_environment,
        python_executable,
    )


def capture_environment_fingerprint(
    *,
    captured_at: datetime | None = None,
) -> EnvironmentFingerprint:
    """Capture one validated fingerprint without mutating project state."""

    project_root = _resolve_project_root()
    (
        working_directory,
        virtual_environment,
        python_executable,
    ) = _validate_execution_context(project_root)

    nvidia_smi_path = shutil.which("nvidia-smi")

    if nvidia_smi_path is None:
        raise RuntimeError("nvidia-smi is required for NVIDIA environment capture")

    capture_time = captured_at if captured_at is not None else datetime.now(UTC)

    return EnvironmentFingerprint(
        captured_at=capture_time,
        project_root=str(project_root),
        working_directory=str(working_directory),
        virtual_environment=str(virtual_environment),
        python_executable=str(python_executable),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        python_prefix=str(Path(sys.prefix).resolve()),
        python_base_prefix=str(Path(sys.base_prefix).resolve()),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        platform_version=platform.version(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
        python_distributions=_capture_python_distributions(),
        nvidia_smi_path=nvidia_smi_path,
        nvidia_gpus=_capture_nvidia_gpus(nvidia_smi_path),
        cuda_toolkit=_capture_cuda_toolkit(),
    )
