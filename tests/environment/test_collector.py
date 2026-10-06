"""Tests for read-only environment fingerprint capture."""

import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
from huyawo_quant.environment import collector


def test_parse_single_gpu_fingerprint() -> None:
    fingerprints = collector._parse_gpu_fingerprints(
        (
            "0, NVIDIA RTX A4500, "
            "GPU-b389adcb-bc21-3a97-57cf-3280bd06f1be, "
            "00000000:01:00.0, 20470, 580.159.03\n"
        ),
        "8.6\n",
    )

    assert len(fingerprints) == 1

    gpu = fingerprints[0]
    assert gpu.observed_index == 0
    assert gpu.name == "NVIDIA RTX A4500"
    assert gpu.uuid == ("GPU-b389adcb-bc21-3a97-57cf-3280bd06f1be")
    assert gpu.pci_bus_id == "00000000:01:00.0"
    assert gpu.memory_total_mib == 20470
    assert gpu.driver_version == "580.159.03"
    assert gpu.compute_capability_major == 8
    assert gpu.compute_capability_minor == 6


def test_gpu_fingerprints_are_sorted_by_uuid() -> None:
    fingerprints = collector._parse_gpu_fingerprints(
        (
            "0, GPU B, GPU-b, 00000000:02:00.0, 100, 1.0\n"
            "1, GPU A, GPU-a, 00000000:01:00.0, 200, 1.0\n"
        ),
        "8.6\n8.0\n",
    )

    assert tuple(gpu.uuid for gpu in fingerprints) == (
        "GPU-a",
        "GPU-b",
    )
    assert tuple(gpu.observed_index for gpu in fingerprints) == (
        1,
        0,
    )


def test_gpu_parser_rejects_mismatched_row_counts() -> None:
    with pytest.raises(ValueError):
        collector._parse_gpu_fingerprints(
            ("0, GPU A, GPU-a, 00000000:01:00.0, 100, 1.0\n"),
            "8.6\n8.0\n",
        )


def test_gpu_parser_rejects_missing_devices() -> None:
    with pytest.raises(ValueError):
        collector._parse_gpu_fingerprints("", "")


def test_gpu_parser_rejects_invalid_column_count() -> None:
    with pytest.raises(ValueError):
        collector._parse_gpu_fingerprints(
            "0, GPU A, GPU-a\n",
            "8.6\n",
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "8",
        "8.x",
        "8.6.1",
    ],
)
def test_compute_capability_parser_rejects_invalid_values(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        collector._parse_compute_capability(value)


def test_nvcc_version_parser_matches_observed_output() -> None:
    release, version = collector._parse_nvcc_version(
        "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.8, V12.8.93\n"
    )

    assert release == "12.8"
    assert version == "12.8.93"


def test_nvcc_version_parser_rejects_unknown_output() -> None:
    with pytest.raises(ValueError):
        collector._parse_nvcc_version("unexpected nvcc output")


def test_cuda_toolkit_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "huyawo_quant.environment.collector.shutil.which",
        lambda command: None if command == "nvcc" else command,
    )

    assert collector._capture_cuda_toolkit() is None


def test_relevant_distributions_cover_quality_environment() -> None:
    required_quality_distributions = {
        "absl-py",
        "accelerate",
        "aiohappyeyeballs",
        "aiohttp",
        "aiosignal",
        "attrs",
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
        "joblib",
        "lm-eval",
        "lxml",
        "mbstrdecoder",
        "more-itertools",
        "multidict",
        "multiprocess",
        "narwhals",
        "nltk",
        "pandas",
        "pathvalidate",
        "peft",
        "portalocker",
        "propcache",
        "psutil",
        "pyarrow",
        "pytablewriter",
        "python-dateutil",
        "pytz",
        "requests",
        "rouge-score",
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
        "typepy",
        "urllib3",
        "word2number",
        "xxhash",
        "yarl",
    }

    assert required_quality_distributions <= set(collector.RELEVANT_DISTRIBUTIONS)
    assert len(collector.RELEVANT_DISTRIBUTIONS) == len(set(collector.RELEVANT_DISTRIBUTIONS))


def test_python_distribution_capture_preserves_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_distribution_version(name: str) -> str:
        if name == "pydantic":
            return "2.13.5"

        raise PackageNotFoundError

    monkeypatch.setattr(
        collector,
        "distribution_version",
        fake_distribution_version,
    )

    fingerprints = collector._capture_python_distributions(("torch", "pydantic"))

    assert tuple(item.name for item in fingerprints) == (
        "pydantic",
        "torch",
    )
    assert fingerprints[0].status == "INSTALLED"
    assert fingerprints[0].version == "2.13.5"
    assert fingerprints[1].status == "ABSENT"
    assert fingerprints[1].version is None


def test_command_failure_preserves_primary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        arguments: tuple[str, ...],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        del check
        del capture_output
        del text

        return subprocess.CompletedProcess(
            args=arguments,
            returncode=3,
            stdout="",
            stderr="observed failure",
        )

    monkeypatch.setattr(
        "huyawo_quant.environment.collector.subprocess.run",
        fake_run,
    )

    with pytest.raises(
        RuntimeError,
        match="observed failure",
    ):
        collector._run_command(("nvidia-smi", "-L"))


def test_execution_context_preserves_venv_interpreter_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "huyawo-quant"
    virtual_environment = project_root / ".venv"
    bin_directory = virtual_environment / "bin"
    bin_directory.mkdir(parents=True)

    system_python = tmp_path / "system-python"
    system_python.write_text("placeholder")

    python_executable = bin_directory / "python"
    python_executable.symlink_to(system_python)

    monkeypatch.chdir(project_root)
    monkeypatch.setenv(
        "VIRTUAL_ENV",
        str(virtual_environment),
    )
    monkeypatch.setattr(
        "huyawo_quant.environment.collector.sys.executable",
        str(python_executable),
    )

    (
        working_directory,
        captured_virtual_environment,
        captured_python_executable,
    ) = collector._validate_execution_context(project_root)

    assert working_directory == project_root
    assert captured_virtual_environment == virtual_environment
    assert captured_python_executable == python_executable
    assert captured_python_executable.resolve() == system_python.resolve()


def test_execution_context_legacy_rejects_sibling_virtual_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "huyawo-quant"
    virtual_environment = project_root / ".venv-runtime"
    bin_directory = virtual_environment / "bin"
    bin_directory.mkdir(parents=True)

    python_executable = bin_directory / "python"
    python_executable.write_text("placeholder")

    monkeypatch.chdir(project_root)
    monkeypatch.setenv(
        "VIRTUAL_ENV",
        str(virtual_environment),
    )
    monkeypatch.setattr(
        "huyawo_quant.environment.collector.sys.executable",
        str(python_executable),
    )

    with pytest.raises(
        RuntimeError,
        match="project-local \\.venv",
    ):
        collector._validate_execution_context(project_root)


def test_execution_context_accepts_explicit_sibling_virtual_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "huyawo-quant"
    virtual_environment = project_root / ".venv-runtime"
    bin_directory = virtual_environment / "bin"
    bin_directory.mkdir(parents=True)

    system_python = tmp_path / "system-python"
    system_python.write_text("placeholder")

    python_executable = bin_directory / "python"
    python_executable.symlink_to(system_python)

    monkeypatch.chdir(project_root)
    monkeypatch.setenv(
        "VIRTUAL_ENV",
        str(virtual_environment),
    )
    monkeypatch.setattr(
        "huyawo_quant.environment.collector.sys.executable",
        str(python_executable),
    )

    (
        working_directory,
        captured_virtual_environment,
        captured_python_executable,
    ) = collector._validate_execution_context(
        project_root,
        expected_virtual_environment=str(virtual_environment),
    )

    assert working_directory == project_root
    assert captured_virtual_environment == virtual_environment
    assert captured_python_executable == python_executable
    assert captured_python_executable.resolve() == system_python.resolve()


def test_capture_rejects_expected_virtual_environment_outside_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "huyawo-quant"
    virtual_environment = project_root / ".venv-runtime"
    outside_environment = tmp_path / "outside-runtime"

    bin_directory = virtual_environment / "bin"
    bin_directory.mkdir(parents=True)

    python_executable = bin_directory / "python"
    python_executable.write_text("placeholder")

    monkeypatch.chdir(project_root)
    monkeypatch.setenv(
        "VIRTUAL_ENV",
        str(virtual_environment),
    )
    monkeypatch.setattr(
        collector,
        "_resolve_project_root",
        lambda: project_root,
    )
    monkeypatch.setattr(
        "huyawo_quant.environment.collector.sys.executable",
        str(python_executable),
    )

    with pytest.raises(
        RuntimeError,
        match="expected virtual environment is outside",
    ):
        collector.capture_environment_fingerprint(
            expected_virtual_environment=outside_environment,
        )


def test_execution_context_rejects_project_root_as_expected_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "huyawo-quant"
    bin_directory = project_root / "bin"
    bin_directory.mkdir(parents=True)

    python_executable = bin_directory / "python"
    python_executable.write_text("placeholder")

    monkeypatch.chdir(project_root)
    monkeypatch.setenv(
        "VIRTUAL_ENV",
        str(project_root),
    )
    monkeypatch.setattr(
        "huyawo_quant.environment.collector.sys.executable",
        str(python_executable),
    )

    with pytest.raises(
        RuntimeError,
        match="must not equal the Huyawo Quant project root",
    ):
        collector._validate_execution_context(
            project_root,
            expected_virtual_environment=project_root,
        )


def test_execution_context_rejects_active_virtual_environment_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "huyawo-quant"
    active_environment = project_root / ".venv-active"
    expected_environment = project_root / ".venv-expected"

    active_bin_directory = active_environment / "bin"
    active_bin_directory.mkdir(parents=True)
    expected_environment.mkdir(parents=True)

    python_executable = active_bin_directory / "python"
    python_executable.write_text("placeholder")

    monkeypatch.chdir(project_root)
    monkeypatch.setenv(
        "VIRTUAL_ENV",
        str(active_environment),
    )
    monkeypatch.setattr(
        "huyawo_quant.environment.collector.sys.executable",
        str(python_executable),
    )

    with pytest.raises(
        RuntimeError,
        match="does not match expected virtual environment",
    ):
        collector._validate_execution_context(
            project_root,
            expected_virtual_environment=expected_environment,
        )


def test_execution_context_rejects_python_outside_active_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "huyawo-quant"
    virtual_environment = project_root / ".venv-runtime"
    virtual_environment.mkdir(parents=True)

    python_executable = tmp_path / "outside-python"
    python_executable.write_text("placeholder")

    monkeypatch.chdir(project_root)
    monkeypatch.setenv(
        "VIRTUAL_ENV",
        str(virtual_environment),
    )
    monkeypatch.setattr(
        "huyawo_quant.environment.collector.sys.executable",
        str(python_executable),
    )

    with pytest.raises(
        RuntimeError,
        match="Python executable is outside the active project virtual environment",
    ):
        collector._validate_execution_context(
            project_root,
            expected_virtual_environment=virtual_environment,
        )


def test_environment_fingerprint_serialization_shape_is_unchanged() -> None:
    properties = collector.EnvironmentFingerprint.model_json_schema()["properties"]

    assert tuple(properties) == (
        "captured_at",
        "project_root",
        "working_directory",
        "virtual_environment",
        "python_executable",
        "python_version",
        "python_implementation",
        "python_prefix",
        "python_base_prefix",
        "platform_system",
        "platform_release",
        "platform_machine",
        "platform_version",
        "cuda_visible_devices",
        "cuda_device_order",
        "python_distributions",
        "nvidia_smi_path",
        "nvidia_gpus",
        "cuda_toolkit",
    )
