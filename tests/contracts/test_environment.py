"""Tests for observed environment fingerprint contracts."""

from datetime import UTC, datetime

import pytest
from huyawo_quant.contracts import (
    CudaToolkitFingerprint,
    EnvironmentFingerprint,
    NvidiaGpuFingerprint,
    PythonDistributionFingerprint,
)
from pydantic import ValidationError

CAPTURED_AT = datetime(
    2026,
    9,
    25,
    13,
    0,
    tzinfo=UTC,
)


def make_distribution(
    *,
    name: str,
    status: str = "INSTALLED",
    version: str | None = "1.0.0",
) -> PythonDistributionFingerprint:
    payload: dict[str, object] = {
        "name": name,
        "status": status,
        "version": version,
    }

    return PythonDistributionFingerprint.model_validate(payload)


def make_gpu(
    *,
    uuid: str = "GPU-b389adcb-bc21-3a97-57cf-3280bd06f1be",
    observed_index: int = 0,
) -> NvidiaGpuFingerprint:
    return NvidiaGpuFingerprint(
        observed_index=observed_index,
        name="NVIDIA RTX A4500",
        uuid=uuid,
        pci_bus_id="00000000:01:00.0",
        memory_total_mib=20470,
        driver_version="580.159.03",
        compute_capability_major=8,
        compute_capability_minor=6,
    )


def make_cuda_toolkit() -> CudaToolkitFingerprint:
    return CudaToolkitFingerprint(
        nvcc_path="/usr/local/cuda/bin/nvcc",
        release="12.8",
        version="12.8.93",
    )


def make_environment() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        captured_at=CAPTURED_AT,
        project_root="/workspace/huyawo-quant",
        working_directory="/workspace/huyawo-quant",
        virtual_environment="/workspace/huyawo-quant/.venv",
        python_executable=("/workspace/huyawo-quant/.venv/bin/python"),
        python_version="3.12.3",
        python_implementation="CPython",
        python_prefix="/workspace/huyawo-quant/.venv",
        python_base_prefix="/usr",
        platform_system="Linux",
        platform_release="6.8.0-1055-nvidia",
        platform_machine="x86_64",
        platform_version=("#58~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Wed May 27 15:45:00 UTC 2"),
        cuda_visible_devices=None,
        cuda_device_order=None,
        python_distributions=(
            make_distribution(
                name="bitsandbytes",
                status="ABSENT",
                version=None,
            ),
            make_distribution(
                name="gptqmodel",
                status="ABSENT",
                version=None,
            ),
            make_distribution(
                name="huggingface-hub",
                status="ABSENT",
                version=None,
            ),
            make_distribution(
                name="llmcompressor",
                status="ABSENT",
                version=None,
            ),
            make_distribution(
                name="mypy",
                version="2.3.1",
            ),
            make_distribution(
                name="pydantic",
                version="2.13.5",
            ),
            make_distribution(
                name="pytest",
                version="9.1.1",
            ),
            make_distribution(
                name="ruff",
                version="0.16.8",
            ),
            make_distribution(
                name="torch",
                status="ABSENT",
                version=None,
            ),
            make_distribution(
                name="torchao",
                status="ABSENT",
                version=None,
            ),
            make_distribution(
                name="transformers",
                status="ABSENT",
                version=None,
            ),
            make_distribution(
                name="vllm",
                status="ABSENT",
                version=None,
            ),
        ),
        nvidia_smi_path="/usr/bin/nvidia-smi",
        nvidia_gpus=(make_gpu(),),
        cuda_toolkit=make_cuda_toolkit(),
    )


def test_environment_fingerprint_serializes_observed_probe() -> None:
    payload = make_environment().model_dump(mode="json")

    assert payload["captured_at"] == "2026-09-25T13:00:00Z"
    assert payload["project_root"] == "/workspace/huyawo-quant"
    assert payload["python_version"] == "3.12.3"
    assert payload["cuda_visible_devices"] is None
    assert payload["cuda_device_order"] is None
    assert payload["nvidia_smi_path"] == "/usr/bin/nvidia-smi"

    gpu = payload["nvidia_gpus"][0]
    assert gpu["name"] == "NVIDIA RTX A4500"
    assert gpu["memory_total_mib"] == 20470
    assert gpu["driver_version"] == "580.159.03"
    assert gpu["compute_capability_major"] == 8
    assert gpu["compute_capability_minor"] == 6

    toolkit = payload["cuda_toolkit"]
    assert toolkit is not None
    assert toolkit["release"] == "12.8"
    assert toolkit["version"] == "12.8.93"


def test_installed_distribution_requires_version() -> None:
    with pytest.raises(ValidationError):
        make_distribution(
            name="pydantic",
            status="INSTALLED",
            version=None,
        )


def test_absent_distribution_forbids_version() -> None:
    with pytest.raises(ValidationError):
        make_distribution(
            name="torch",
            status="ABSENT",
            version="2.0.0",
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", ""),
        ("name", "   "),
        ("version", ""),
        ("version", "   "),
    ],
)
def test_distribution_rejects_empty_text(
    field_name: str,
    value: str,
) -> None:
    payload: dict[str, object] = {
        "name": "pydantic",
        "status": "INSTALLED",
        "version": "2.13.5",
    }
    payload[field_name] = value

    with pytest.raises(ValidationError):
        PythonDistributionFingerprint.model_validate(payload)


def test_gpu_rejects_negative_observed_index() -> None:
    payload = make_gpu().model_dump()
    payload["observed_index"] = -1

    with pytest.raises(ValidationError):
        NvidiaGpuFingerprint.model_validate(payload)


def test_gpu_rejects_nonpositive_memory() -> None:
    payload = make_gpu().model_dump()
    payload["memory_total_mib"] = 0

    with pytest.raises(ValidationError):
        NvidiaGpuFingerprint.model_validate(payload)


@pytest.mark.parametrize(
    "field_name",
    [
        "compute_capability_major",
        "compute_capability_minor",
    ],
)
def test_gpu_rejects_negative_compute_capability(
    field_name: str,
) -> None:
    payload = make_gpu().model_dump()
    payload[field_name] = -1

    with pytest.raises(ValidationError):
        NvidiaGpuFingerprint.model_validate(payload)


def test_cuda_toolkit_rejects_relative_nvcc_path() -> None:
    with pytest.raises(ValidationError):
        CudaToolkitFingerprint(
            nvcc_path="usr/local/cuda/bin/nvcc",
            release="12.8",
            version="12.8.93",
        )


def test_environment_accepts_absent_cuda_toolkit() -> None:
    payload = make_environment().model_dump()
    payload["cuda_toolkit"] = None

    environment = EnvironmentFingerprint.model_validate(payload)

    assert environment.cuda_toolkit is None


def test_environment_preserves_empty_cuda_environment_values() -> None:
    payload = make_environment().model_dump()
    payload["cuda_visible_devices"] = ""
    payload["cuda_device_order"] = ""

    environment = EnvironmentFingerprint.model_validate(payload)

    assert environment.cuda_visible_devices == ""
    assert environment.cuda_device_order == ""


def test_environment_rejects_naive_capture_timestamp() -> None:
    payload = make_environment().model_dump()
    payload["captured_at"] = datetime(2026, 9, 25, 13, 0)

    with pytest.raises(ValidationError):
        EnvironmentFingerprint.model_validate(payload)


@pytest.mark.parametrize(
    "field_name",
    [
        "project_root",
        "working_directory",
        "virtual_environment",
        "python_executable",
        "python_prefix",
        "python_base_prefix",
        "nvidia_smi_path",
    ],
)
def test_environment_requires_absolute_paths(
    field_name: str,
) -> None:
    payload = make_environment().model_dump()
    payload[field_name] = "relative/path"

    with pytest.raises(ValidationError):
        EnvironmentFingerprint.model_validate(payload)


def test_environment_rejects_duplicate_distribution_names() -> None:
    payload = make_environment().model_dump()
    distribution = make_distribution(
        name="pydantic",
        version="2.13.5",
    )
    payload["python_distributions"] = (
        distribution,
        distribution,
    )

    with pytest.raises(ValidationError):
        EnvironmentFingerprint.model_validate(payload)


def test_environment_rejects_unsorted_distributions() -> None:
    payload = make_environment().model_dump()
    payload["python_distributions"] = (
        make_distribution(
            name="pytest",
            version="9.1.1",
        ),
        make_distribution(
            name="pydantic",
            version="2.13.5",
        ),
    )

    with pytest.raises(ValidationError):
        EnvironmentFingerprint.model_validate(payload)


def test_environment_accepts_sorted_distributions() -> None:
    environment = make_environment()

    names = tuple(distribution.name for distribution in environment.python_distributions)

    assert names == tuple(sorted(names))


def test_environment_rejects_duplicate_gpu_uuids() -> None:
    payload = make_environment().model_dump()
    gpu = make_gpu()
    payload["nvidia_gpus"] = (gpu, gpu)

    with pytest.raises(ValidationError):
        EnvironmentFingerprint.model_validate(payload)


def test_environment_rejects_unsorted_gpu_uuids() -> None:
    payload = make_environment().model_dump()
    payload["nvidia_gpus"] = (
        make_gpu(uuid="GPU-b"),
        make_gpu(uuid="GPU-a", observed_index=1),
    )

    with pytest.raises(ValidationError):
        EnvironmentFingerprint.model_validate(payload)


def test_environment_accepts_sorted_gpu_uuids() -> None:
    payload = make_environment().model_dump()
    payload["nvidia_gpus"] = (
        make_gpu(uuid="GPU-a"),
        make_gpu(uuid="GPU-b", observed_index=1),
    )

    environment = EnvironmentFingerprint.model_validate(payload)

    assert tuple(gpu.uuid for gpu in environment.nvidia_gpus) == (
        "GPU-a",
        "GPU-b",
    )


def test_environment_requires_nonempty_distribution_tuple() -> None:
    payload = make_environment().model_dump()
    payload["python_distributions"] = ()

    with pytest.raises(ValidationError):
        EnvironmentFingerprint.model_validate(payload)


def test_environment_requires_nonempty_gpu_tuple() -> None:
    payload = make_environment().model_dump()
    payload["nvidia_gpus"] = ()

    with pytest.raises(ValidationError):
        EnvironmentFingerprint.model_validate(payload)


def test_environment_uses_strict_distribution_tuple_typing() -> None:
    payload = make_environment().model_dump()
    payload["python_distributions"] = list(make_environment().python_distributions)

    with pytest.raises(ValidationError):
        EnvironmentFingerprint.model_validate(payload)


def test_environment_uses_strict_gpu_tuple_typing() -> None:
    payload = make_environment().model_dump()
    payload["nvidia_gpus"] = [make_gpu()]

    with pytest.raises(ValidationError):
        EnvironmentFingerprint.model_validate(payload)


def test_environment_rejects_extra_fields() -> None:
    payload = make_environment().model_dump()
    payload["hostname"] = "should-not-be-recorded"

    with pytest.raises(ValidationError):
        EnvironmentFingerprint.model_validate(payload)


def test_environment_is_immutable() -> None:
    environment = make_environment()

    with pytest.raises(ValidationError):
        environment.__setattr__(
            "python_version",
            "3.13.0",
        )
