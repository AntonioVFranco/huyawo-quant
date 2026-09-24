"""Tests for hardware and runtime target contracts."""

import pytest
from huyawo_quant.contracts import HardwareTarget, RuntimeTarget
from pydantic import ValidationError


def test_hardware_target_serializes_expected_fields() -> None:
    target = HardwareTarget(
        device_name="NVIDIA RTX A4500",
        device_count=1,
        memory_bytes_per_device=21_474_836_480,
        compute_capability_major=8,
        compute_capability_minor=6,
    )

    assert target.model_dump(mode="json") == {
        "vendor": "nvidia",
        "accelerator": "cuda",
        "device_name": "NVIDIA RTX A4500",
        "device_count": 1,
        "memory_bytes_per_device": 21_474_836_480,
        "compute_capability_major": 8,
        "compute_capability_minor": 6,
    }


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("device_count", 0),
        ("device_count", -1),
        ("memory_bytes_per_device", 0),
        ("memory_bytes_per_device", -1),
        ("compute_capability_major", -1),
        ("compute_capability_minor", -1),
    ],
)
def test_hardware_target_rejects_invalid_numeric_fields(
    field_name: str,
    value: int,
) -> None:
    payload: dict[str, object] = {
        "device_name": "NVIDIA RTX A4500",
        "device_count": 1,
        "memory_bytes_per_device": 21_474_836_480,
        "compute_capability_major": 8,
        "compute_capability_minor": 6,
    }
    payload[field_name] = value

    with pytest.raises(ValidationError):
        HardwareTarget.model_validate(payload)


@pytest.mark.parametrize("device_name", ["", "   "])
def test_hardware_target_rejects_empty_device_name(device_name: str) -> None:
    with pytest.raises(ValidationError):
        HardwareTarget(
            device_name=device_name,
            device_count=1,
            memory_bytes_per_device=21_474_836_480,
            compute_capability_major=8,
            compute_capability_minor=6,
        )


def test_hardware_target_rejects_unknown_vendor() -> None:
    with pytest.raises(ValidationError):
        HardwareTarget.model_validate(
            {
                "vendor": "amd",
                "accelerator": "cuda",
                "device_name": "Example GPU",
                "device_count": 1,
                "memory_bytes_per_device": 1,
                "compute_capability_major": 8,
                "compute_capability_minor": 0,
            }
        )


def test_hardware_target_rejects_unknown_accelerator() -> None:
    with pytest.raises(ValidationError):
        HardwareTarget.model_validate(
            {
                "vendor": "nvidia",
                "accelerator": "rocm",
                "device_name": "Example GPU",
                "device_count": 1,
                "memory_bytes_per_device": 1,
                "compute_capability_major": 8,
                "compute_capability_minor": 0,
            }
        )


def test_hardware_target_uses_strict_typing() -> None:
    with pytest.raises(ValidationError):
        HardwareTarget.model_validate(
            {
                "device_name": "NVIDIA RTX A4500",
                "device_count": "1",
                "memory_bytes_per_device": 21_474_836_480,
                "compute_capability_major": 8,
                "compute_capability_minor": 6,
            }
        )


def test_hardware_target_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HardwareTarget.model_validate(
            {
                "device_name": "NVIDIA RTX A4500",
                "device_count": 1,
                "memory_bytes_per_device": 21_474_836_480,
                "compute_capability_major": 8,
                "compute_capability_minor": 6,
                "driver_version": "550.142",
            }
        )


def test_hardware_target_is_immutable() -> None:
    target = HardwareTarget(
        device_name="NVIDIA RTX A4500",
        device_count=1,
        memory_bytes_per_device=21_474_836_480,
        compute_capability_major=8,
        compute_capability_minor=6,
    )

    with pytest.raises(ValidationError):
        target.__setattr__("device_count", 2)


@pytest.mark.parametrize("runtime", ["vllm", "transformers"])
def test_runtime_target_accepts_initial_runtimes(runtime: str) -> None:
    target = RuntimeTarget.model_validate(
        {
            "runtime": runtime,
            "version": "0.10.0",
        }
    )

    assert target.model_dump(mode="json") == {
        "runtime": runtime,
        "version": "0.10.0",
    }


@pytest.mark.parametrize("version", ["", "   "])
def test_runtime_target_rejects_empty_version(version: str) -> None:
    with pytest.raises(ValidationError):
        RuntimeTarget(runtime="vllm", version=version)


def test_runtime_target_rejects_unknown_runtime() -> None:
    with pytest.raises(ValidationError):
        RuntimeTarget.model_validate(
            {
                "runtime": "sglang",
                "version": "1.0.0",
            }
        )


def test_runtime_target_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RuntimeTarget.model_validate(
            {
                "runtime": "vllm",
                "version": "0.10.0",
                "tensor_parallel_size": 1,
            }
        )
