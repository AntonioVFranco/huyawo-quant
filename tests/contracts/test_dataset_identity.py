"""Tests for dataset identity contracts."""

import pytest
from huyawo_quant.contracts import DatasetIdentity
from pydantic import ValidationError


def test_dataset_identity_serializes_with_config_name() -> None:
    identity = DatasetIdentity(
        dataset_id="nyu-mll/glue",
        config_name="sst2",
        split="train",
        requested_revision="main",
        resolved_revision="0123456789abcdef",
    )

    assert identity.model_dump(mode="json") == {
        "source": "huggingface",
        "dataset_id": "nyu-mll/glue",
        "config_name": "sst2",
        "split": "train",
        "requested_revision": "main",
        "resolved_revision": "0123456789abcdef",
    }


def test_dataset_identity_accepts_missing_config_name() -> None:
    identity = DatasetIdentity(
        dataset_id="cornell-movie-review-data/rotten_tomatoes",
        split="train",
        requested_revision="main",
        resolved_revision="0123456789abcdef",
    )

    assert identity.config_name is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("dataset_id", ""),
        ("dataset_id", "   "),
        ("split", ""),
        ("split", "   "),
        ("requested_revision", ""),
        ("requested_revision", "   "),
        ("resolved_revision", ""),
        ("resolved_revision", "   "),
    ],
)
def test_dataset_identity_rejects_empty_required_fields(
    field_name: str,
    value: str,
) -> None:
    payload = {
        "dataset_id": "nyu-mll/glue",
        "config_name": "sst2",
        "split": "train",
        "requested_revision": "main",
        "resolved_revision": "0123456789abcdef",
    }
    payload[field_name] = value

    with pytest.raises(ValidationError):
        DatasetIdentity.model_validate(payload)


@pytest.mark.parametrize("config_name", ["", "   "])
def test_dataset_identity_rejects_empty_config_name(config_name: str) -> None:
    with pytest.raises(ValidationError):
        DatasetIdentity(
            dataset_id="nyu-mll/glue",
            config_name=config_name,
            split="train",
            requested_revision="main",
            resolved_revision="0123456789abcdef",
        )


def test_dataset_identity_uses_strict_typing() -> None:
    with pytest.raises(ValidationError):
        DatasetIdentity.model_validate(
            {
                "dataset_id": 123,
                "split": "train",
                "requested_revision": "main",
                "resolved_revision": "0123456789abcdef",
            }
        )


def test_dataset_identity_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DatasetIdentity.model_validate(
            {
                "dataset_id": "nyu-mll/glue",
                "config_name": "sst2",
                "split": "train",
                "requested_revision": "main",
                "resolved_revision": "0123456789abcdef",
                "seed": 42,
            }
        )


def test_dataset_identity_is_immutable() -> None:
    identity = DatasetIdentity(
        dataset_id="nyu-mll/glue",
        config_name="sst2",
        split="train",
        requested_revision="main",
        resolved_revision="0123456789abcdef",
    )

    with pytest.raises(ValidationError):
        identity.__setattr__("split", "validation")
