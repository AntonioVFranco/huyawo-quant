"""Tests for calibration contracts."""

import json

import pytest
from huyawo_quant.contracts import (
    CalibrationManifest,
    CalibrationPreprocessing,
    DatasetIdentity,
    TokenizerIdentity,
)
from pydantic import ValidationError


def make_dataset_identity() -> DatasetIdentity:
    return DatasetIdentity(
        dataset_id="HuggingFaceH4/ultrachat_200k",
        config_name=None,
        split="train_sft",
        requested_revision="main",
        resolved_revision="0123456789abcdef",
    )


def make_tokenizer_identity() -> TokenizerIdentity:
    return TokenizerIdentity(
        tokenizer_id="Qwen/Qwen2.5-0.5B-Instruct",
        requested_revision="main",
        resolved_revision="fedcba9876543210",
    )


def make_preprocessing() -> CalibrationPreprocessing:
    return CalibrationPreprocessing(
        source_column="messages",
        render_mode="chat_template",
        add_generation_prompt=False,
        add_special_tokens=False,
        truncation=True,
    )


def make_manifest() -> CalibrationManifest:
    return CalibrationManifest(
        schema_version="1",
        dataset=make_dataset_identity(),
        tokenizer=make_tokenizer_identity(),
        preprocessing=make_preprocessing(),
        sample_count=3,
        sample_indices=(7, 2, 11),
        max_sequence_length=2048,
        token_budget=4096,
        seed=42,
        shuffle=True,
    )


def make_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "1",
        "dataset": make_dataset_identity(),
        "tokenizer": make_tokenizer_identity(),
        "preprocessing": make_preprocessing(),
        "sample_count": 3,
        "sample_indices": (7, 2, 11),
        "max_sequence_length": 2048,
        "token_budget": 4096,
        "seed": 42,
        "shuffle": True,
    }


def test_calibration_preprocessing_serializes_expected_fields() -> None:
    preprocessing = make_preprocessing()

    assert preprocessing.model_dump(mode="json") == {
        "source_column": "messages",
        "render_mode": "chat_template",
        "add_generation_prompt": False,
        "add_special_tokens": False,
        "truncation": True,
    }


@pytest.mark.parametrize("source_column", ["", "   "])
def test_calibration_preprocessing_rejects_empty_source_column(
    source_column: str,
) -> None:
    with pytest.raises(ValidationError):
        CalibrationPreprocessing(
            source_column=source_column,
            render_mode="raw_text",
            add_generation_prompt=False,
            add_special_tokens=True,
            truncation=True,
        )


def test_calibration_preprocessing_rejects_unknown_render_mode() -> None:
    with pytest.raises(ValidationError):
        CalibrationPreprocessing.model_validate(
            {
                "source_column": "text",
                "render_mode": "custom",
                "add_generation_prompt": False,
                "add_special_tokens": True,
                "truncation": True,
            }
        )


def test_raw_text_rejects_generation_prompt() -> None:
    with pytest.raises(
        ValidationError,
        match="raw_text preprocessing cannot add a generation prompt",
    ):
        CalibrationPreprocessing(
            source_column="text",
            render_mode="raw_text",
            add_generation_prompt=True,
            add_special_tokens=True,
            truncation=True,
        )


def test_chat_template_accepts_generation_prompt() -> None:
    preprocessing = CalibrationPreprocessing(
        source_column="messages",
        render_mode="chat_template",
        add_generation_prompt=True,
        add_special_tokens=False,
        truncation=True,
    )

    assert preprocessing.add_generation_prompt is True


def test_calibration_preprocessing_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CalibrationPreprocessing.model_validate(
            {
                "source_column": "text",
                "render_mode": "raw_text",
                "add_generation_prompt": False,
                "add_special_tokens": True,
                "truncation": True,
                "formatter": "custom",
            }
        )


def test_calibration_preprocessing_is_immutable() -> None:
    preprocessing = make_preprocessing()

    with pytest.raises(ValidationError):
        preprocessing.__setattr__("source_column", "text")


def test_calibration_manifest_serializes_expected_fields() -> None:
    manifest = make_manifest()

    assert manifest.model_dump(mode="json") == {
        "schema_version": "1",
        "dataset": {
            "source": "huggingface",
            "dataset_id": "HuggingFaceH4/ultrachat_200k",
            "config_name": None,
            "split": "train_sft",
            "requested_revision": "main",
            "resolved_revision": "0123456789abcdef",
        },
        "tokenizer": {
            "source": "huggingface",
            "tokenizer_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "requested_revision": "main",
            "resolved_revision": "fedcba9876543210",
        },
        "preprocessing": {
            "source_column": "messages",
            "render_mode": "chat_template",
            "add_generation_prompt": False,
            "add_special_tokens": False,
            "truncation": True,
        },
        "sample_count": 3,
        "sample_indices": [7, 2, 11],
        "max_sequence_length": 2048,
        "token_budget": 4096,
        "seed": 42,
        "shuffle": True,
    }


def test_calibration_manifest_json_round_trip_is_deterministic() -> None:
    manifest = make_manifest()

    encoded_once = manifest.model_dump_json()
    encoded_twice = manifest.model_dump_json()

    assert encoded_once == encoded_twice
    assert json.loads(encoded_once) == manifest.model_dump(mode="json")
    assert CalibrationManifest.model_validate_json(encoded_once) == manifest


@pytest.mark.parametrize("schema_version", ["", "   "])
def test_calibration_manifest_rejects_empty_schema_version(
    schema_version: str,
) -> None:
    payload = make_manifest_payload()
    payload["schema_version"] = schema_version

    with pytest.raises(ValidationError):
        CalibrationManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sample_count", 0),
        ("sample_count", -1),
        ("max_sequence_length", 0),
        ("max_sequence_length", -1),
        ("seed", -1),
    ],
)
def test_calibration_manifest_rejects_invalid_integer_bounds(
    field_name: str,
    value: int,
) -> None:
    payload = make_manifest_payload()
    payload[field_name] = value

    with pytest.raises(ValidationError):
        CalibrationManifest.model_validate(payload)


@pytest.mark.parametrize("token_budget", [0, -1])
def test_calibration_manifest_rejects_invalid_token_budget(
    token_budget: int,
) -> None:
    payload = make_manifest_payload()
    payload["token_budget"] = token_budget

    with pytest.raises(ValidationError):
        CalibrationManifest.model_validate(payload)


def test_calibration_manifest_accepts_missing_token_budget() -> None:
    payload = make_manifest_payload()
    payload["token_budget"] = None

    manifest = CalibrationManifest.model_validate(payload)

    assert manifest.token_budget is None


def test_calibration_manifest_rejects_empty_sample_indices() -> None:
    payload = make_manifest_payload()
    payload["sample_count"] = 1
    payload["sample_indices"] = ()

    with pytest.raises(ValidationError):
        CalibrationManifest.model_validate(payload)


def test_calibration_manifest_rejects_negative_sample_index() -> None:
    payload = make_manifest_payload()
    payload["sample_indices"] = (7, -1, 11)

    with pytest.raises(ValidationError):
        CalibrationManifest.model_validate(payload)


def test_calibration_manifest_rejects_duplicate_sample_indices() -> None:
    payload = make_manifest_payload()
    payload["sample_indices"] = (7, 7, 11)

    with pytest.raises(ValidationError, match="sample_indices must be unique"):
        CalibrationManifest.model_validate(payload)


def test_calibration_manifest_rejects_sample_count_mismatch() -> None:
    payload = make_manifest_payload()
    payload["sample_count"] = 2

    with pytest.raises(
        ValidationError,
        match="sample_count must equal len",
    ):
        CalibrationManifest.model_validate(payload)


def test_calibration_manifest_preserves_sample_order() -> None:
    manifest = make_manifest()

    assert manifest.sample_indices == (7, 2, 11)


def test_calibration_manifest_preserves_identity_values() -> None:
    manifest = make_manifest()

    assert manifest.dataset == make_dataset_identity()
    assert manifest.tokenizer == make_tokenizer_identity()


def test_calibration_manifest_rejects_extra_fields() -> None:
    payload = make_manifest_payload()
    payload["backend_kwargs"] = {"num_calibration_samples": 3}

    with pytest.raises(ValidationError):
        CalibrationManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sample_count", "3"),
        ("max_sequence_length", "2048"),
        ("seed", "42"),
        ("shuffle", 1),
    ],
)
def test_calibration_manifest_uses_strict_typing(
    field_name: str,
    value: object,
) -> None:
    payload = make_manifest_payload()
    payload[field_name] = value

    with pytest.raises(ValidationError):
        CalibrationManifest.model_validate(payload)


def test_calibration_manifest_is_immutable() -> None:
    manifest = make_manifest()

    with pytest.raises(ValidationError):
        manifest.__setattr__("seed", 7)


def test_calibration_manifest_sample_indices_are_immutable() -> None:
    manifest = make_manifest()

    with pytest.raises(ValidationError):
        manifest.__setattr__("sample_indices", (11, 2, 7))
