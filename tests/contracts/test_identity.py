"""Tests for model and tokenizer identity contracts."""

import pytest
from huyawo_quant.contracts import ModelIdentity, TokenizerIdentity
from pydantic import ValidationError


def test_model_identity_serializes_expected_fields() -> None:
    identity = ModelIdentity(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        requested_revision="main",
        resolved_revision="0123456789abcdef",
    )

    assert identity.model_dump(mode="json") == {
        "source": "huggingface",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "requested_revision": "main",
        "resolved_revision": "0123456789abcdef",
    }


def test_tokenizer_identity_serializes_expected_fields() -> None:
    identity = TokenizerIdentity(
        tokenizer_id="Qwen/Qwen2.5-3B-Instruct",
        requested_revision="main",
        resolved_revision="0123456789abcdef",
    )

    assert identity.model_dump(mode="json") == {
        "source": "huggingface",
        "tokenizer_id": "Qwen/Qwen2.5-3B-Instruct",
        "requested_revision": "main",
        "resolved_revision": "0123456789abcdef",
    }


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("model_id", ""),
        ("model_id", "   "),
        ("requested_revision", ""),
        ("requested_revision", "   "),
        ("resolved_revision", ""),
        ("resolved_revision", "   "),
    ],
)
def test_model_identity_rejects_empty_identity_fields(
    field_name: str,
    value: str,
) -> None:
    payload = {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "requested_revision": "main",
        "resolved_revision": "0123456789abcdef",
    }
    payload[field_name] = value

    with pytest.raises(ValidationError):
        ModelIdentity.model_validate(payload)


def test_model_identity_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        ModelIdentity.model_validate(
            {
                "source": "local",
                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                "requested_revision": "main",
                "resolved_revision": "0123456789abcdef",
            }
        )


def test_tokenizer_identity_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TokenizerIdentity.model_validate(
            {
                "tokenizer_id": "Qwen/Qwen2.5-3B-Instruct",
                "requested_revision": "main",
                "resolved_revision": "0123456789abcdef",
                "cache_path": "/tmp/tokenizer",
            }
        )
