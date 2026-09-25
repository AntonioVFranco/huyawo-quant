"""Tests for Hugging Face model and tokenizer identity resolution."""

from types import SimpleNamespace
from typing import cast

import pytest
from huggingface_hub import HfApi
from huyawo_quant.contracts import ModelIdentity, TokenizerIdentity
from huyawo_quant.identity import (
    InvalidHuggingFaceRevisionError,
    resolve_huggingface_model_identity,
    resolve_huggingface_tokenizer_identity,
)

_VALID_SHA = "0123456789abcdef0123456789abcdef01234567"


class RecordingApi:
    """Minimal model-info API double for revision-resolution tests."""

    def __init__(
        self,
        *,
        sha: str | None = _VALID_SHA,
        error: Exception | None = None,
    ) -> None:
        self.sha = sha
        self.error = error
        self.calls: list[tuple[str, str | None, bool | str | None]] = []

    def model_info(
        self,
        repo_id: str,
        revision: str | None = None,
        *,
        token: bool | str | None = None,
    ) -> object:
        self.calls.append((repo_id, revision, token))

        if self.error is not None:
            raise self.error

        return SimpleNamespace(sha=self.sha)


def _as_hf_api(api: RecordingApi) -> HfApi:
    return cast(HfApi, api)


def test_resolve_model_identity_returns_expected_contract() -> None:
    api = RecordingApi()

    identity = resolve_huggingface_model_identity(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "main",
        api=_as_hf_api(api),
    )

    assert isinstance(identity, ModelIdentity)
    assert identity.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert identity.requested_revision == "main"
    assert identity.resolved_revision == _VALID_SHA
    assert api.calls == [("Qwen/Qwen2.5-0.5B-Instruct", "main", False)]


def test_resolve_tokenizer_identity_returns_expected_contract() -> None:
    api = RecordingApi()

    identity = resolve_huggingface_tokenizer_identity(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "refs/pr/17",
        api=_as_hf_api(api),
    )

    assert isinstance(identity, TokenizerIdentity)
    assert identity.tokenizer_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert identity.requested_revision == "refs/pr/17"
    assert identity.resolved_revision == _VALID_SHA
    assert api.calls == [("Qwen/Qwen2.5-0.5B-Instruct", "refs/pr/17", False)]


@pytest.mark.parametrize(
    "resolved_revision",
    [
        None,
        "",
        "a" * 39,
        "a" * 41,
        "g" * 40,
        "0123456789abcdef0123456789abcdef0123456-",
    ],
)
def test_resolver_rejects_missing_or_malformed_sha(
    resolved_revision: str | None,
) -> None:
    api = RecordingApi(sha=resolved_revision)

    with pytest.raises(
        InvalidHuggingFaceRevisionError,
        match="missing or malformed commit SHA",
    ):
        resolve_huggingface_model_identity(
            "Qwen/Qwen2.5-0.5B-Instruct",
            "main",
            api=_as_hf_api(api),
        )


def test_resolver_accepts_uppercase_hex_sha_without_normalizing() -> None:
    resolved_revision = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
    api = RecordingApi(sha=resolved_revision)

    identity = resolve_huggingface_model_identity(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "main",
        api=_as_hf_api(api),
    )

    assert identity.resolved_revision == resolved_revision


def test_resolver_propagates_upstream_failure_without_fallback() -> None:
    api = RecordingApi(error=RuntimeError("hub unavailable"))

    with pytest.raises(RuntimeError, match="hub unavailable"):
        resolve_huggingface_model_identity(
            "Qwen/Qwen2.5-0.5B-Instruct",
            "main",
            api=_as_hf_api(api),
        )

    assert api.calls == [("Qwen/Qwen2.5-0.5B-Instruct", "main", False)]


@pytest.mark.parametrize(
    ("repo_id", "requested_revision"),
    [
        ("", "main"),
        ("   ", "main"),
        ("Qwen/Qwen2.5-0.5B-Instruct", ""),
        ("Qwen/Qwen2.5-0.5B-Instruct", "   "),
    ],
)
def test_resolver_rejects_empty_input_before_network_call(
    repo_id: str,
    requested_revision: str,
) -> None:
    api = RecordingApi()

    with pytest.raises(ValueError):
        resolve_huggingface_model_identity(
            repo_id,
            requested_revision,
            api=_as_hf_api(api),
        )

    assert api.calls == []
