"""Hugging Face model and tokenizer identity resolution."""

from __future__ import annotations

import re
from typing import Final

from huggingface_hub import HfApi

from huyawo_quant.contracts import ModelIdentity, TokenizerIdentity

_COMMIT_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-fA-F]{40}")


class InvalidHuggingFaceRevisionError(ValueError):
    """Raised when Hugging Face returns an invalid resolved revision."""


def _require_non_empty(field_name: str, value: str) -> None:
    if not value or value.isspace():
        raise ValueError(f"{field_name} must contain non-whitespace characters")


def _resolve_revision_sha(
    repo_id: str,
    requested_revision: str,
    *,
    api: HfApi | None = None,
) -> str:
    _require_non_empty("repo_id", repo_id)
    _require_non_empty("requested_revision", requested_revision)

    client = HfApi() if api is None else api

    info = client.model_info(
        repo_id=repo_id,
        revision=requested_revision,
        token=False,
    )
    resolved_revision = info.sha

    if (
        not isinstance(resolved_revision, str)
        or _COMMIT_SHA_PATTERN.fullmatch(resolved_revision) is None
    ):
        raise InvalidHuggingFaceRevisionError(
            "Hugging Face returned a missing or malformed commit SHA"
        )

    return resolved_revision


def resolve_huggingface_model_identity(
    model_id: str,
    requested_revision: str,
    *,
    api: HfApi | None = None,
) -> ModelIdentity:
    """Resolve a public Hugging Face model revision to an immutable identity."""

    resolved_revision = _resolve_revision_sha(
        model_id,
        requested_revision,
        api=api,
    )

    return ModelIdentity(
        model_id=model_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
    )


def resolve_huggingface_tokenizer_identity(
    tokenizer_id: str,
    requested_revision: str,
    *,
    api: HfApi | None = None,
) -> TokenizerIdentity:
    """Resolve a public Hugging Face tokenizer revision to an immutable identity."""

    resolved_revision = _resolve_revision_sha(
        tokenizer_id,
        requested_revision,
        api=api,
    )

    return TokenizerIdentity(
        tokenizer_id=tokenizer_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
    )
