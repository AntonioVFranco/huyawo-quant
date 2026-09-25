"""Tests for deterministic baseline model loading."""

from typing import cast

import pytest
import torch
from huyawo_quant.baseline import BaselineDtype, load_baseline
from huyawo_quant.contracts import ModelIdentity, TokenizerIdentity
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

_MODEL_SHA = "0123456789abcdef0123456789abcdef01234567"
_TOKENIZER_SHA = "89abcdef0123456789abcdef0123456789abcdef"


class RecordingModel:
    """Minimal model double that records device placement and eval calls."""

    def __init__(self) -> None:
        self.to_calls: list[torch.device] = []
        self.eval_calls = 0

    def to(self, device: torch.device) -> "RecordingModel":
        self.to_calls.append(device)
        return self

    def eval(self) -> "RecordingModel":
        self.eval_calls += 1
        return self


class RecordingTokenizer:
    """Minimal tokenizer double for loader tests."""


def _identities() -> tuple[ModelIdentity, TokenizerIdentity]:
    return (
        ModelIdentity(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            requested_revision="main",
            resolved_revision=_MODEL_SHA,
        ),
        TokenizerIdentity(
            tokenizer_id="Qwen/Qwen2.5-0.5B-Instruct",
            requested_revision="tokenizer-main",
            resolved_revision=_TOKENIZER_SHA,
        ),
    )


def _patch_valid_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


def _unexpected_loader(*args: object, **kwargs: object) -> object:
    raise AssertionError("Loader must not be called")


@pytest.mark.parametrize(
    ("dtype", "expected_torch_dtype"),
    [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
    ],
)
def test_load_baseline_uses_immutable_identity_and_explicit_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
    dtype: BaselineDtype,
    expected_torch_dtype: torch.dtype,
) -> None:
    model_identity, tokenizer_identity = _identities()
    model_double = RecordingModel()
    tokenizer_double = RecordingTokenizer()
    tokenizer_calls: list[tuple[str, dict[str, object]]] = []
    model_calls: list[tuple[str, dict[str, object]]] = []

    def fake_tokenizer_loader(
        repo_id: str,
        **kwargs: object,
    ) -> RecordingTokenizer:
        tokenizer_calls.append((repo_id, kwargs))
        return tokenizer_double

    def fake_model_loader(
        repo_id: str,
        **kwargs: object,
    ) -> RecordingModel:
        model_calls.append((repo_id, kwargs))
        return model_double

    _patch_valid_cuda(monkeypatch)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        staticmethod(fake_tokenizer_loader),
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(fake_model_loader),
    )

    model, tokenizer = load_baseline(
        model_identity,
        tokenizer_identity,
        dtype=dtype,
        cuda_device_index=0,
    )

    assert model is cast(PreTrainedModel, model_double)
    assert tokenizer is cast(PreTrainedTokenizerBase, tokenizer_double)

    assert tokenizer_calls == [
        (
            tokenizer_identity.tokenizer_id,
            {
                "revision": tokenizer_identity.resolved_revision,
                "token": False,
                "trust_remote_code": False,
            },
        )
    ]
    assert tokenizer_calls[0][1]["revision"] != tokenizer_identity.requested_revision

    assert len(model_calls) == 1
    model_repo_id, model_kwargs = model_calls[0]

    assert model_repo_id == model_identity.model_id
    assert model_kwargs["revision"] == model_identity.resolved_revision
    assert model_kwargs["revision"] != model_identity.requested_revision
    assert model_kwargs["token"] is False
    assert model_kwargs["trust_remote_code"] is False
    assert model_kwargs["use_safetensors"] is True
    assert model_kwargs["dtype"] is expected_torch_dtype
    assert "device_map" not in model_kwargs

    assert model_double.to_calls == [torch.device("cuda", 0)]
    assert model_double.eval_calls == 1


def test_unsupported_dtype_fails_before_runtime_or_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_identity, tokenizer_identity = _identities()

    def unexpected_cuda_check() -> bool:
        raise AssertionError("CUDA must not be inspected for an invalid dtype")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_check)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        staticmethod(_unexpected_loader),
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(_unexpected_loader),
    )

    with pytest.raises(ValueError, match="Unsupported baseline dtype"):
        load_baseline(
            model_identity,
            tokenizer_identity,
            dtype=cast(BaselineDtype, "float32"),
            cuda_device_index=0,
        )


def test_cuda_unavailable_fails_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_identity, tokenizer_identity = _identities()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        staticmethod(_unexpected_loader),
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(_unexpected_loader),
    )

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        load_baseline(
            model_identity,
            tokenizer_identity,
            dtype="float16",
            cuda_device_index=0,
        )


@pytest.mark.parametrize("cuda_device_index", [-1, 1])
def test_invalid_cuda_device_fails_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    cuda_device_index: int,
) -> None:
    model_identity, tokenizer_identity = _identities()

    _patch_valid_cuda(monkeypatch)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        staticmethod(_unexpected_loader),
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(_unexpected_loader),
    )

    with pytest.raises(ValueError, match="Invalid CUDA device index"):
        load_baseline(
            model_identity,
            tokenizer_identity,
            dtype="float16",
            cuda_device_index=cuda_device_index,
        )


def test_invalid_model_identity_fails_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, tokenizer_identity = _identities()

    def unexpected_cuda_check() -> bool:
        raise AssertionError("CUDA must not be inspected for invalid identity")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_check)

    with pytest.raises(
        TypeError,
        match="model_identity must be a ModelIdentity",
    ):
        load_baseline(
            cast(ModelIdentity, object()),
            tokenizer_identity,
            dtype="float16",
            cuda_device_index=0,
        )


def test_invalid_tokenizer_identity_fails_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_identity, _ = _identities()

    def unexpected_cuda_check() -> bool:
        raise AssertionError("CUDA must not be inspected for invalid identity")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_check)

    with pytest.raises(
        TypeError,
        match="tokenizer_identity must be a TokenizerIdentity",
    ):
        load_baseline(
            model_identity,
            cast(TokenizerIdentity, object()),
            dtype="float16",
            cuda_device_index=0,
        )


def test_tokenizer_failure_propagates_without_model_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_identity, tokenizer_identity = _identities()
    tokenizer_calls = 0

    def failing_tokenizer_loader(
        repo_id: str,
        **kwargs: object,
    ) -> object:
        nonlocal tokenizer_calls
        tokenizer_calls += 1
        raise RuntimeError("tokenizer load failed")

    _patch_valid_cuda(monkeypatch)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        staticmethod(failing_tokenizer_loader),
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(_unexpected_loader),
    )

    with pytest.raises(RuntimeError, match="tokenizer load failed"):
        load_baseline(
            model_identity,
            tokenizer_identity,
            dtype="bfloat16",
            cuda_device_index=0,
        )

    assert tokenizer_calls == 1


def test_model_failure_propagates_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_identity, tokenizer_identity = _identities()
    tokenizer_double = RecordingTokenizer()
    model_calls = 0

    def fake_tokenizer_loader(
        repo_id: str,
        **kwargs: object,
    ) -> RecordingTokenizer:
        return tokenizer_double

    def failing_model_loader(
        repo_id: str,
        **kwargs: object,
    ) -> object:
        nonlocal model_calls
        model_calls += 1
        raise RuntimeError("model load failed")

    _patch_valid_cuda(monkeypatch)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        staticmethod(fake_tokenizer_loader),
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(failing_model_loader),
    )

    with pytest.raises(RuntimeError, match="model load failed"):
        load_baseline(
            model_identity,
            tokenizer_identity,
            dtype="float16",
            cuda_device_index=0,
        )

    assert model_calls == 1


def test_boolean_cuda_device_index_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_identity, tokenizer_identity = _identities()

    def unexpected_cuda_check() -> bool:
        raise AssertionError("CUDA must not be inspected for invalid device type")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_check)

    with pytest.raises(TypeError, match="cuda_device_index must be an integer"):
        load_baseline(
            model_identity,
            tokenizer_identity,
            dtype="float16",
            cuda_device_index=cast(int, True),
        )
