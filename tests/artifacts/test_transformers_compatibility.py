from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import huyawo_quant.artifacts.compatibility as compatibility
import pytest
import torch
from huyawo_quant.artifacts.compatibility import (
    validate_transformers_artifact_compatibility,
)
from huyawo_quant.contracts import (
    ArtifactFileIdentity,
    FailureRecord,
    QuantArtifactIdentity,
)


class _InferenceContext(AbstractContextManager[None]):
    def __init__(
        self,
        events: list[str],
    ) -> None:
        self._events = events

    def __enter__(self) -> None:
        self._events.append("inference.enter")
        return None

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self._events.append("inference.exit")
        return None


class _FakeCuda:
    def __init__(
        self,
        state: dict[str, Any],
        *,
        available: bool,
        device_count: int,
    ) -> None:
        self._state = state
        self._available = available
        self._device_count = device_count

    def is_available(self) -> bool:
        self._state["events"].append("cuda.is_available")
        return self._available

    def device_count(self) -> int:
        self._state["events"].append("cuda.device_count")
        return self._device_count

    def manual_seed_all(
        self,
        seed: int,
    ) -> None:
        self._state["events"].append("cuda.manual_seed_all")
        self._state["cuda_seed"] = seed


class _FakeTorch:
    Tensor = torch.Tensor

    def __init__(
        self,
        state: dict[str, Any],
        *,
        cuda_available: bool,
        cuda_device_count: int,
    ) -> None:
        self._state = state
        self.cuda = _FakeCuda(
            state,
            available=cuda_available,
            device_count=cuda_device_count,
        )

    def manual_seed(
        self,
        seed: int,
    ) -> object:
        self._state["events"].append("torch.manual_seed")
        self._state["torch_seed"] = seed
        return object()

    def device(
        self,
        device_type: str,
        index: int,
    ) -> str:
        value = f"{device_type}:{index}"
        self._state["events"].append(f"torch.device:{value}")
        return value

    def inference_mode(
        self,
    ) -> AbstractContextManager[None]:
        return _InferenceContext(self._state["events"])

    def equal(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> bool:
        return torch.equal(
            left,
            right,
        )


class _FakeEncoding(dict[str, torch.Tensor]):
    def __init__(
        self,
        *,
        state: dict[str, Any],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        super().__init__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        self._state = state

    def to(
        self,
        device: object,
    ) -> _FakeEncoding:
        self._state["events"].append(f"encoding.to:{device}")
        self._state["encoding_device"] = device
        return self


def _identity() -> QuantArtifactIdentity:
    return QuantArtifactIdentity(
        layout="huggingface_pretrained",
        serialization_format="safetensors",
        files=(
            ArtifactFileIdentity(
                relative_path="model.safetensors",
                sha256="0" * 64,
                size_bytes=1,
            ),
        ),
    )


def _install_verifier(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, Any],
    *,
    errors: tuple[
        Exception | None,
        ...,
    ] = (
        None,
        None,
    ),
) -> None:
    call_index = 0

    def fake_verify(
        *,
        artifact_root: str | Path,
        expected: QuantArtifactIdentity,
    ) -> None:
        nonlocal call_index

        state["events"].append("verify")
        state.setdefault(
            "verify_roots",
            [],
        ).append(Path(artifact_root))
        state.setdefault(
            "verify_expected",
            [],
        ).append(expected)

        error = errors[call_index] if call_index < len(errors) else None
        call_index += 1

        if error is not None:
            raise error

    monkeypatch.setattr(
        compatibility,
        "verify_quant_artifact_identity",
        fake_verify,
    )


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, Any],
    *,
    tokenizer_load_error: Exception | None = None,
    model_load_error: Exception | None = None,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    generated: torch.Tensor | None = None,
    cuda_available: bool = True,
    cuda_device_count: int = 1,
) -> None:
    effective_input_ids = (
        input_ids
        if input_ids is not None
        else torch.tensor(
            [[1, 2, 3]],
            dtype=torch.long,
        )
    )

    effective_attention_mask = (
        attention_mask if attention_mask is not None else torch.ones_like(effective_input_ids)
    )

    effective_generated = (
        generated
        if generated is not None
        else torch.tensor(
            [[1, 2, 3, 4, 5]],
            dtype=torch.long,
        )
    )

    fake_torch = _FakeTorch(
        state,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
    )

    class FakeTokenizer:
        def __call__(
            self,
            prompt: str,
            **kwargs: Any,
        ) -> _FakeEncoding:
            state["events"].append("tokenizer.call")
            state["prompt"] = prompt
            state["tokenizer_call_kwargs"] = kwargs

            return _FakeEncoding(
                state=state,
                input_ids=effective_input_ids,
                attention_mask=effective_attention_mask,
            )

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(
            path: str,
            **kwargs: Any,
        ) -> FakeTokenizer:
            state["events"].append("tokenizer.load")
            state["tokenizer_load_path"] = path
            state["tokenizer_load_kwargs"] = kwargs

            if tokenizer_load_error is not None:
                raise tokenizer_load_error

            return FakeTokenizer()

    class FakeModel:
        def eval(
            self,
        ) -> FakeModel:
            state["events"].append("model.eval")
            return self

        def generate(
            self,
            **kwargs: Any,
        ) -> torch.Tensor:
            state["events"].append("model.generate")
            state["generation_kwargs"] = kwargs
            return effective_generated

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(
            path: str,
            **kwargs: Any,
        ) -> FakeModel:
            state["events"].append("model.load")
            state["model_load_path"] = path
            state["model_load_kwargs"] = kwargs

            if model_load_error is not None:
                raise model_load_error

            return FakeModel()

    monkeypatch.setattr(
        compatibility,
        "_load_runtime_components",
        lambda: (
            fake_torch,
            FakeAutoModel,
            FakeAutoTokenizer,
        ),
    )


def _run(
    tmp_path: Path,
    *,
    prompt: str = "Hello",
    max_input_tokens: int = 8,
    max_new_tokens: int = 2,
    seed: int = 7,
    cuda_device_index: int = 0,
) -> FailureRecord | None:
    return validate_transformers_artifact_compatibility(
        artifact_root=tmp_path,
        expected=_identity(),
        prompt=prompt,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cuda_device_index=cuda_device_index,
    )


def test_validate_transformers_artifact_compatibility_binds_local_only_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "events": [],
    }

    _install_verifier(
        monkeypatch,
        state,
    )
    _install_runtime(
        monkeypatch,
        state,
    )

    result = _run(
        tmp_path,
    )

    assert result is None
    assert state["events"][0] == "verify"
    assert state["events"][-1] == "verify"
    assert state["events"].count("verify") == 2

    assert state["tokenizer_load_path"] == str(tmp_path)
    assert state["tokenizer_load_kwargs"] == {
        "local_files_only": True,
        "token": False,
        "trust_remote_code": False,
    }

    assert state["model_load_path"] == str(tmp_path)
    assert state["model_load_kwargs"] == {
        "local_files_only": True,
        "token": False,
        "trust_remote_code": False,
        "use_safetensors": True,
        "weights_only": True,
        "dtype": "auto",
        "device_map": 0,
    }

    assert state["prompt"] == "Hello"
    assert state["tokenizer_call_kwargs"] == {
        "add_special_tokens": True,
        "padding": False,
        "truncation": False,
        "return_attention_mask": True,
        "return_tensors": "pt",
    }

    generation_kwargs = state["generation_kwargs"]

    assert torch.equal(
        generation_kwargs["input_ids"],
        torch.tensor(
            [[1, 2, 3]],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        generation_kwargs["attention_mask"],
        torch.ones(
            (1, 3),
            dtype=torch.long,
        ),
    )

    assert {
        key: value
        for key, value in generation_kwargs.items()
        if key
        not in {
            "input_ids",
            "attention_mask",
        }
    } == {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": 2,
    }

    assert state["torch_seed"] == 7
    assert state["cuda_seed"] == 7
    assert state["encoding_device"] == "cuda:0"


@pytest.mark.parametrize(
    (
        "field",
        "value",
        "error_type",
        "match",
    ),
    (
        (
            "prompt",
            "",
            ValueError,
            "non-whitespace",
        ),
        (
            "prompt",
            "   ",
            ValueError,
            "non-whitespace",
        ),
        (
            "max_input_tokens",
            0,
            ValueError,
            "must be >= 1",
        ),
        (
            "max_new_tokens",
            0,
            ValueError,
            "must be >= 1",
        ),
        (
            "seed",
            -1,
            ValueError,
            "must be >= 0",
        ),
        (
            "cuda_device_index",
            -1,
            ValueError,
            "must be >= 0",
        ),
    ),
)
def test_operator_input_validation_happens_before_artifact_or_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    error_type: type[Exception],
    match: str,
) -> None:
    def forbidden_verify(
        **kwargs: Any,
    ) -> None:
        raise AssertionError("artifact verification must not run")

    def forbidden_runtime() -> tuple[Any, Any, Any]:
        raise AssertionError("runtime loading must not run")

    monkeypatch.setattr(
        compatibility,
        "verify_quant_artifact_identity",
        forbidden_verify,
    )
    monkeypatch.setattr(
        compatibility,
        "_load_runtime_components",
        forbidden_runtime,
    )

    arguments: dict[str, object] = {
        "artifact_root": tmp_path,
        "expected": _identity(),
        "prompt": "Hello",
        "max_input_tokens": 8,
        "max_new_tokens": 2,
        "seed": 7,
        "cuda_device_index": 0,
    }
    arguments[field] = value

    with pytest.raises(
        error_type,
        match=match,
    ):
        validate_transformers_artifact_compatibility(
            **arguments,  # type: ignore[arg-type]
        )


def test_precheck_failure_returns_artifact_load_failure_without_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "events": [],
    }

    _install_verifier(
        monkeypatch,
        state,
        errors=(ValueError("artifact identity mismatch"),),
    )

    def forbidden_runtime() -> tuple[Any, Any, Any]:
        raise AssertionError("runtime loading must not run")

    monkeypatch.setattr(
        compatibility,
        "_load_runtime_components",
        forbidden_runtime,
    )

    result = _run(
        tmp_path,
    )

    assert result is not None
    assert result.stage == "artifact_load"
    assert result.category == ("artifact_identity_precheck_error")
    assert result.message == ("ValueError: artifact identity mismatch")
    assert state["events"] == ["verify"]


def test_model_load_failure_is_translated_and_postverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "events": [],
    }

    _install_verifier(
        monkeypatch,
        state,
    )
    _install_runtime(
        monkeypatch,
        state,
        model_load_error=OSError("checkpoint cannot be loaded"),
    )

    result = _run(
        tmp_path,
    )

    assert result is not None
    assert result.stage == "artifact_load"
    assert result.category == ("transformers_artifact_load_error")
    assert result.message == ("OSError: checkpoint cannot be loaded")
    assert state["events"].count("verify") == 2
    assert state["events"][-1] == "verify"


def test_unavailable_cuda_is_translated_and_postverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "events": [],
    }

    _install_verifier(
        monkeypatch,
        state,
    )
    _install_runtime(
        monkeypatch,
        state,
        cuda_available=False,
    )

    result = _run(
        tmp_path,
    )

    assert result is not None
    assert result.category == ("transformers_artifact_load_error")
    assert result.message == ("RuntimeError: CUDA is not available")
    assert "tokenizer.load" not in state["events"]
    assert state["events"].count("verify") == 2


def test_input_token_bound_fails_before_generation_and_postverifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "events": [],
    }

    _install_verifier(
        monkeypatch,
        state,
    )
    _install_runtime(
        monkeypatch,
        state,
        input_ids=torch.tensor(
            [[1, 2, 3, 4]],
            dtype=torch.long,
        ),
        attention_mask=torch.ones(
            (1, 4),
            dtype=torch.long,
        ),
    )

    result = _run(
        tmp_path,
        max_input_tokens=3,
    )

    assert result is not None
    assert result.category == ("transformers_artifact_load_error")
    assert "exceeds max_input_tokens" in result.message
    assert "model.generate" not in state["events"]
    assert state["events"].count("verify") == 2


def test_generation_requires_at_least_one_new_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "events": [],
    }

    generated = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    _install_verifier(
        monkeypatch,
        state,
    )
    _install_runtime(
        monkeypatch,
        state,
        generated=generated,
    )

    result = _run(
        tmp_path,
    )

    assert result is not None
    assert result.category == ("transformers_artifact_load_error")
    assert "produced no new tokens" in result.message
    assert state["events"].count("verify") == 2


def test_generation_rejects_more_than_max_new_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "events": [],
    }

    generated = torch.tensor(
        [[1, 2, 3, 4, 5, 6]],
        dtype=torch.long,
    )

    _install_verifier(
        monkeypatch,
        state,
    )
    _install_runtime(
        monkeypatch,
        state,
        generated=generated,
    )

    result = _run(
        tmp_path,
        max_new_tokens=2,
    )

    assert result is not None
    assert result.category == ("transformers_artifact_load_error")
    assert "exceeded max_new_tokens" in result.message


def test_generation_requires_preserved_prompt_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "events": [],
    }

    generated = torch.tensor(
        [[9, 2, 3, 4]],
        dtype=torch.long,
    )

    _install_verifier(
        monkeypatch,
        state,
    )
    _install_runtime(
        monkeypatch,
        state,
        generated=generated,
    )

    result = _run(
        tmp_path,
    )

    assert result is not None
    assert result.category == ("transformers_artifact_load_error")
    assert "does not preserve the prompt prefix" in result.message


def test_postcheck_failure_is_preserved_after_successful_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "events": [],
    }

    _install_verifier(
        monkeypatch,
        state,
        errors=(
            None,
            RuntimeError("artifact mutated"),
        ),
    )
    _install_runtime(
        monkeypatch,
        state,
    )

    result = _run(
        tmp_path,
    )

    assert result is not None
    assert result.stage == "artifact_load"
    assert result.category == ("artifact_identity_postcheck_error")
    assert result.message == ("RuntimeError: artifact mutated")


def test_primary_and_postcheck_failures_are_both_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "events": [],
    }

    _install_verifier(
        monkeypatch,
        state,
        errors=(
            None,
            RuntimeError("artifact mutated"),
        ),
    )
    _install_runtime(
        monkeypatch,
        state,
        model_load_error=OSError("broken model"),
    )

    result = _run(
        tmp_path,
    )

    assert result is not None
    assert result.stage == "artifact_load"
    assert result.category == ("transformers_artifact_load_and_integrity_error")
    assert result.message == (
        "primary=OSError: broken model; post_validation=RuntimeError: artifact mutated"
    )
