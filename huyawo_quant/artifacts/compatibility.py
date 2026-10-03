from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, cast

from huyawo_quant.artifacts.identity import verify_quant_artifact_identity
from huyawo_quant.contracts import FailureRecord, QuantArtifactIdentity

__all__ = ["validate_transformers_artifact_compatibility"]


def _load_runtime_components() -> tuple[Any, Any, Any]:
    torch_module = import_module("torch")
    transformers_module = import_module("transformers")

    auto_model = transformers_module.AutoModelForCausalLM
    auto_tokenizer = transformers_module.AutoTokenizer

    return (
        torch_module,
        auto_model,
        auto_tokenizer,
    )


def _require_integer(
    name: str,
    value: object,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")

    return value


def _validate_inputs(
    *,
    expected: QuantArtifactIdentity,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    seed: int,
    cuda_device_index: int,
) -> None:
    if not isinstance(
        expected,
        QuantArtifactIdentity,
    ):
        raise TypeError("expected must be a QuantArtifactIdentity")

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    if not prompt.strip():
        raise ValueError("prompt must contain non-whitespace text")

    _require_integer(
        "max_input_tokens",
        max_input_tokens,
        minimum=1,
    )
    _require_integer(
        "max_new_tokens",
        max_new_tokens,
        minimum=1,
    )
    _require_integer(
        "seed",
        seed,
        minimum=0,
    )
    _require_integer(
        "cuda_device_index",
        cuda_device_index,
        minimum=0,
    )


def _format_exception(
    error: Exception,
) -> str:
    detail = str(error).strip()

    if not detail:
        detail = "<no message>"

    return f"{type(error).__name__}: {detail}"


def _translate_artifact_load_failure(
    error: Exception,
    *,
    category: str,
) -> FailureRecord:
    return FailureRecord(
        stage="artifact_load",
        category=category,
        message=_format_exception(error),
    )


def _resolve_cuda_device(
    torch_module: Any,
    cuda_device_index: int,
) -> Any:
    cuda = torch_module.cuda

    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device_count = cuda.device_count()

    if isinstance(
        device_count,
        bool,
    ) or not isinstance(
        device_count,
        int,
    ):
        raise RuntimeError("torch.cuda.device_count must return an integer")

    if cuda_device_index < 0 or cuda_device_index >= device_count:
        raise ValueError(
            "Invalid CUDA device index: "
            f"{cuda_device_index}; "
            f"available device count: {device_count}"
        )

    return torch_module.device(
        "cuda",
        cuda_device_index,
    )


def _run_transformers_compatibility_smoke(
    *,
    artifact_root: Path,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    seed: int,
    cuda_device_index: int,
) -> None:
    (
        torch_module,
        auto_model,
        auto_tokenizer,
    ) = _load_runtime_components()

    device = _resolve_cuda_device(
        torch_module,
        cuda_device_index,
    )

    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)

    tokenizer = auto_tokenizer.from_pretrained(
        str(artifact_root),
        local_files_only=True,
        token=False,
        trust_remote_code=False,
    )

    if tokenizer is None:
        raise RuntimeError("Transformers tokenizer loading returned None")

    model = auto_model.from_pretrained(
        str(artifact_root),
        local_files_only=True,
        token=False,
        trust_remote_code=False,
        use_safetensors=True,
        weights_only=True,
        dtype="auto",
        device_map=cuda_device_index,
    )

    if model is None:
        raise RuntimeError("Transformers model loading returned None")

    model.eval()

    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )

    to_device = getattr(
        encoded,
        "to",
        None,
    )

    if not callable(to_device):
        raise RuntimeError("Tokenizer output must support device transfer")

    encoded = to_device(device)

    try:
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Tokenizer output must contain input_ids and attention_mask") from error

    tensor_type = getattr(
        torch_module,
        "Tensor",
        None,
    )

    if (
        not isinstance(tensor_type, type)
        or not isinstance(
            input_ids,
            tensor_type,
        )
        or not isinstance(
            attention_mask,
            tensor_type,
        )
    ):
        raise RuntimeError("Tokenizer output tensors are invalid")

    input_ids_tensor = cast(Any, input_ids)
    attention_mask_tensor = cast(Any, attention_mask)

    input_shape = tuple(int(dimension) for dimension in input_ids_tensor.shape)
    attention_shape = tuple(int(dimension) for dimension in attention_mask_tensor.shape)

    if len(input_shape) != 2 or input_shape[0] != 1 or input_shape[1] <= 0:
        raise RuntimeError("Tokenizer input_ids must have shape (1, positive_sequence_length)")

    if attention_shape != input_shape:
        raise RuntimeError("Tokenizer attention_mask shape must match input_ids")

    input_token_count = input_shape[1]

    if input_token_count > max_input_tokens:
        raise ValueError(
            "Tokenized compatibility prompt exceeds max_input_tokens: "
            f"{input_token_count} > {max_input_tokens}"
        )

    with torch_module.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )

    if not isinstance(
        generated,
        tensor_type,
    ):
        raise RuntimeError("model.generate must return a torch.Tensor")

    generated_tensor = cast(Any, generated)

    generated_shape = tuple(int(dimension) for dimension in generated_tensor.shape)

    if len(generated_shape) != 2 or generated_shape[0] != 1:
        raise RuntimeError("model.generate output must have rank 2 and batch size 1")

    generated_token_count = generated_shape[1] - input_token_count

    if generated_token_count < 1:
        raise RuntimeError("model.generate produced no new tokens")

    if generated_token_count > max_new_tokens:
        raise RuntimeError("model.generate exceeded max_new_tokens")

    if not torch_module.equal(
        generated_tensor[:, :input_token_count],
        input_ids_tensor,
    ):
        raise RuntimeError("model.generate output does not preserve the prompt prefix")


def validate_transformers_artifact_compatibility(
    *,
    artifact_root: str | Path,
    expected: QuantArtifactIdentity,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    seed: int,
    cuda_device_index: int,
) -> FailureRecord | None:
    """Validate one artifact through a bounded local-only Transformers smoke."""

    _validate_inputs(
        expected=expected,
        prompt=prompt,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cuda_device_index=cuda_device_index,
    )

    root = Path(artifact_root)

    try:
        verify_quant_artifact_identity(
            artifact_root=root,
            expected=expected,
        )
    except Exception as error:
        return _translate_artifact_load_failure(
            error,
            category="artifact_identity_precheck_error",
        )

    primary_error: Exception | None = None

    try:
        _run_transformers_compatibility_smoke(
            artifact_root=root,
            prompt=prompt,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cuda_device_index=cuda_device_index,
        )
    except Exception as error:
        primary_error = error

    postcheck_error: Exception | None = None

    try:
        verify_quant_artifact_identity(
            artifact_root=root,
            expected=expected,
        )
    except Exception as error:
        postcheck_error = error

    if primary_error is not None and postcheck_error is not None:
        return FailureRecord(
            stage="artifact_load",
            category=("transformers_artifact_load_and_integrity_error"),
            message=(
                "primary="
                f"{_format_exception(primary_error)}; "
                "post_validation="
                f"{_format_exception(postcheck_error)}"
            ),
        )

    if postcheck_error is not None:
        return _translate_artifact_load_failure(
            postcheck_error,
            category="artifact_identity_postcheck_error",
        )

    if primary_error is not None:
        return _translate_artifact_load_failure(
            primary_error,
            category="transformers_artifact_load_error",
        )

    return None
