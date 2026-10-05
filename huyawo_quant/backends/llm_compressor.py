"""LLM Compressor backend adapter boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from importlib.metadata import (
    PackageNotFoundError,
)
from importlib.metadata import (
    version as distribution_version,
)
from pathlib import Path
from typing import Literal, cast

from huyawo_quant.contracts import FailureRecord, QuantPlan

LLM_COMPRESSOR_BACKEND = "llm_compressor"
LLM_COMPRESSOR_DISTRIBUTION = "llmcompressor"
LLM_COMPRESSOR_SUPPORTED_VERSION = "0.14.0"
COMPRESSED_TENSORS_DISTRIBUTION = "compressed-tensors"

FailureStage = Literal[
    "model_load",
    "calibration",
    "quantization",
    "artifact_save",
    "unknown",
]
OneshotRunner = Callable[..., object]

_SCHEME_MIN_COMPUTE_CAPABILITY: dict[
    str,
    tuple[int, int],
] = {
    "w4a16": (7, 5),
    "w8a16": (7, 5),
    "w8a8_int8": (7, 5),
    "w8a8_fp8": (8, 9),
}

_SUPPORTED_ALGORITHMS = frozenset(
    {
        "rtn",
        "gptq",
        "awq",
    }
)


def get_llm_compressor_version() -> str | None:
    """Return the installed LLM Compressor version, if available."""

    try:
        return distribution_version(LLM_COMPRESSOR_DISTRIBUTION)
    except PackageNotFoundError:
        return None


def _require_supported_llm_compressor_version() -> str:
    installed_version = get_llm_compressor_version()

    if installed_version is None:
        raise RuntimeError("LLM Compressor is not installed")

    if installed_version != LLM_COMPRESSOR_SUPPORTED_VERSION:
        raise RuntimeError(
            "unsupported installed LLM Compressor version: "
            f"{installed_version} != "
            f"{LLM_COMPRESSOR_SUPPORTED_VERSION}"
        )

    return installed_version


def _require_distribution_version(
    distribution: str,
) -> str:
    try:
        return distribution_version(distribution)
    except PackageNotFoundError as error:
        raise RuntimeError(f"required distribution is not installed: {distribution}") from error


def get_llm_compressor_backend_metadata() -> dict[str, str]:
    """Return backend-owned version and artifact-format metadata."""

    backend_version = get_llm_compressor_version()

    if backend_version is None:
        raise RuntimeError("LLM Compressor is not installed")

    compressed_tensors_version = _require_distribution_version(COMPRESSED_TENSORS_DISTRIBUTION)

    return {
        "backend": LLM_COMPRESSOR_BACKEND,
        "backend_distribution": (LLM_COMPRESSOR_DISTRIBUTION),
        "backend_version": backend_version,
        "artifact_format": "compressed-tensors",
        "compressed_tensors_version": (compressed_tensors_version),
    }


def llm_compressor_scheme_min_compute_capability() -> dict[str, tuple[int, int]]:
    """Return documented minimum compute capability per supported scheme."""

    return dict(_SCHEME_MIN_COMPUTE_CAPABILITY)


def validate_llm_compressor_plan(
    plan: QuantPlan,
) -> None:
    """Validate backend-specific plan constraints without executing work."""

    if plan.backend != LLM_COMPRESSOR_BACKEND:
        raise ValueError("QuantPlan backend must be llm_compressor")

    installed_version = _require_supported_llm_compressor_version()

    if plan.backend_version != installed_version:
        raise ValueError(
            "QuantPlan backend_version does not match "
            "the installed LLM Compressor version: "
            f"{plan.backend_version} != {installed_version}"
        )

    if plan.recipe.algorithm not in _SUPPORTED_ALGORITHMS:
        raise ValueError(
            "QuantRecipe algorithm is not supported by "
            "the LLM Compressor adapter: "
            f"{plan.recipe.algorithm}"
        )

    minimum = _SCHEME_MIN_COMPUTE_CAPABILITY[plan.recipe.scheme]

    observed = (
        plan.hardware.compute_capability_major,
        plan.hardware.compute_capability_minor,
    )

    if observed < minimum:
        raise ValueError(
            "QuantRecipe scheme is unsupported on the "
            "declared hardware by LLM Compressor: "
            f"scheme={plan.recipe.scheme}, "
            f"required={minimum[0]}.{minimum[1]}, "
            f"observed={observed[0]}.{observed[1]}"
        )


def build_llm_compressor_oneshot_kwargs(
    *,
    plan: QuantPlan,
    native_recipe: object,
    model_source: str | Path | None = None,
    dataset: object | None = None,
    output_dir: str | Path | None = None,
    save_compressed: bool = True,
    num_calibration_samples: int | None = None,
    max_seq_length: int | None = None,
) -> dict[str, object]:
    """Translate Huyawo execution facts to the native oneshot boundary."""

    validate_llm_compressor_plan(plan)

    if native_recipe is None:
        raise ValueError("native_recipe must not be None")

    resolved_model_source = str(model_source) if model_source is not None else plan.model.model_id

    kwargs: dict[str, object] = {
        "model": resolved_model_source,
        "model_revision": plan.model.resolved_revision,
        "recipe": native_recipe,
        "dataset": dataset,
        "save_compressed": save_compressed,
        "output_dir": (str(output_dir) if output_dir is not None else None),
    }

    if num_calibration_samples is not None:
        if num_calibration_samples <= 0:
            raise ValueError("num_calibration_samples must be greater than zero")

        kwargs["num_calibration_samples"] = num_calibration_samples

    if max_seq_length is not None:
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be greater than zero")

        kwargs["max_seq_length"] = max_seq_length

    return kwargs


def _load_oneshot_runner() -> OneshotRunner:
    _require_supported_llm_compressor_version()

    try:
        module = import_module(LLM_COMPRESSOR_DISTRIBUTION)
    except ModuleNotFoundError as error:
        raise RuntimeError("LLM Compressor is not importable") from error

    candidate = getattr(
        module,
        "oneshot",
        None,
    )

    if not callable(candidate):
        raise RuntimeError("LLM Compressor does not expose a callable oneshot")

    return cast(
        OneshotRunner,
        candidate,
    )


def execute_llm_compressor_oneshot(
    *,
    kwargs: Mapping[str, object],
    runner: OneshotRunner | None = None,
) -> object:
    """Execute the native oneshot boundary without altering Huyawo contracts."""

    selected_runner = runner if runner is not None else _load_oneshot_runner()

    return selected_runner(**dict(kwargs))


def translate_llm_compressor_failure(
    error: Exception,
    *,
    stage: FailureStage,
) -> FailureRecord:
    """Translate a backend exception into a Huyawo-owned failure record."""

    if isinstance(error, ValueError):
        category = "llm_compressor_validation_error"
    elif isinstance(error, OSError):
        category = "llm_compressor_io_error"
    elif isinstance(error, RuntimeError):
        category = "llm_compressor_runtime_error"
    else:
        category = "llm_compressor_error"

    detail = str(error).strip()

    if not detail:
        detail = "<no message>"

    return FailureRecord(
        stage=stage,
        category=category,
        message=(f"{type(error).__name__}: {detail}"),
    )
