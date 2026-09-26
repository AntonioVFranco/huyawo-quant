"""Thin lm-evaluation-harness configuration adapter."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import version as distribution_version
from importlib.resources import as_file, files
from re import fullmatch
from typing import cast

from lm_eval.tasks._yaml_loader import load_yaml  # type: ignore[import-untyped]

from huyawo_quant.contracts import DatasetIdentity, EvaluationProfile

_EXPECTED_LM_EVAL_VERSION = "0.4.13"
_EXPECTED_TASK = "hellaswag"
_EXPECTED_DATASET_ID = "Rowan/hellaswag"
_EXPECTED_DATASET_CONFIG = None
_EXPECTED_SPLIT = "validation"
_EXPECTED_OUTPUT_TYPE = "multiple_choice"
_EXPECTED_METADATA_VERSION = 1.0
_EXPECTED_METRIC_SIGNATURE = (
    ("acc", "mean", True),
    ("acc_norm", "mean", True),
)
_IMMUTABLE_REVISION_PATTERN = r"[0-9a-fA-F]{40}"

NativeTaskConfig = dict[str, object]


def _require_lm_eval_version() -> str:
    observed = distribution_version("lm-eval")

    if observed != _EXPECTED_LM_EVAL_VERSION:
        raise RuntimeError(
            f"Unexpected lm-eval version: {observed}; expected {_EXPECTED_LM_EVAL_VERSION}"
        )

    return observed


def _load_native_hellaswag_config() -> NativeTaskConfig:
    resource = files("lm_eval.tasks.hellaswag").joinpath("hellaswag.yaml")

    with as_file(resource) as path:
        loaded = load_yaml(
            path,
            resolve_func=True,
        )

    return cast(NativeTaskConfig, loaded)


def _validate_native_hellaswag_config(config: NativeTaskConfig) -> None:
    expected_fields: dict[str, object] = {
        "task": _EXPECTED_TASK,
        "dataset_path": _EXPECTED_DATASET_ID,
        "dataset_name": _EXPECTED_DATASET_CONFIG,
        "validation_split": _EXPECTED_SPLIT,
        "output_type": _EXPECTED_OUTPUT_TYPE,
    }

    for field_name, expected in expected_fields.items():
        observed = config.get(field_name)

        if observed != expected:
            raise RuntimeError(
                "Unexpected native HellaSwag configuration for "
                f"{field_name}: {observed!r}; expected {expected!r}"
            )

    metadata_value = config.get("metadata")

    if not isinstance(metadata_value, Mapping):
        raise RuntimeError("Native HellaSwag metadata must be a mapping")

    metadata = cast(Mapping[str, object], metadata_value)

    if metadata.get("version") != _EXPECTED_METADATA_VERSION:
        raise RuntimeError(
            f"Unexpected native HellaSwag metadata version: {metadata.get('version')!r}"
        )

    if not callable(config.get("process_docs")):
        raise RuntimeError("Native HellaSwag process_docs must resolve to a callable")

    metric_list_value = config.get("metric_list")

    if not isinstance(metric_list_value, list):
        raise RuntimeError("Native HellaSwag metric_list must be a list")

    metric_signature: list[tuple[object, object, object]] = []

    for entry_value in metric_list_value:
        if not isinstance(entry_value, Mapping):
            raise RuntimeError("Native HellaSwag metric entries must be mappings")

        entry = cast(Mapping[str, object], entry_value)

        metric_signature.append(
            (
                entry.get("metric"),
                entry.get("aggregation"),
                entry.get("higher_is_better"),
            )
        )

    if tuple(metric_signature) != _EXPECTED_METRIC_SIGNATURE:
        raise RuntimeError(
            f"Unexpected native HellaSwag metric configuration: {tuple(metric_signature)!r}"
        )


def _validate_dataset_identity(dataset_identity: DatasetIdentity) -> None:
    if not isinstance(dataset_identity, DatasetIdentity):
        raise TypeError("dataset_identity must be a DatasetIdentity")

    if dataset_identity.dataset_id != _EXPECTED_DATASET_ID:
        raise ValueError("dataset_identity must target Rowan/hellaswag")

    if dataset_identity.config_name is not None:
        raise ValueError("HellaSwag dataset config_name must be None")

    if dataset_identity.split != _EXPECTED_SPLIT:
        raise ValueError("HellaSwag dataset split must be validation")

    if (
        fullmatch(
            _IMMUTABLE_REVISION_PATTERN,
            dataset_identity.resolved_revision,
        )
        is None
    ):
        raise ValueError(
            "dataset_identity resolved_revision must be an immutable "
            "40-character hexadecimal commit SHA"
        )


def _validated_native_hellaswag_config() -> tuple[str, NativeTaskConfig]:
    benchmark_version = _require_lm_eval_version()
    config = _load_native_hellaswag_config()
    _validate_native_hellaswag_config(config)
    return benchmark_version, config


def build_hellaswag_evaluation_profile() -> EvaluationProfile:
    """Build the accepted HellaSwag quality-evaluation profile."""
    benchmark_version, _ = _validated_native_hellaswag_config()

    return EvaluationProfile(
        benchmark="lm-evaluation-harness",
        benchmark_version=benchmark_version,
        tasks=(_EXPECTED_TASK,),
        metrics=("acc", "acc_norm"),
        sample_limit=None,
        seed=42,
    )


def build_hellaswag_task_config(
    dataset_identity: DatasetIdentity,
) -> NativeTaskConfig:
    """Build a native HellaSwag task config bound to an immutable dataset revision."""
    _validate_dataset_identity(dataset_identity)
    _, native_config = _validated_native_hellaswag_config()

    task_config = dict(native_config)
    existing_dataset_kwargs = task_config.get("dataset_kwargs")

    if existing_dataset_kwargs is None:
        dataset_kwargs: dict[str, object] = {}
    elif isinstance(existing_dataset_kwargs, Mapping):
        dataset_kwargs = dict(cast(Mapping[str, object], existing_dataset_kwargs))
    else:
        raise RuntimeError("Native HellaSwag dataset_kwargs must be a mapping when present")

    existing_revision = dataset_kwargs.get("revision")

    if existing_revision is not None and existing_revision != dataset_identity.resolved_revision:
        raise RuntimeError(
            "Native HellaSwag dataset revision conflicts with the supplied DatasetIdentity"
        )

    dataset_kwargs["revision"] = dataset_identity.resolved_revision

    task_config["dataset_kwargs"] = dataset_kwargs
    task_config["num_fewshot"] = 0

    return task_config
