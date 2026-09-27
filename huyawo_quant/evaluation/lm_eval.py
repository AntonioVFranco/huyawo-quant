"""Thin lm-evaluation-harness configuration adapter."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version as distribution_version
from importlib.resources import as_file, files
from math import isfinite
from re import fullmatch
from typing import cast

from lm_eval.tasks._yaml_loader import load_yaml  # type: ignore[import-untyped]

from huyawo_quant.contracts import (
    BenchmarkMetric,
    BenchmarkResult,
    DatasetIdentity,
    EnvironmentFingerprint,
    EvaluationProfile,
    ModelIdentity,
    TokenizerIdentity,
)

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


_EXPECTED_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_EXPECTED_TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_EXPECTED_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_EXPECTED_DTYPE = "bfloat16"
_EXPECTED_DEVICE = "cuda:0"
_EXPECTED_BATCH_SIZE = 1
_EXPECTED_SEED = 42
_EXPECTED_USE_FAST_TOKENIZER = True


def _validate_model_identity(
    model_identity: ModelIdentity,
) -> None:
    if not isinstance(model_identity, ModelIdentity):
        raise TypeError("model_identity must be a ModelIdentity")

    if model_identity.model_id != _EXPECTED_MODEL_ID:
        raise ValueError("model_identity must target Qwen/Qwen2.5-0.5B-Instruct")

    if model_identity.resolved_revision != _EXPECTED_MODEL_REVISION:
        raise ValueError(
            "model_identity resolved_revision must match the accepted immutable model revision"
        )


def _validate_tokenizer_identity(
    tokenizer_identity: TokenizerIdentity,
) -> None:
    if not isinstance(
        tokenizer_identity,
        TokenizerIdentity,
    ):
        raise TypeError("tokenizer_identity must be a TokenizerIdentity")

    if tokenizer_identity.tokenizer_id != _EXPECTED_TOKENIZER_ID:
        raise ValueError("tokenizer_identity must target Qwen/Qwen2.5-0.5B-Instruct")

    if tokenizer_identity.resolved_revision != _EXPECTED_MODEL_REVISION:
        raise ValueError(
            "tokenizer_identity resolved_revision must match "
            "the accepted immutable tokenizer revision"
        )


def _validate_evaluation_limit(
    limit: int | None,
) -> None:
    if limit is None:
        return

    if type(limit) is not int or limit <= 0:
        raise ValueError(
            "limit must be None for authoritative evaluation "
            "or a positive integer for smoke evaluation"
        )


def build_hellaswag_model_args(
    model_identity: ModelIdentity,
    tokenizer_identity: TokenizerIdentity,
) -> dict[str, str | int | float | bool]:
    """Build deterministic HFLM arguments for the accepted quality baseline."""
    _validate_model_identity(model_identity)
    _validate_tokenizer_identity(tokenizer_identity)

    return {
        "pretrained": model_identity.model_id,
        "revision": model_identity.resolved_revision,
        "tokenizer": tokenizer_identity.tokenizer_id,
        "dtype": _EXPECTED_DTYPE,
        "trust_remote_code": False,
        "use_fast_tokenizer": _EXPECTED_USE_FAST_TOKENIZER,
    }


def build_hellaswag_simple_evaluate_kwargs(
    model_identity: ModelIdentity,
    tokenizer_identity: TokenizerIdentity,
    dataset_identity: DatasetIdentity,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """Build deterministic simple_evaluate kwargs without executing evaluation."""
    _validate_evaluation_limit(limit)

    model_args = build_hellaswag_model_args(
        model_identity,
        tokenizer_identity,
    )

    task_config = build_hellaswag_task_config(dataset_identity)

    return {
        "model": "hf",
        "model_args": model_args,
        "tasks": [task_config],
        "num_fewshot": 0,
        "batch_size": _EXPECTED_BATCH_SIZE,
        "device": _EXPECTED_DEVICE,
        "limit": limit,
        "log_samples": True,
        "apply_chat_template": False,
        "predict_only": False,
        "random_seed": _EXPECTED_SEED,
        "numpy_random_seed": _EXPECTED_SEED,
        "torch_random_seed": _EXPECTED_SEED,
        "fewshot_random_seed": _EXPECTED_SEED,
    }


def _require_mapping(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")

    return cast(Mapping[str, object], value)


def _qualified_callable_name(
    value: object,
    *,
    field_name: str,
) -> str:
    if not callable(value):
        raise ValueError(f"{field_name} must be callable")

    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)

    if not isinstance(module, str) or not module:
        raise ValueError(f"{field_name} callable must declare __module__")

    if not isinstance(qualname, str) or not qualname:
        raise ValueError(f"{field_name} callable must declare __qualname__")

    return f"{module}.{qualname}"


def _build_hellaswag_manifest_task_config(
    task_config: NativeTaskConfig,
) -> dict[str, object]:
    dataset_kwargs = _require_mapping(
        task_config.get("dataset_kwargs"),
        field_name="HellaSwag dataset_kwargs",
    )

    return {
        "task": task_config["task"],
        "dataset_path": task_config["dataset_path"],
        "dataset_name": task_config["dataset_name"],
        "validation_split": task_config["validation_split"],
        "output_type": task_config["output_type"],
        "dataset_kwargs": dict(dataset_kwargs),
        "num_fewshot": task_config["num_fewshot"],
        "metadata": {
            "version": _EXPECTED_METADATA_VERSION,
        },
        "metric_list": [
            {
                "metric": metric,
                "aggregation": aggregation,
                "higher_is_better": higher_is_better,
            }
            for (
                metric,
                aggregation,
                higher_is_better,
            ) in _EXPECTED_METRIC_SIGNATURE
        ],
        "process_docs": _qualified_callable_name(
            task_config.get("process_docs"),
            field_name="HellaSwag process_docs",
        ),
    }


def build_hellaswag_authoritative_run_manifest(
    model_identity: ModelIdentity,
    tokenizer_identity: TokenizerIdentity,
    dataset_identity: DatasetIdentity,
    environment: EnvironmentFingerprint,
    *,
    run_id: str,
    created_at: datetime,
) -> dict[str, object]:
    """Build the JSON-compatible authoritative HellaSwag run manifest."""
    if not isinstance(environment, EnvironmentFingerprint):
        raise TypeError("environment must be an EnvironmentFingerprint")

    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")

    if not run_id or run_id.isspace():
        raise ValueError("run_id must contain a non-whitespace character")

    if not isinstance(created_at, datetime):
        raise TypeError("created_at must be a datetime")

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")

    evaluation_profile = build_hellaswag_evaluation_profile()

    invocation = build_hellaswag_simple_evaluate_kwargs(
        model_identity,
        tokenizer_identity,
        dataset_identity,
    )

    tasks_value = invocation.get("tasks")

    if not isinstance(tasks_value, list) or len(tasks_value) != 1:
        raise RuntimeError("HellaSwag authoritative invocation must contain exactly one task")

    task_value = tasks_value[0]

    if not isinstance(task_value, dict):
        raise RuntimeError("HellaSwag authoritative task config must be a dictionary")

    manifest_invocation = dict(invocation)
    manifest_invocation["tasks"] = [
        _build_hellaswag_manifest_task_config(
            cast(NativeTaskConfig, task_value),
        )
    ]

    return {
        "schema_version": "1",
        "run_id": run_id,
        "run_kind": "baseline",
        "purpose": "quality_baseline",
        "authoritative": True,
        "quant_version": distribution_version("huyawo-quant"),
        "created_at": created_at.astimezone(UTC).isoformat(),
        "model_identity": model_identity.model_dump(mode="json"),
        "tokenizer_identity": tokenizer_identity.model_dump(mode="json"),
        "dataset_identity": dataset_identity.model_dump(mode="json"),
        "evaluation_profile": evaluation_profile.model_dump(mode="json"),
        "environment": environment.model_dump(mode="json"),
        "simple_evaluate_kwargs": manifest_invocation,
    }


def _require_finite_number(
    value: object,
    *,
    field_name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(f"{field_name} must be a numeric value")

    numeric_value = float(value)

    if not isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite")

    return numeric_value


def _require_positive_int(
    value: object,
    *,
    field_name: str,
) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")

    return value


def build_hellaswag_benchmark_result(
    result_payload: Mapping[str, object],
) -> BenchmarkResult:
    """Project one complete authoritative HellaSwag result."""
    if result_payload.get("lm_eval_version") != _EXPECTED_LM_EVAL_VERSION:
        raise ValueError(
            "lm-eval result version must match the accepted "
            f"benchmark version {_EXPECTED_LM_EVAL_VERSION}"
        )

    config = _require_mapping(
        result_payload.get("config"),
        field_name="lm-eval config",
    )

    expected_config: dict[str, object] = {
        "batch_size": _EXPECTED_BATCH_SIZE,
        "device": _EXPECTED_DEVICE,
        "limit": None,
        "random_seed": _EXPECTED_SEED,
        "numpy_seed": _EXPECTED_SEED,
        "torch_seed": _EXPECTED_SEED,
        "fewshot_seed": _EXPECTED_SEED,
    }

    for field_name, expected in expected_config.items():
        if field_name not in config:
            raise ValueError(f"lm-eval config is missing {field_name}")

        observed = config[field_name]

        if observed != expected:
            raise ValueError(f"lm-eval config {field_name}={observed!r}; expected {expected!r}")

    sample_counts = _require_mapping(
        result_payload.get("n-samples"),
        field_name="lm-eval n-samples",
    )

    task_sample_counts = _require_mapping(
        sample_counts.get(_EXPECTED_TASK),
        field_name="lm-eval HellaSwag sample counts",
    )

    original_samples = _require_positive_int(
        task_sample_counts.get("original"),
        field_name="lm-eval HellaSwag original sample count",
    )

    effective_samples = _require_positive_int(
        task_sample_counts.get("effective"),
        field_name="lm-eval HellaSwag effective sample count",
    )

    if effective_samples != original_samples:
        raise ValueError("authoritative HellaSwag result must cover the complete sample scope")

    results = _require_mapping(
        result_payload.get("results"),
        field_name="lm-eval results",
    )

    task_result = _require_mapping(
        results.get(_EXPECTED_TASK),
        field_name="lm-eval HellaSwag result",
    )

    if task_result.get("name") != _EXPECTED_TASK:
        raise ValueError("lm-eval HellaSwag result must declare name='hellaswag'")

    sample_len = _require_positive_int(
        task_result.get("sample_len"),
        field_name="lm-eval HellaSwag sample_len",
    )

    if sample_len != effective_samples:
        raise ValueError("lm-eval HellaSwag sample_len does not match effective sample count")

    samples = _require_mapping(
        result_payload.get("samples"),
        field_name="lm-eval samples",
    )

    task_samples = samples.get(_EXPECTED_TASK)

    if not isinstance(task_samples, list):
        raise ValueError("lm-eval HellaSwag samples must be a list")

    if len(task_samples) != effective_samples:
        raise ValueError("lm-eval HellaSwag samples do not match effective sample count")

    acc = _require_finite_number(
        task_result.get("acc,none"),
        field_name="hellaswag acc",
    )

    acc_norm = _require_finite_number(
        task_result.get("acc_norm,none"),
        field_name="hellaswag acc_norm",
    )

    _require_finite_number(
        task_result.get("acc_stderr,none"),
        field_name="hellaswag acc stderr",
    )

    _require_finite_number(
        task_result.get("acc_norm_stderr,none"),
        field_name="hellaswag acc_norm stderr",
    )

    return BenchmarkResult(
        result_kind="quality",
        evaluation_profile=build_hellaswag_evaluation_profile(),
        metrics=(
            BenchmarkMetric(
                name="acc",
                scope=_EXPECTED_TASK,
                unit="ratio",
                direction="higher_is_better",
                observations=(acc,),
                aggregation="mean",
                aggregate=acc,
            ),
            BenchmarkMetric(
                name="acc_norm",
                scope=_EXPECTED_TASK,
                unit="ratio",
                direction="higher_is_better",
                observations=(acc_norm,),
                aggregation="mean",
                aggregate=acc_norm,
            ),
        ),
    )
