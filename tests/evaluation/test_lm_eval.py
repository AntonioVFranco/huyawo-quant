"""Tests for the thin lm-evaluation-harness adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import version as distribution_version
from typing import cast

import huyawo_quant.evaluation.lm_eval as lm_eval_adapter
import pytest
from huyawo_quant.contracts import (
    BenchmarkResult,
    CudaToolkitFingerprint,
    DatasetIdentity,
    EnvironmentFingerprint,
    EvaluationProfile,
    ModelIdentity,
    NvidiaGpuFingerprint,
    PythonDistributionFingerprint,
    TokenizerIdentity,
)
from huyawo_quant.evaluation import (
    build_hellaswag_authoritative_run_manifest,
    build_hellaswag_benchmark_result,
    build_hellaswag_evaluation_profile,
    build_hellaswag_model_args,
    build_hellaswag_simple_evaluate_kwargs,
    build_hellaswag_task_config,
)

_MODEL_SHA = "7ae557604adf67be50417f59c2c2f167def9a775"
_DATASET_SHA = "218ec52e09a7e7462a5400043bb9a69a41d06b76"


def _dataset_identity(
    *,
    dataset_id: str = "Rowan/hellaswag",
    config_name: str | None = None,
    split: str = "validation",
    resolved_revision: str = _DATASET_SHA,
) -> DatasetIdentity:
    return DatasetIdentity(
        dataset_id=dataset_id,
        config_name=config_name,
        split=split,
        requested_revision="main",
        resolved_revision=resolved_revision,
    )


def test_build_hellaswag_evaluation_profile_matches_accepted_protocol() -> None:
    profile = build_hellaswag_evaluation_profile()

    assert isinstance(profile, EvaluationProfile)
    assert profile.model_dump(mode="json") == {
        "benchmark": "lm-evaluation-harness",
        "benchmark_version": "0.4.13",
        "tasks": ["hellaswag"],
        "metrics": ["acc", "acc_norm"],
        "sample_limit": None,
        "seed": 42,
    }


def test_build_hellaswag_task_config_binds_immutable_dataset_revision() -> None:
    task_config = build_hellaswag_task_config(_dataset_identity())

    assert task_config["task"] == "hellaswag"
    assert task_config["dataset_path"] == "Rowan/hellaswag"
    assert task_config["dataset_name"] is None
    assert task_config["validation_split"] == "validation"
    assert task_config["output_type"] == "multiple_choice"
    assert callable(task_config["process_docs"])
    assert task_config["metadata"] == {"version": 1.0}
    assert task_config["metric_list"] == [
        {
            "metric": "acc",
            "aggregation": "mean",
            "higher_is_better": True,
        },
        {
            "metric": "acc_norm",
            "aggregation": "mean",
            "higher_is_better": True,
        },
    ]
    assert task_config["dataset_kwargs"] == {
        "revision": _DATASET_SHA,
    }
    assert task_config["num_fewshot"] == 0


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        (
            "dataset_id",
            "other/hellaswag",
            "must target Rowan/hellaswag",
        ),
        (
            "config_name",
            "default",
            "config_name must be None",
        ),
        (
            "split",
            "train",
            "split must be validation",
        ),
    ],
)
def test_task_config_rejects_wrong_dataset_selection(
    field_name: str,
    value: str,
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "dataset_id": "Rowan/hellaswag",
        "config_name": None,
        "split": "validation",
        "resolved_revision": _DATASET_SHA,
    }
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=message):
        build_hellaswag_task_config(
            _dataset_identity(
                dataset_id=cast(str, kwargs["dataset_id"]),
                config_name=cast(str | None, kwargs["config_name"]),
                split=cast(str, kwargs["split"]),
                resolved_revision=cast(
                    str,
                    kwargs["resolved_revision"],
                ),
            )
        )


@pytest.mark.parametrize(
    "resolved_revision",
    [
        "main",
        "a" * 39,
        "a" * 41,
        "g" * 40,
        "0123456789abcdef0123456789abcdef0123456-",
    ],
)
def test_task_config_rejects_nonimmutable_dataset_revision(
    resolved_revision: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="40-character hexadecimal commit SHA",
    ):
        build_hellaswag_task_config(
            _dataset_identity(
                resolved_revision=resolved_revision,
            )
        )


def test_adapter_fails_closed_on_lm_eval_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_distribution_version(_: str) -> str:
        return "0.4.12"

    monkeypatch.setattr(
        lm_eval_adapter,
        "distribution_version",
        fake_distribution_version,
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected lm-eval version",
    ):
        build_hellaswag_evaluation_profile()


def test_adapter_fails_closed_on_native_task_semantic_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def process_docs() -> None:
        return None

    drifted_config: dict[str, object] = {
        "task": "hellaswag",
        "dataset_path": "Rowan/hellaswag",
        "dataset_name": None,
        "validation_split": "validation",
        "output_type": "generate_until",
        "process_docs": process_docs,
        "metric_list": [
            {
                "metric": "acc",
                "aggregation": "mean",
                "higher_is_better": True,
            },
            {
                "metric": "acc_norm",
                "aggregation": "mean",
                "higher_is_better": True,
            },
        ],
        "metadata": {"version": 1.0},
    }

    def fake_native_config() -> dict[str, object]:
        return drifted_config

    monkeypatch.setattr(
        lm_eval_adapter,
        "_load_native_hellaswag_config",
        fake_native_config,
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected native HellaSwag configuration",
    ):
        build_hellaswag_evaluation_profile()


def _model_identity(
    *,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    resolved_revision: str = _MODEL_SHA,
) -> ModelIdentity:
    return ModelIdentity(
        model_id=model_id,
        requested_revision="main",
        resolved_revision=resolved_revision,
    )


def _tokenizer_identity(
    *,
    tokenizer_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    resolved_revision: str = _MODEL_SHA,
) -> TokenizerIdentity:
    return TokenizerIdentity(
        tokenizer_id=tokenizer_id,
        requested_revision="main",
        resolved_revision=resolved_revision,
    )


def test_build_hellaswag_model_args_matches_accepted_protocol() -> None:
    model_args = build_hellaswag_model_args(
        _model_identity(),
        _tokenizer_identity(),
    )

    assert model_args == {
        "pretrained": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": _MODEL_SHA,
        "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
    }


def test_build_simple_evaluate_kwargs_matches_authoritative_protocol() -> None:
    kwargs = build_hellaswag_simple_evaluate_kwargs(
        _model_identity(),
        _tokenizer_identity(),
        _dataset_identity(),
    )

    assert set(kwargs) == {
        "model",
        "model_args",
        "tasks",
        "num_fewshot",
        "batch_size",
        "device",
        "limit",
        "log_samples",
        "apply_chat_template",
        "predict_only",
        "random_seed",
        "numpy_random_seed",
        "torch_random_seed",
        "fewshot_random_seed",
    }

    assert kwargs["model"] == "hf"
    assert kwargs["model_args"] == {
        "pretrained": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": _MODEL_SHA,
        "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
    }

    tasks = cast(
        list[dict[str, object]],
        kwargs["tasks"],
    )

    assert len(tasks) == 1
    assert tasks[0]["task"] == "hellaswag"
    assert tasks[0]["dataset_kwargs"] == {
        "revision": _DATASET_SHA,
    }
    assert tasks[0]["num_fewshot"] == 0

    assert kwargs["num_fewshot"] == 0
    assert kwargs["batch_size"] == 1
    assert kwargs["device"] == "cuda:0"
    assert kwargs["limit"] is None
    assert kwargs["log_samples"] is True
    assert kwargs["apply_chat_template"] is False
    assert kwargs["predict_only"] is False
    assert kwargs["random_seed"] == 42
    assert kwargs["numpy_random_seed"] == 42
    assert kwargs["torch_random_seed"] == 42
    assert kwargs["fewshot_random_seed"] == 42
    assert "output_path" not in kwargs


def test_build_simple_evaluate_kwargs_accepts_positive_smoke_limit() -> None:
    kwargs = build_hellaswag_simple_evaluate_kwargs(
        _model_identity(),
        _tokenizer_identity(),
        _dataset_identity(),
        limit=8,
    )

    assert kwargs["limit"] == 8


@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
        cast(int, True),
    ],
)
def test_build_simple_evaluate_kwargs_rejects_invalid_smoke_limit(
    limit: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        build_hellaswag_simple_evaluate_kwargs(
            _model_identity(),
            _tokenizer_identity(),
            _dataset_identity(),
            limit=limit,
        )


@pytest.mark.parametrize(
    ("model_id", "resolved_revision", "message"),
    [
        (
            "other/model",
            _MODEL_SHA,
            "must target Qwen/Qwen2.5-0.5B-Instruct",
        ),
        (
            "Qwen/Qwen2.5-0.5B-Instruct",
            "a" * 40,
            "accepted immutable model revision",
        ),
    ],
)
def test_model_args_rejects_wrong_model_identity(
    model_id: str,
    resolved_revision: str,
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        build_hellaswag_model_args(
            _model_identity(
                model_id=model_id,
                resolved_revision=resolved_revision,
            ),
            _tokenizer_identity(),
        )


@pytest.mark.parametrize(
    ("tokenizer_id", "resolved_revision", "message"),
    [
        (
            "other/tokenizer",
            _MODEL_SHA,
            "must target Qwen/Qwen2.5-0.5B-Instruct",
        ),
        (
            "Qwen/Qwen2.5-0.5B-Instruct",
            "a" * 40,
            "accepted immutable tokenizer revision",
        ),
    ],
)
def test_model_args_rejects_wrong_tokenizer_identity(
    tokenizer_id: str,
    resolved_revision: str,
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        build_hellaswag_model_args(
            _model_identity(),
            _tokenizer_identity(
                tokenizer_id=tokenizer_id,
                resolved_revision=resolved_revision,
            ),
        )


def _environment_fingerprint() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        captured_at=datetime(
            2026,
            9,
            27,
            14,
            0,
            tzinfo=UTC,
        ),
        project_root="/workspace/huyawo-quant",
        working_directory="/workspace/huyawo-quant",
        virtual_environment="/workspace/huyawo-quant/.venv",
        python_executable=("/workspace/huyawo-quant/.venv/bin/python"),
        python_version="3.12.3",
        python_implementation="CPython",
        python_prefix="/workspace/huyawo-quant/.venv",
        python_base_prefix="/usr",
        platform_system="Linux",
        platform_release="test-release",
        platform_machine="x86_64",
        platform_version="test-version",
        cuda_visible_devices=None,
        cuda_device_order=None,
        python_distributions=(
            PythonDistributionFingerprint(
                name="lm-eval",
                status="INSTALLED",
                version="0.4.13",
            ),
        ),
        nvidia_smi_path="/usr/bin/nvidia-smi",
        nvidia_gpus=(
            NvidiaGpuFingerprint(
                observed_index=0,
                name="NVIDIA RTX A5000",
                uuid="GPU-test",
                pci_bus_id="00000000:01:00.0",
                memory_total_mib=24564,
                driver_version="580.159.04",
                compute_capability_major=8,
                compute_capability_minor=6,
            ),
        ),
        cuda_toolkit=CudaToolkitFingerprint(
            nvcc_path="/usr/local/cuda/bin/nvcc",
            release="12.8",
            version="12.8.93",
        ),
    )


def _authoritative_lm_eval_result(
    *,
    acc: object = 0.5,
    acc_norm: object = 0.6,
) -> dict[str, object]:
    return {
        "lm_eval_version": "0.4.13",
        "config": {
            "batch_size": 1,
            "device": "cuda:0",
            "limit": None,
            "random_seed": 42,
            "numpy_seed": 42,
            "torch_seed": 42,
            "fewshot_seed": 42,
        },
        "n-samples": {
            "hellaswag": {
                "original": 2,
                "effective": 2,
            },
        },
        "results": {
            "hellaswag": {
                "name": "hellaswag",
                "alias": "hellaswag",
                "sample_len": 2,
                "acc,none": acc,
                "acc_norm,none": acc_norm,
                "acc_stderr,none": 0.1,
                "acc_norm_stderr,none": 0.1,
            },
        },
        "samples": {
            "hellaswag": [
                {"doc_id": 0},
                {"doc_id": 1},
            ],
        },
    }


def test_build_authoritative_run_manifest_is_json_compatible() -> None:
    manifest = build_hellaswag_authoritative_run_manifest(
        _model_identity(),
        _tokenizer_identity(),
        _dataset_identity(),
        _environment_fingerprint(),
        run_id="m5-hellaswag-baseline-001",
        created_at=datetime(
            2026,
            9,
            27,
            14,
            30,
            tzinfo=UTC,
        ),
    )

    assert manifest["schema_version"] == "1"
    assert manifest["run_id"] == "m5-hellaswag-baseline-001"
    assert manifest["run_kind"] == "baseline"
    assert manifest["purpose"] == "quality_baseline"
    assert manifest["authoritative"] is True
    assert manifest["quant_version"] == distribution_version("huyawo-quant")
    assert manifest["created_at"] == "2026-09-27T14:30:00+00:00"

    profile = cast(
        dict[str, object],
        manifest["evaluation_profile"],
    )

    assert profile["sample_limit"] is None
    assert profile["seed"] == 42

    invocation = cast(
        dict[str, object],
        manifest["simple_evaluate_kwargs"],
    )

    assert invocation["limit"] is None
    assert invocation["batch_size"] == 1
    assert invocation["device"] == "cuda:0"
    assert invocation["random_seed"] == 42
    assert invocation["numpy_random_seed"] == 42
    assert invocation["torch_random_seed"] == 42
    assert invocation["fewshot_random_seed"] == 42

    tasks = cast(
        list[dict[str, object]],
        invocation["tasks"],
    )

    assert len(tasks) == 1
    assert tasks[0]["dataset_kwargs"] == {
        "revision": _DATASET_SHA,
    }
    assert tasks[0]["metadata"] == {
        "version": 1.0,
    }
    assert tasks[0]["process_docs"] == "lm_eval.tasks.hellaswag.utils.process_docs"

    serialized = json.dumps(
        manifest,
        sort_keys=True,
    )

    assert json.loads(serialized) == manifest


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "   ",
    ],
)
def test_build_authoritative_run_manifest_rejects_empty_run_id(
    run_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="run_id",
    ):
        build_hellaswag_authoritative_run_manifest(
            _model_identity(),
            _tokenizer_identity(),
            _dataset_identity(),
            _environment_fingerprint(),
            run_id=run_id,
            created_at=datetime(
                2026,
                9,
                27,
                14,
                30,
                tzinfo=UTC,
            ),
        )


def test_build_authoritative_run_manifest_rejects_naive_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        build_hellaswag_authoritative_run_manifest(
            _model_identity(),
            _tokenizer_identity(),
            _dataset_identity(),
            _environment_fingerprint(),
            run_id="m5-hellaswag-baseline-001",
            created_at=datetime(
                2026,
                9,
                27,
                14,
                30,
            ),
        )


def test_build_authoritative_run_manifest_rejects_wrong_environment_type() -> None:
    with pytest.raises(
        TypeError,
        match="EnvironmentFingerprint",
    ):
        build_hellaswag_authoritative_run_manifest(
            _model_identity(),
            _tokenizer_identity(),
            _dataset_identity(),
            cast(
                EnvironmentFingerprint,
                object(),
            ),
            run_id="m5-hellaswag-baseline-001",
            created_at=datetime(
                2026,
                9,
                27,
                14,
                30,
                tzinfo=UTC,
            ),
        )


def test_build_hellaswag_benchmark_result_projects_authoritative_metrics() -> None:
    result = build_hellaswag_benchmark_result(
        _authoritative_lm_eval_result(),
    )

    assert isinstance(
        result,
        BenchmarkResult,
    )

    assert result.model_dump(mode="json") == {
        "result_kind": "quality",
        "evaluation_profile": {
            "benchmark": "lm-evaluation-harness",
            "benchmark_version": "0.4.13",
            "tasks": ["hellaswag"],
            "metrics": ["acc", "acc_norm"],
            "sample_limit": None,
            "seed": 42,
        },
        "workload_profile": None,
        "metrics": [
            {
                "name": "acc",
                "scope": "hellaswag",
                "unit": "ratio",
                "direction": "higher_is_better",
                "observations": [0.5],
                "aggregation": "mean",
                "aggregate": 0.5,
            },
            {
                "name": "acc_norm",
                "scope": "hellaswag",
                "unit": "ratio",
                "direction": "higher_is_better",
                "observations": [0.6],
                "aggregation": "mean",
                "aggregate": 0.6,
            },
        ],
    }


def test_build_hellaswag_benchmark_result_rejects_wrong_lm_eval_version() -> None:
    payload = _authoritative_lm_eval_result()
    payload["lm_eval_version"] = "0.4.12"

    with pytest.raises(
        ValueError,
        match="result version",
    ):
        build_hellaswag_benchmark_result(payload)


def test_build_hellaswag_benchmark_result_rejects_missing_metric() -> None:
    payload = _authoritative_lm_eval_result()

    results = cast(
        dict[str, object],
        payload["results"],
    )

    hellaswag = cast(
        dict[str, object],
        results["hellaswag"],
    )

    del hellaswag["acc,none"]

    with pytest.raises(
        ValueError,
        match="hellaswag acc",
    ):
        build_hellaswag_benchmark_result(payload)


def test_build_hellaswag_benchmark_result_rejects_boolean_metric() -> None:
    with pytest.raises(
        ValueError,
        match="numeric value",
    ):
        build_hellaswag_benchmark_result(
            _authoritative_lm_eval_result(
                acc=True,
            )
        )


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_build_hellaswag_benchmark_result_rejects_nonfinite_metric(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        build_hellaswag_benchmark_result(
            _authoritative_lm_eval_result(
                acc_norm=value,
            )
        )


def test_build_hellaswag_benchmark_result_rejects_limited_result() -> None:
    payload = _authoritative_lm_eval_result()

    config = cast(
        dict[str, object],
        payload["config"],
    )

    config["limit"] = 8

    with pytest.raises(
        ValueError,
        match="limit",
    ):
        build_hellaswag_benchmark_result(payload)
