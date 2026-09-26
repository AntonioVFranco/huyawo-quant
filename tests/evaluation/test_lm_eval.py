"""Tests for the thin lm-evaluation-harness adapter."""

from __future__ import annotations

from typing import cast

import huyawo_quant.evaluation.lm_eval as lm_eval_adapter
import pytest
from huyawo_quant.contracts import DatasetIdentity, EvaluationProfile
from huyawo_quant.evaluation import (
    build_hellaswag_evaluation_profile,
    build_hellaswag_task_config,
)

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
