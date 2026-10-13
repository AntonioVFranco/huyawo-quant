"""CPU-only regression tests for read-only quality provenance audits."""

import hashlib
import json
from pathlib import Path

import pytest
from huyawo_quant.comparison.quality_provenance import (
    QualityEvidenceBlocked,
    audit_zero_shot_hellaswag_pair,
)


def _sample(doc_id: int, gold: int, ll: list[float]) -> dict:
    choices = ["a", "long", "bb", "ccc"]

    raw_pred = max(
        range(4),
        key=lambda i: ll[i],
    )
    norm_pred = max(
        range(4),
        key=lambda i: ll[i] / len(choices[i]),
    )

    return {
        "doc_id": doc_id,
        "filter": "none",
        "doc": {
            "gold": gold,
            "choices": choices,
        },
        "target": gold,
        "doc_hash": "a" * 64,
        "prompt_hash": "b" * 64,
        "target_hash": "c" * 64,
        "arguments": [["prompt", choice] for choice in choices],
        "filtered_resps": [[value, False] for value in ll],
        "metrics": ["acc", "acc_norm"],
        "acc": float(raw_pred == gold),
        "acc_norm": float(norm_pred == gold),
    }


def _record(pretrained: str, process_docs: str) -> dict:
    samples = [
        _sample(0, 0, [-0.1, -0.3, -1.2, -2]),
        _sample(1, 1, [-0.8, -0.3, -1.5, -2]),
    ]

    return {
        "lm_eval_version": "0.4.13",
        "configs": {
            "hellaswag": {
                "task": "hellaswag",
                "dataset_path": "Rowan/hellaswag",
                "validation_split": "validation",
                "output_type": "multiple_choice",
                "num_fewshot": 0,
                "fewshot_config": {
                    "process_docs": process_docs,
                },
                "metadata": {
                    "pretrained": pretrained,
                },
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
            }
        },
        "samples": {
            "hellaswag": samples,
        },
        "results": {
            "hellaswag": {
                "acc,none": (sum(sample["acc"] for sample in samples) / len(samples)),
                "acc_norm,none": (sum(sample["acc_norm"] for sample in samples) / len(samples)),
            }
        },
    }


def _write(path: Path, record: dict) -> str:
    data = json.dumps(record, sort_keys=True).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _audit(
    tmp_path: Path,
    left: dict,
    right: dict,
) -> dict[str, object]:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"

    return audit_zero_shot_hellaswag_pair(
        baseline,
        _write(baseline, left),
        candidate,
        _write(candidate, right),
        expected_samples=2,
    )


def test_exact_inputs_scoring_and_zero_shot_exclusions(
    tmp_path: Path,
) -> None:
    result = _audit(
        tmp_path,
        _record("bf16", "first"),
        _record("gptq", "second"),
    )

    assert result["status"] == "RECORDED_INPUT_AND_SCORING_AUDITED"
    assert result["sample_count"] == 2
    assert result["formal_bindings_created"] is False
    assert result["metric_deltas"] == ()


def test_changed_arguments_blocked(tmp_path: Path) -> None:
    left = _record("bf16", "a")
    right = _record("gptq", "b")

    right["samples"]["hellaswag"][0]["arguments"][0][0] = "different"

    with pytest.raises(
        QualityEvidenceBlocked,
        match="SAMPLE_INPUT_MISMATCH",
    ):
        _audit(tmp_path, left, right)


def test_changed_metric_blocked(tmp_path: Path) -> None:
    left = _record("bf16", "a")
    right = _record("gptq", "b")

    right["samples"]["hellaswag"][0]["acc_norm"] = 1.0

    with pytest.raises(
        QualityEvidenceBlocked,
        match="SAMPLE_SCORING_MISMATCH",
    ):
        _audit(tmp_path, left, right)


def test_fewshot_or_effective_config_drift_blocked(
    tmp_path: Path,
) -> None:
    left = _record("bf16", "a")
    right = _record("gptq", "b")

    right["configs"]["hellaswag"]["num_fewshot"] = 1

    with pytest.raises(
        QualityEvidenceBlocked,
        match="NOT_ZERO_SHOT",
    ):
        _audit(tmp_path, left, right)

    right = _record("gptq", "b")
    right["configs"]["hellaswag"]["doc_to_text"] = "different"

    with pytest.raises(
        QualityEvidenceBlocked,
        match="EFFECTIVE_TASK_CONFIG_MISMATCH",
    ):
        _audit(tmp_path, left, right)


def test_raw_sha_mismatch_blocked(tmp_path: Path) -> None:
    left = _record("bf16", "a")
    right = _record("gptq", "b")

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"

    _write(baseline, left)
    candidate_hash = _write(candidate, right)

    with pytest.raises(
        QualityEvidenceBlocked,
        match="SOURCE_SHA_MISMATCH",
    ):
        audit_zero_shot_hellaswag_pair(
            baseline,
            "0" * 64,
            candidate,
            candidate_hash,
            expected_samples=2,
        )


def _draft_fixture(
    tmp_path: Path,
    *,
    candidate_seed: int = 42,
    candidate_revision: str = "a" * 40,
) -> dict[str, object]:
    from huyawo_quant.comparison.quality_provenance import (
        draft_effective_quality_task_signature,
    )

    left = _record("bf16", "first")
    right = _record("gptq", "second")

    returned = {
        "random_seed": 42,
        "numpy_seed": 42,
        "torch_seed": 42,
        "fewshot_seed": 42,
        "batch_size": 1,
        "limit": None,
    }

    for record, revision in (
        (left, "a" * 40),
        (right, candidate_revision),
    ):
        record["configs"]["hellaswag"]["dataset_kwargs"] = {"revision": revision}
        record["config"] = dict(returned)

    right["config"]["numpy_seed"] = candidate_seed

    manifest = {
        "dataset_identity": {
            "resolved_revision": "a" * 40,
        },
        "evaluation_profile": {
            "benchmark_version": "0.4.13",
            "sample_limit": None,
        },
        "simple_evaluate_kwargs": {
            "num_fewshot": 0,
            "batch_size": 1,
            "log_samples": True,
            "apply_chat_template": False,
            "random_seed": 42,
            "numpy_random_seed": 42,
            "torch_random_seed": 42,
            "fewshot_random_seed": 42,
        },
    }

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    manifest_path = tmp_path / "manifest.json"

    baseline_sha = _write(baseline, left)
    candidate_sha = _write(candidate, right)
    manifest_sha = _write(manifest_path, manifest)

    return draft_effective_quality_task_signature(
        baseline,
        baseline_sha,
        candidate,
        candidate_sha,
        manifest_path,
        manifest_sha,
        expected_samples=2,
    )


def test_draft_signature_is_deterministic_and_non_authoritative(
    tmp_path: Path,
) -> None:
    first = _draft_fixture(tmp_path)
    second = _draft_fixture(tmp_path)

    assert first == second
    assert first["status"] == ("DRAFT_HISTORICAL_SOURCE_ATTESTATION_PENDING")
    assert first["formal_bindings_created"] is False
    assert first["metric_deltas"] == ()
    assert len(first["task_projection_draft_sha256"]) == 64
    assert len(first["scoring_projection_draft_sha256"]) == 64
    assert (
        first["raw_task_config_sha256_by_run"]["M5"]
        != first["raw_task_config_sha256_by_run"]["M13"]
    )
    assert (
        first["excluded_paths_sha256_by_run"]["M5"] != first["excluded_paths_sha256_by_run"]["M13"]
    )


def test_draft_signature_rejects_changed_returned_seed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        QualityEvidenceBlocked,
        match="RETURNED_SEED_MISMATCH:M13:numpy_seed",
    ):
        _draft_fixture(tmp_path, candidate_seed=43)


def test_draft_signature_rejects_dataset_revision_drift(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        QualityEvidenceBlocked,
        match="DATASET_REVISION_MISMATCH:M13",
    ):
        _draft_fixture(tmp_path, candidate_revision="b" * 40)
