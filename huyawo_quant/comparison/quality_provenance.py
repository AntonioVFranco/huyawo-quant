"""Read-only zero-shot HellaSwag evidence audit, not a comparison authorization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


class QualityEvidenceBlocked(ValueError):
    """Evidence is insufficient for the audited quality protocol."""


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise QualityEvidenceBlocked(code)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(
        path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 128 * 1024 * 1024,
        "SOURCE_INVALID",
    )

    raw = path.read_bytes()

    _require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        "SOURCE_SHA_MISMATCH",
    )

    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise QualityEvidenceBlocked("SOURCE_JSON_INVALID") from error

    _require(isinstance(data, dict), "SOURCE_SHAPE_INVALID")
    return data


def _inspect(
    data: Mapping[str, Any],
    expected_samples: int,
) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, str]]]:
    _require(
        data.get("lm_eval_version") == "0.4.13",
        "HARNESS_VERSION_MISMATCH",
    )

    configs = data.get("configs")
    _require(isinstance(configs, dict), "CONFIGS_MISSING")

    task = configs.get("hellaswag")
    _require(isinstance(task, dict), "TASK_MISSING")

    _require(task.get("task") == "hellaswag", "TASK_ID_MISMATCH")
    _require(
        task.get("dataset_path") == "Rowan/hellaswag",
        "DATASET_MISMATCH",
    )
    _require(
        task.get("validation_split") == "validation",
        "SPLIT_MISMATCH",
    )
    _require(
        task.get("output_type") == "multiple_choice",
        "OUTPUT_TYPE_MISMATCH",
    )
    _require(
        type(task.get("num_fewshot")) is int and task["num_fewshot"] == 0,
        "NOT_ZERO_SHOT",
    )

    metrics = task.get("metric_list")

    _require(
        isinstance(metrics, list)
        and len(metrics) == 2
        and [item.get("metric") for item in metrics if isinstance(item, dict)]
        == ["acc", "acc_norm"]
        and all(
            isinstance(item, dict)
            and item.get("aggregation") == "mean"
            and item.get("higher_is_better") is True
            for item in metrics
        ),
        "METRIC_CONFIG_MISMATCH",
    )

    _require(
        task.get("process_results") is None,
        "CUSTOM_SCORING_UNVERIFIED",
    )

    fewshot = task.get("fewshot_config")
    metadata = task.get("metadata")

    _require(
        isinstance(fewshot, dict) and isinstance(metadata, dict),
        "EXCLUDED_PATHS_UNVERIFIED",
    )
    _require(
        "process_docs" in fewshot and "pretrained" in metadata,
        "EXCLUDED_PATHS_MISSING",
    )

    samples_obj = data.get("samples")
    _require(isinstance(samples_obj, dict), "SAMPLES_MISSING")

    samples = samples_obj.get("hellaswag")
    _require(
        isinstance(samples, list) and len(samples) == expected_samples,
        "SAMPLE_COUNT_MISMATCH",
    )

    indexed: dict[tuple[int, str], dict[str, str]] = {}
    score_totals = {"acc": 0, "acc_norm": 0}
    filters: set[str] = set()

    for sample in samples:
        _require(isinstance(sample, dict), "SAMPLE_INVALID")

        doc_id = sample.get("doc_id")
        filter_name = sample.get("filter")

        _require(
            type(doc_id) is int and isinstance(filter_name, str) and bool(filter_name),
            "SAMPLE_KEY_INVALID",
        )

        key = (doc_id, filter_name)
        _require(key not in indexed, "DUPLICATE_SAMPLE")
        filters.add(filter_name)

        _require(
            all(
                field in sample
                for field in (
                    "doc_hash",
                    "prompt_hash",
                    "target_hash",
                    "arguments",
                )
            ),
            "INPUT_FIELD_MISSING",
        )

        _require(
            all(
                isinstance(sample[field], str) and bool(sample[field])
                for field in (
                    "doc_hash",
                    "prompt_hash",
                    "target_hash",
                )
            ),
            "INPUT_HASH_INVALID",
        )

        indexed[key] = {
            "doc_hash": sample["doc_hash"],
            "prompt_hash": sample["prompt_hash"],
            "target_hash": sample["target_hash"],
            "arguments_sha256": _digest(sample["arguments"]),
        }

        doc = sample.get("doc")
        _require(isinstance(doc, dict), "DOCUMENT_MISSING")

        choices = doc.get("choices")
        gold = doc.get("gold")

        _require(
            isinstance(choices, list)
            and len(choices) == 4
            and all(isinstance(choice, str) and len(choice) > 0 for choice in choices)
            and type(gold) is int
            and 0 <= gold < len(choices),
            "CHOICE_OR_GOLD_INVALID",
        )

        _require(
            str(sample.get("target")) == str(gold),
            "TARGET_MISMATCH",
        )

        responses = sample.get("filtered_resps")
        _require(
            isinstance(responses, list) and len(responses) == 4,
            "RESPONSE_SHAPE_INVALID",
        )

        lls: list[float] = []

        for response in responses:
            _require(
                isinstance(response, list) and len(response) == 2,
                "LOGPROB_RESPONSE_INVALID",
            )

            value = response[0]

            _require(
                type(value) in (int, float) and math.isfinite(value),
                "LOGPROB_NONFINITE",
            )

            lls.append(float(value))

        predictions = {
            "acc": max(
                range(4),
                key=lambda idx: lls[idx],
            ),
            "acc_norm": max(
                range(4),
                key=lambda idx: lls[idx] / len(choices[idx]),
            ),
        }

        _require(
            sample.get("metrics") == ["acc", "acc_norm"],
            "RECORDED_METRIC_SET_MISMATCH",
        )

        for metric, prediction in predictions.items():
            expected = int(prediction == gold)

            _require(
                type(sample.get(metric)) in (int, float) and sample[metric] == expected,
                "SAMPLE_SCORING_MISMATCH:" + metric,
            )

            score_totals[metric] += expected

    _require(
        len(filters) == 1,
        "MULTIPLE_FILTERS_UNVERIFIED",
    )
    filter_name = next(iter(filters))

    results = data.get("results")
    _require(
        isinstance(results, dict) and isinstance(results.get("hellaswag"), dict),
        "AGGREGATES_MISSING",
    )

    for metric, total in score_totals.items():
        aggregate = results["hellaswag"].get(metric + "," + filter_name)

        _require(
            type(aggregate) in (int, float)
            and math.isclose(
                float(aggregate),
                total / expected_samples,
                rel_tol=0,
                abs_tol=1e-12,
            ),
            "AGGREGATE_MISMATCH:" + metric,
        )

    normalized = deepcopy(task)
    normalized["fewshot_config"].pop("process_docs")
    normalized["metadata"].pop("pretrained")

    return normalized, indexed


def audit_zero_shot_hellaswag_pair(
    baseline_path: Path,
    baseline_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
    *,
    expected_samples: int = 10042,
) -> dict[str, object]:
    """Verify recorded inputs and scoring without issuing formal bindings."""

    _require(
        type(expected_samples) is int and expected_samples > 0,
        "EXPECTED_COUNT_INVALID",
    )

    left_config, left_inputs = _inspect(
        _read(baseline_path, baseline_sha256),
        expected_samples,
    )

    right_config, right_inputs = _inspect(
        _read(candidate_path, candidate_sha256),
        expected_samples,
    )

    left_dataset = left_config.get("dataset_kwargs")
    right_dataset = right_config.get("dataset_kwargs")

    if isinstance(left_dataset, dict) and isinstance(right_dataset, dict):
        _require(
            left_dataset.get("revision") == right_dataset.get("revision"),
            "DATASET_REVISION_MISMATCH:M13",
        )

    _require(
        left_config == right_config,
        "EFFECTIVE_TASK_CONFIG_MISMATCH",
    )
    _require(
        left_inputs == right_inputs,
        "SAMPLE_INPUT_MISMATCH",
    )

    ordered = [
        {
            "doc_id": key[0],
            "filter": key[1],
            **left_inputs[key],
        }
        for key in sorted(left_inputs)
    ]

    return {
        "status": "RECORDED_INPUT_AND_SCORING_AUDITED",
        "sample_count": expected_samples,
        "input_audit_sha256": _digest(
            {
                "schema": "huyawo.quant.input_audit.v1",
                "samples": ordered,
            }
        ),
        "effective_config_audit_sha256": _digest(
            {
                "schema": "huyawo.quant.task_audit.v1",
                "config": left_config,
            }
        ),
        "formal_bindings_created": False,
        "metric_deltas": (),
    }


def _verified_harness_blobs() -> dict[str, str]:
    import base64
    import sys
    from importlib.metadata import distribution

    expected = {
        "lm_eval/loggers/utils.py": "6f24d111eefe6e0ec3aa0313865948e124ca3522",
        "lm_eval/api/task.py": "9255170de444bae6edd53797e3477031cd5d5da5",
        "lm_eval/evaluator.py": "2251597c304fb160ccb3204173e07a4df9de2c89",
        "lm_eval/tasks/hellaswag/hellaswag.yaml": "e0a2b87d7a2fd8831406d7bbce6c8b49065fbbca",
        "lm_eval/tasks/hellaswag/utils.py": "b526a9e93076f7db54221072d58ca4bd7161ee97",
    }

    installed = distribution("lm-eval")
    _require(installed.version == "0.4.13", "HARNESS_VERSION_DRIFT")
    _require(installed.files is not None, "HARNESS_RECORD_MISSING")

    records = {str(item): item for item in installed.files}
    root = Path(sys.prefix).resolve()

    for name, blob in expected.items():
        record = records.get(name)
        _require(record is not None, "SOURCE_RECORD_MISSING:" + name)
        _require(
            record.hash is not None and record.hash.mode == "sha256",
            "SOURCE_RECORD_HASH_MISSING:" + name,
        )

        path = Path(installed.locate_file(name))
        _require(
            path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            "SOURCE_PATH_INVALID:" + name,
        )

        raw = path.read_bytes()
        record_hash = base64.urlsafe_b64decode(
            record.hash.value + "=" * (-len(record.hash.value) % 4)
        )

        _require(
            hashlib.sha256(raw).digest() == record_hash,
            "SOURCE_RECORD_HASH_MISMATCH:" + name,
        )

        git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()

        _require(git_blob == blob, "SOURCE_TAG_MISMATCH:" + name)

    return expected


def draft_effective_quality_task_signature(
    baseline_path: Path,
    baseline_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
    m5_manifest_path: Path,
    m5_manifest_sha256: str,
    *,
    expected_samples: int = 10042,
) -> dict[str, object]:
    """Derive draft task and scoring projections without granting comparability."""

    pair = audit_zero_shot_hellaswag_pair(
        baseline_path,
        baseline_sha256,
        candidate_path,
        candidate_sha256,
        expected_samples=expected_samples,
    )

    manifest = _read(m5_manifest_path, m5_manifest_sha256)

    _require(
        isinstance(manifest.get("dataset_identity"), dict),
        "DATASET_IDENTITY_MISSING",
    )
    _require(
        isinstance(manifest.get("evaluation_profile"), dict),
        "PROFILE_MISSING",
    )
    _require(
        isinstance(manifest.get("simple_evaluate_kwargs"), dict),
        "INVOCATION_MISSING",
    )

    dataset = manifest["dataset_identity"]
    profile = manifest["evaluation_profile"]
    invocation = manifest["simple_evaluate_kwargs"]

    revision = dataset.get("resolved_revision")

    _require(
        isinstance(revision, str)
        and len(revision) == 40
        and all(char in "0123456789abcdef" for char in revision),
        "DATASET_REVISION_INVALID",
    )

    _require(
        profile.get("benchmark_version") == "0.4.13" and profile.get("sample_limit") is None,
        "PROFILE_PROTOCOL_MISMATCH",
    )

    _require(
        invocation.get("num_fewshot") == 0
        and invocation.get("batch_size") == 1
        and invocation.get("log_samples") is True
        and invocation.get("apply_chat_template") is False,
        "INVOCATION_PROTOCOL_MISMATCH",
    )

    input_seeds = (
        "random_seed",
        "numpy_random_seed",
        "torch_random_seed",
        "fewshot_random_seed",
    )
    output_seeds = (
        "random_seed",
        "numpy_seed",
        "torch_seed",
        "fewshot_seed",
    )

    for name in input_seeds:
        _require(
            type(invocation.get(name)) is int and invocation[name] == 42,
            "INVOCATION_SEED_MISMATCH:" + name,
        )

    records = {
        "M5": _read(baseline_path, baseline_sha256),
        "M13": _read(candidate_path, candidate_sha256),
    }

    configs: dict[str, object] = {}
    filters: dict[str, list[str]] = {}
    raw_config_sha: dict[str, str] = {}
    excluded_sha: dict[str, dict[str, str]] = {}

    for label, record in records.items():
        task = record["configs"]["hellaswag"]

        _require(
            isinstance(task.get("dataset_kwargs"), dict)
            and task["dataset_kwargs"].get("revision") == revision,
            "DATASET_REVISION_MISMATCH:" + label,
        )

        returned = record.get("config")
        _require(
            isinstance(returned, dict),
            "RETURNED_CONFIG_MISSING:" + label,
        )

        for name in output_seeds:
            _require(
                type(returned.get(name)) is int and returned[name] == 42,
                "RETURNED_SEED_MISMATCH:" + label + ":" + name,
            )

        _require(
            returned.get("batch_size") == 1 and returned.get("limit") is None,
            "RETURNED_EVALUATION_OPTIONS_MISMATCH:" + label,
        )

        configs[label] = {key: returned[key] for key in (*output_seeds, "batch_size", "limit")}

        filters[label] = sorted({sample["filter"] for sample in record["samples"]["hellaswag"]})

        raw_config_sha[label] = _digest(task)

        excluded_sha[label] = {
            "fewshot_config.process_docs": _digest(task["fewshot_config"]["process_docs"]),
            "metadata.pretrained": _digest(task["metadata"]["pretrained"]),
        }

    _require(
        configs["M5"] == configs["M13"],
        "RETURNED_EVALUATION_OPTIONS_MISMATCH",
    )
    _require(
        filters["M5"] == filters["M13"] == ["none"],
        "FILTER_SEMANTICS_MISMATCH",
    )

    source_blobs = _verified_harness_blobs()

    task_projection = {
        "schema": "huyawo.quant.effective_quality_task.draft.v1",
        "harness_version": "0.4.13",
        "source_blobs_installed": source_blobs,
        "dataset_id": "Rowan/hellaswag",
        "dataset_revision": revision,
        "dataset_split": "validation",
        "effective_config_audit_sha256": pair["effective_config_audit_sha256"],
        "sample_input_audit_sha256": pair["input_audit_sha256"],
        "sample_count": expected_samples,
        "returned_evaluation_options": configs["M5"],
        "filters": filters["M5"],
        "excluded_paths": [
            "fewshot_config.process_docs",
            "metadata.pretrained",
        ],
        "source_field_paths": {
            "dataset_revision": (
                "M5.run_manifest.dataset_identity.resolved_revision;"
                "M5/M13.configs.hellaswag.dataset_kwargs.revision"
            ),
            "effective_config": (
                "M5/M13.configs.hellaswag excluding "
                "fewshot_config.process_docs and metadata.pretrained"
            ),
            "evaluation_options": ("M5.run_manifest.simple_evaluate_kwargs;M5/M13.config"),
            "samples": (
                "M5/M13.samples.hellaswag[]."
                "{doc_id,filter,doc_hash,prompt_hash,target_hash,arguments}"
            ),
        },
    }

    scoring_projection = {
        "schema": "huyawo.quant.scoring_semantics.draft.v1",
        "harness_version": "0.4.13",
        "relevant_source_blobs_installed": {
            key: source_blobs[key]
            for key in (
                "lm_eval/api/task.py",
                "lm_eval/evaluator.py",
                "lm_eval/tasks/hellaswag/hellaswag.yaml",
                "lm_eval/tasks/hellaswag/utils.py",
            )
        },
        "filter": "none",
        "choice_order": "recorded_original_order",
        "tie_break": "first_maximum_index",
        "acc": "argmax_float_loglikelihood_equals_gold",
        "acc_norm": ("argmax_float_loglikelihood_divided_by_unicode_character_count_equals_gold"),
        "aggregation": "arithmetic_mean",
        "recorded_input_audit_sha256": pair["input_audit_sha256"],
    }

    return {
        "status": "DRAFT_HISTORICAL_SOURCE_ATTESTATION_PENDING",
        "schema": task_projection["schema"],
        "task_projection_draft_sha256": _digest(task_projection),
        "scoring_projection_draft_sha256": _digest(scoring_projection),
        "raw_task_config_sha256_by_run": raw_config_sha,
        "excluded_paths_sha256_by_run": excluded_sha,
        "source_attestation": "RESTORED_INSTALL_ONLY",
        "formal_bindings_created": False,
        "metric_deltas": (),
    }
