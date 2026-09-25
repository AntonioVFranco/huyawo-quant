"""Tests for canonical evidence bundle contracts."""

from datetime import UTC, datetime, timedelta

import pytest
from huyawo_quant.contracts import (
    ArtifactFileIdentity,
    EvidenceBundle,
    EvidenceFileReference,
    FailureRecord,
    HardwareTarget,
    ModelIdentity,
    QualificationResult,
    QuantArtifactIdentity,
    QuantPlan,
    QuantRecipe,
    RuntimeTarget,
    TokenizerIdentity,
)
from pydantic import ValidationError

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64

STARTED_AT = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
FINISHED_AT = datetime(2026, 9, 25, 12, 5, tzinfo=UTC)


def make_model() -> ModelIdentity:
    return ModelIdentity(
        model_id="org/model",
        requested_revision="main",
        resolved_revision="a" * 40,
    )


def make_tokenizer() -> TokenizerIdentity:
    return TokenizerIdentity(
        tokenizer_id="org/model",
        requested_revision="main",
        resolved_revision="b" * 40,
    )


def make_hardware(
    *,
    device_name: str = "NVIDIA RTX A4500",
) -> HardwareTarget:
    return HardwareTarget(
        device_name=device_name,
        device_count=1,
        memory_bytes_per_device=21_474_836_480,
        compute_capability_major=8,
        compute_capability_minor=6,
    )


def make_runtime(
    *,
    version: str = "0.0.0-test",
) -> RuntimeTarget:
    return RuntimeTarget(
        runtime="vllm",
        version=version,
    )


def make_recipe() -> QuantRecipe:
    return QuantRecipe(
        algorithm="rtn",
        scheme="w4a16",
        weight_granularity="group",
        weight_group_size=128,
        weight_symmetric=True,
        activation_mode="none",
    )


def make_plan(
    *,
    model: ModelIdentity | None = None,
    hardware: HardwareTarget | None = None,
    runtime: RuntimeTarget | None = None,
) -> QuantPlan:
    return QuantPlan(
        model=model or make_model(),
        recipe=make_recipe(),
        hardware=hardware or make_hardware(),
        runtime=runtime or make_runtime(),
        backend="llm_compressor",
        backend_version="0.0.0-test",
        resolved_modules=("model.layers.0.mlp.down_proj",),
    )


def make_artifact(
    *,
    model_sha: str = SHA_B,
) -> QuantArtifactIdentity:
    return QuantArtifactIdentity(
        layout="huggingface_pretrained",
        serialization_format="compressed_tensors",
        files=(
            ArtifactFileIdentity(
                relative_path="config.json",
                sha256=SHA_A,
                size_bytes=512,
            ),
            ArtifactFileIdentity(
                relative_path="model.safetensors",
                sha256=model_sha,
                size_bytes=4096,
            ),
        ),
    )


def make_failure() -> FailureRecord:
    return FailureRecord(
        stage="runtime_validation",
        category="runtime_mismatch",
        message="Declared runtime rejected the produced artifact.",
    )


def make_evidence_file(
    *,
    relative_path: str = "environment/fingerprint.json",
    category: str = "environment_fingerprint",
    sha256: str = SHA_C,
) -> EvidenceFileReference:
    payload: dict[str, object] = {
        "category": category,
        "relative_path": relative_path,
        "sha256": sha256,
        "size_bytes": 1024,
    }

    return EvidenceFileReference.model_validate(payload)


def make_baseline_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        schema_version="1",
        quant_version="0.0.0",
        run_id="baseline-run-001",
        run_kind="baseline",
        execution_status="COMPLETED",
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
        model=make_model(),
        tokenizer=make_tokenizer(),
        hardware=make_hardware(),
        runtime=make_runtime(),
        evidence_files=(make_evidence_file(),),
    )


def make_candidate_bundle(
    *,
    artifact: QuantArtifactIdentity | None = None,
    execution_status: str = "COMPLETED",
    failures: tuple[FailureRecord, ...] = (),
) -> EvidenceBundle:
    model = make_model()
    hardware = make_hardware()
    runtime = make_runtime()
    selected_artifact = artifact or make_artifact()

    payload: dict[str, object] = {
        "schema_version": "1",
        "quant_version": "0.0.0",
        "run_id": "candidate-run-001",
        "run_kind": "candidate",
        "execution_status": execution_status,
        "started_at": STARTED_AT,
        "finished_at": FINISHED_AT,
        "model": model,
        "tokenizer": make_tokenizer(),
        "hardware": hardware,
        "runtime": runtime,
        "plan": make_plan(
            model=model,
            hardware=hardware,
            runtime=runtime,
        ),
        "artifact": selected_artifact,
        "failures": failures,
    }

    return EvidenceBundle.model_validate(payload)


def test_baseline_bundle_serializes_json_compatible_values() -> None:
    payload = make_baseline_bundle().model_dump(mode="json")

    assert payload["schema_version"] == "1"
    assert payload["run_kind"] == "baseline"
    assert payload["execution_status"] == "COMPLETED"
    assert payload["started_at"] == "2026-09-25T12:00:00Z"
    assert payload["finished_at"] == "2026-09-25T12:05:00Z"
    assert payload["plan"] is None
    assert payload["artifact"] is None
    assert payload["qualification"] is None
    assert payload["failures"] == []


def test_candidate_bundle_preserves_plan_and_artifact() -> None:
    bundle = make_candidate_bundle()

    assert bundle.plan is not None
    assert bundle.artifact == make_artifact()


@pytest.mark.parametrize(
    "category",
    [
        "environment_fingerprint",
        "calibration_manifest",
        "evaluation_manifest",
        "execution_log",
        "reproducibility_metadata",
        "backend_metadata",
        "runtime_validation",
        "comparison_evidence",
        "other",
    ],
)
def test_evidence_file_reference_accepts_categories(
    category: str,
) -> None:
    reference = make_evidence_file(category=category)

    assert reference.category == category


@pytest.mark.parametrize(
    "relative_path",
    [
        "/logs/run.jsonl",
        "../logs/run.jsonl",
        "logs/../run.jsonl",
        "./logs/run.jsonl",
        "logs//run.jsonl",
        "logs\\run.jsonl",
        "logs/",
        ".",
        "..",
    ],
)
def test_evidence_file_reference_rejects_unsafe_paths(
    relative_path: str,
) -> None:
    with pytest.raises(ValidationError):
        make_evidence_file(relative_path=relative_path)


@pytest.mark.parametrize(
    "sha256",
    [
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
    ],
)
def test_evidence_file_reference_rejects_invalid_sha256(
    sha256: str,
) -> None:
    with pytest.raises(ValidationError):
        make_evidence_file(sha256=sha256)


def test_evidence_file_reference_accepts_zero_size() -> None:
    reference = EvidenceFileReference(
        category="execution_log",
        relative_path="logs/empty.log",
        sha256=SHA_A,
        size_bytes=0,
    )

    assert reference.size_bytes == 0


def test_evidence_file_reference_rejects_negative_size() -> None:
    payload = make_evidence_file().model_dump()
    payload["size_bytes"] = -1

    with pytest.raises(ValidationError):
        EvidenceFileReference.model_validate(payload)


@pytest.mark.parametrize(
    "stage",
    [
        "model_load",
        "planning",
        "calibration",
        "quantization",
        "artifact_save",
        "artifact_load",
        "runtime_validation",
        "quality_evaluation",
        "performance_benchmark",
        "comparison",
        "qualification",
        "export",
        "unknown",
    ],
)
def test_failure_record_accepts_stages(stage: str) -> None:
    payload: dict[str, object] = {
        "stage": stage,
        "category": "test_failure",
        "message": "Observed failure.",
    }

    record = FailureRecord.model_validate(payload)

    assert record.stage == stage


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("category", ""),
        ("category", "   "),
        ("message", ""),
        ("message", "   "),
    ],
)
def test_failure_record_rejects_empty_text(
    field_name: str,
    value: str,
) -> None:
    payload: dict[str, object] = make_failure().model_dump()
    payload[field_name] = value

    with pytest.raises(ValidationError):
        FailureRecord.model_validate(payload)


def test_bundle_rejects_naive_started_at() -> None:
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["started_at"] = datetime(2026, 9, 25, 12, 0)

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_bundle_rejects_naive_finished_at() -> None:
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["finished_at"] = datetime(2026, 9, 25, 12, 5)

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_bundle_rejects_finished_at_before_started_at() -> None:
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["finished_at"] = STARTED_AT - timedelta(seconds=1)

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_baseline_rejects_plan() -> None:
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["plan"] = make_plan()

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_baseline_rejects_artifact() -> None:
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["artifact"] = make_artifact()

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_baseline_rejects_qualification() -> None:
    artifact = make_artifact()
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["qualification"] = QualificationResult(
        candidate_artifact=artifact,
        state="ELIGIBLE",
        pareto_optimal=True,
    )

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_candidate_requires_plan() -> None:
    payload: dict[str, object] = make_candidate_bundle().model_dump()
    payload["plan"] = None

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_completed_candidate_requires_artifact() -> None:
    payload: dict[str, object] = make_candidate_bundle().model_dump()
    payload["artifact"] = None

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_candidate_rejects_plan_model_mismatch() -> None:
    payload: dict[str, object] = make_candidate_bundle().model_dump()
    mismatched_model = ModelIdentity(
        model_id="org/other-model",
        requested_revision="main",
        resolved_revision="c" * 40,
    )
    payload["plan"] = make_plan(model=mismatched_model)

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_candidate_rejects_plan_hardware_mismatch() -> None:
    payload: dict[str, object] = make_candidate_bundle().model_dump()
    payload["plan"] = make_plan(
        hardware=make_hardware(device_name="NVIDIA A40"),
    )

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_candidate_rejects_plan_runtime_mismatch() -> None:
    payload: dict[str, object] = make_candidate_bundle().model_dump()
    payload["plan"] = make_plan(runtime=make_runtime(version="9.9.9"))

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_qualification_requires_artifact() -> None:
    artifact = make_artifact()
    payload: dict[str, object] = make_candidate_bundle().model_dump()
    payload["artifact"] = None
    payload["qualification"] = QualificationResult(
        candidate_artifact=artifact,
        state="ELIGIBLE",
        pareto_optimal=True,
    )

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_qualification_artifact_must_match_bundle_artifact() -> None:
    artifact = make_artifact()
    other_artifact = make_artifact(model_sha=SHA_C)
    payload: dict[str, object] = make_candidate_bundle(
        artifact=artifact,
    ).model_dump()
    payload["qualification"] = QualificationResult(
        candidate_artifact=other_artifact,
        state="ELIGIBLE",
        pareto_optimal=True,
    )

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_completed_run_rejects_failures() -> None:
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["failures"] = (make_failure(),)

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


@pytest.mark.parametrize(
    "execution_status",
    ["FAILED", "BLOCKED"],
)
def test_noncompleted_run_requires_failure(
    execution_status: str,
) -> None:
    payload: dict[str, object] = make_candidate_bundle().model_dump()
    payload["execution_status"] = execution_status
    payload["artifact"] = None
    payload["failures"] = ()

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


@pytest.mark.parametrize(
    "execution_status",
    ["FAILED", "BLOCKED"],
)
def test_noncompleted_candidate_accepts_early_failure_without_artifact(
    execution_status: str,
) -> None:
    payload: dict[str, object] = make_candidate_bundle().model_dump()
    payload["execution_status"] = execution_status
    payload["artifact"] = None
    payload["benchmarks"] = ()
    payload["qualification"] = None
    payload["failures"] = (make_failure(),)

    bundle = EvidenceBundle.model_validate(payload)

    assert bundle.artifact is None
    assert bundle.benchmarks == ()
    assert bundle.qualification is None


def test_blocked_qualification_requires_blocked_execution() -> None:
    artifact = make_artifact()
    payload: dict[str, object] = make_candidate_bundle(
        artifact=artifact,
    ).model_dump()
    payload["qualification"] = QualificationResult(
        candidate_artifact=artifact,
        state="BLOCKED",
        reason_codes=("runtime_validation_missing",),
    )

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_blocked_bundle_accepts_blocked_qualification() -> None:
    artifact = make_artifact()
    payload: dict[str, object] = make_candidate_bundle(
        artifact=artifact,
    ).model_dump()
    payload["execution_status"] = "BLOCKED"
    payload["failures"] = (make_failure(),)
    payload["qualification"] = QualificationResult(
        candidate_artifact=artifact,
        state="BLOCKED",
        reason_codes=("runtime_validation_missing",),
    )

    bundle = EvidenceBundle.model_validate(payload)

    assert bundle.qualification is not None
    assert bundle.qualification.state == "BLOCKED"


def test_bundle_rejects_duplicate_evidence_paths() -> None:
    reference = make_evidence_file()
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["evidence_files"] = (reference, reference)

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_bundle_rejects_unsorted_evidence_paths() -> None:
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["evidence_files"] = (
        make_evidence_file(
            relative_path="logs/run.jsonl",
            category="execution_log",
            sha256=SHA_B,
        ),
        make_evidence_file(
            relative_path="environment/fingerprint.json",
            category="environment_fingerprint",
            sha256=SHA_A,
        ),
    )

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_bundle_accepts_sorted_evidence_paths() -> None:
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["evidence_files"] = (
        make_evidence_file(
            relative_path="environment/fingerprint.json",
            category="environment_fingerprint",
            sha256=SHA_A,
        ),
        make_evidence_file(
            relative_path="logs/run.jsonl",
            category="execution_log",
            sha256=SHA_B,
        ),
    )

    bundle = EvidenceBundle.model_validate(payload)

    assert len(bundle.evidence_files) == 2


def test_bundle_uses_strict_tuple_typing() -> None:
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["evidence_files"] = [make_evidence_file()]

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_evidence_bundle_rejects_extra_fields() -> None:
    payload: dict[str, object] = make_baseline_bundle().model_dump()
    payload["public_record_id"] = "unexpected"

    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


def test_evidence_bundle_is_immutable() -> None:
    bundle = make_baseline_bundle()

    with pytest.raises(ValidationError):
        bundle.__setattr__("execution_status", "FAILED")
