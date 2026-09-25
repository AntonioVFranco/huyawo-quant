"""Tests for candidate qualification result contracts."""

import pytest
from huyawo_quant.contracts import (
    ArtifactFileIdentity,
    QualificationResult,
    QuantArtifactIdentity,
)
from pydantic import ValidationError

SHA_A = "a" * 64
SHA_B = "b" * 64


def make_candidate_artifact() -> QuantArtifactIdentity:
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
                sha256=SHA_B,
                size_bytes=4096,
            ),
        ),
    )


def make_eligible_result(
    *,
    pareto_optimal: bool = True,
) -> QualificationResult:
    return QualificationResult(
        candidate_artifact=make_candidate_artifact(),
        state="ELIGIBLE",
        pareto_optimal=pareto_optimal,
    )


def test_eligible_result_serializes_expected_fields() -> None:
    result = make_eligible_result()

    assert result.model_dump(mode="json") == {
        "candidate_artifact": {
            "layout": "huggingface_pretrained",
            "serialization_format": "compressed_tensors",
            "files": [
                {
                    "relative_path": "config.json",
                    "sha256": SHA_A,
                    "size_bytes": 512,
                },
                {
                    "relative_path": "model.safetensors",
                    "sha256": SHA_B,
                    "size_bytes": 4096,
                },
            ],
        },
        "state": "ELIGIBLE",
        "reason_codes": [],
        "pareto_optimal": True,
        "selection_objective": None,
    }


def test_qualification_result_preserves_candidate_artifact() -> None:
    artifact = make_candidate_artifact()

    result = QualificationResult(
        candidate_artifact=artifact,
        state="ELIGIBLE",
        pareto_optimal=False,
    )

    assert result.candidate_artifact == artifact


@pytest.mark.parametrize(
    ("state", "reason_codes", "pareto_optimal", "selection_objective"),
    [
        ("ELIGIBLE", (), False, None),
        ("ELIGIBLE", (), True, None),
        ("INELIGIBLE", ("quality_threshold_failed",), None, None),
        ("BLOCKED", ("runtime_validation_missing",), None, None),
        ("PREFERRED", (), True, "minimum VRAM subject to quality threshold"),
    ],
)
def test_qualification_result_accepts_valid_states(
    state: str,
    reason_codes: tuple[str, ...],
    pareto_optimal: bool | None,
    selection_objective: str | None,
) -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": state,
        "reason_codes": reason_codes,
        "pareto_optimal": pareto_optimal,
        "selection_objective": selection_objective,
    }

    result = QualificationResult.model_validate(payload)

    assert result.state == state


def test_qualification_result_rejects_unknown_state() -> None:
    payload: dict[str, object] = make_eligible_result().model_dump()
    payload["state"] = "UNKNOWN"

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


@pytest.mark.parametrize(
    "state",
    ["INELIGIBLE", "BLOCKED"],
)
def test_noneligible_states_require_reason_codes(
    state: str,
) -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": state,
        "reason_codes": (),
        "pareto_optimal": None,
        "selection_objective": None,
    }

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


@pytest.mark.parametrize(
    "state",
    ["INELIGIBLE", "BLOCKED"],
)
def test_noneligible_states_require_null_pareto_state(
    state: str,
) -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": state,
        "reason_codes": ("constraint_failed",),
        "pareto_optimal": False,
        "selection_objective": None,
    }

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


@pytest.mark.parametrize(
    "state",
    ["INELIGIBLE", "BLOCKED"],
)
def test_noneligible_states_reject_selection_objective(
    state: str,
) -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": state,
        "reason_codes": ("constraint_failed",),
        "pareto_optimal": None,
        "selection_objective": "minimum VRAM",
    }

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


def test_eligible_requires_explicit_pareto_state() -> None:
    with pytest.raises(ValidationError):
        QualificationResult(
            candidate_artifact=make_candidate_artifact(),
            state="ELIGIBLE",
        )


def test_eligible_rejects_reason_codes() -> None:
    with pytest.raises(ValidationError):
        QualificationResult(
            candidate_artifact=make_candidate_artifact(),
            state="ELIGIBLE",
            reason_codes=("informational_reason",),
            pareto_optimal=True,
        )


def test_eligible_rejects_selection_objective() -> None:
    with pytest.raises(ValidationError):
        QualificationResult(
            candidate_artifact=make_candidate_artifact(),
            state="ELIGIBLE",
            pareto_optimal=True,
            selection_objective="minimum VRAM",
        )


def test_preferred_requires_pareto_true() -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": "PREFERRED",
        "reason_codes": (),
        "pareto_optimal": False,
        "selection_objective": "minimum VRAM",
    }

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


def test_preferred_requires_selection_objective() -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": "PREFERRED",
        "reason_codes": (),
        "pareto_optimal": True,
        "selection_objective": None,
    }

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


@pytest.mark.parametrize(
    "selection_objective",
    ["", "   "],
)
def test_preferred_rejects_empty_selection_objective(
    selection_objective: str,
) -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": "PREFERRED",
        "reason_codes": (),
        "pareto_optimal": True,
        "selection_objective": selection_objective,
    }

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


def test_preferred_rejects_reason_codes() -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": "PREFERRED",
        "reason_codes": ("unexpected_reason",),
        "pareto_optimal": True,
        "selection_objective": "minimum VRAM",
    }

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


@pytest.mark.parametrize(
    "reason_code",
    ["", "   "],
)
def test_reason_codes_must_be_non_empty(
    reason_code: str,
) -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": "BLOCKED",
        "reason_codes": (reason_code,),
        "pareto_optimal": None,
        "selection_objective": None,
    }

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


def test_reason_codes_must_be_unique() -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": "BLOCKED",
        "reason_codes": (
            "artifact_load_failed",
            "artifact_load_failed",
        ),
        "pareto_optimal": None,
        "selection_objective": None,
    }

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


def test_reason_codes_must_be_sorted() -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": "BLOCKED",
        "reason_codes": (
            "runtime_validation_missing",
            "artifact_load_failed",
        ),
        "pareto_optimal": None,
        "selection_objective": None,
    }

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


def test_sorted_reason_codes_are_accepted() -> None:
    result = QualificationResult(
        candidate_artifact=make_candidate_artifact(),
        state="BLOCKED",
        reason_codes=(
            "artifact_load_failed",
            "runtime_validation_missing",
        ),
    )

    assert result.reason_codes == (
        "artifact_load_failed",
        "runtime_validation_missing",
    )


def test_reason_codes_use_strict_tuple_typing() -> None:
    payload: dict[str, object] = {
        "candidate_artifact": make_candidate_artifact(),
        "state": "BLOCKED",
        "reason_codes": ["artifact_load_failed"],
        "pareto_optimal": None,
        "selection_objective": None,
    }

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


def test_qualification_result_rejects_extra_fields() -> None:
    payload: dict[str, object] = make_eligible_result().model_dump()
    payload["benchmark_delta"] = 0.10

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(payload)


def test_qualification_result_is_immutable() -> None:
    result = make_eligible_result()

    with pytest.raises(ValidationError):
        result.__setattr__("state", "PREFERRED")
