"""Candidate qualification result contracts."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from huyawo_quant.contracts._base import ContractModel
from huyawo_quant.contracts.artifacts import QuantArtifactIdentity

NonEmptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]


class QualificationResult(ContractModel):
    """Experiment-local qualification state for one candidate artifact."""

    candidate_artifact: QuantArtifactIdentity
    state: Literal["ELIGIBLE", "INELIGIBLE", "BLOCKED", "PREFERRED"]
    reason_codes: tuple[NonEmptyString, ...] = ()
    pareto_optimal: bool | None = None
    selection_objective: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_qualification_state(self) -> Self:
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("reason_codes must be unique")

        if self.reason_codes != tuple(sorted(self.reason_codes)):
            raise ValueError("reason_codes must be ordered lexicographically")

        if self.state in {"INELIGIBLE", "BLOCKED"}:
            if not self.reason_codes:
                raise ValueError(f"{self.state} requires at least one reason code")

            if self.pareto_optimal is not None:
                raise ValueError(f"{self.state} requires pareto_optimal to be null")

            if self.selection_objective is not None:
                raise ValueError(f"{self.state} must not include selection_objective")

            return self

        if self.state == "ELIGIBLE":
            if self.reason_codes:
                raise ValueError("ELIGIBLE requires an empty reason_codes tuple")

            if self.pareto_optimal is None:
                raise ValueError("ELIGIBLE requires explicit pareto_optimal state")

            if self.selection_objective is not None:
                raise ValueError("ELIGIBLE must not include selection_objective")

            return self

        if self.reason_codes:
            raise ValueError("PREFERRED requires an empty reason_codes tuple")

        if self.pareto_optimal is not True:
            raise ValueError("PREFERRED requires pareto_optimal to be true")

        if self.selection_objective is None:
            raise ValueError("PREFERRED requires selection_objective")

        return self
