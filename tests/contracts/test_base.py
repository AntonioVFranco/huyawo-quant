"""Tests for the Huyawo Quant contract base."""

import pytest
from huyawo_quant.contracts import ContractModel
from pydantic import ValidationError


class ExampleContract(ContractModel):
    value: str


def test_contract_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ExampleContract.model_validate(
            {
                "value": "expected",
                "unexpected": "not allowed",
            }
        )


def test_contract_uses_strict_validation() -> None:
    with pytest.raises(ValidationError):
        ExampleContract.model_validate({"value": 123})


def test_contract_is_immutable() -> None:
    contract = ExampleContract(value="expected")

    with pytest.raises(ValidationError):
        contract.value = "changed"


def test_contract_serializes_to_json_compatible_data() -> None:
    contract = ExampleContract(value="expected")

    assert contract.model_dump(mode="json") == {"value": "expected"}
