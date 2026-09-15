import pytest
from pydantic import ValidationError

from ci_triage.miner.schema import CaseRecord

GREEN = "b" * 40  # the fix-window head SHA used by the record_dict fixture


def test_valid_record(record_dict):
    case = CaseRecord.model_validate(record_dict())
    assert case.schema_version == "1.0"


def test_fix_sha_leaking_into_input_is_rejected(record_dict):
    with pytest.raises(ValidationError, match="leaked"):
        CaseRecord.model_validate(record_dict(log_excerpt=f"see fix {GREEN}"))


def test_unknown_fields_are_rejected(record_dict):
    with pytest.raises(ValidationError):
        CaseRecord.model_validate(record_dict(fix_diff="sneaky"))


def test_unknown_category_is_rejected(record_dict):
    bad = record_dict()
    bad["labels"]["category"] = "COSMIC_RAYS"
    with pytest.raises(ValidationError):
        CaseRecord.model_validate(bad)
