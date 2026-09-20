"""Bucket classification for the localization error analysis.

The regression that matters here is `packed_file_paths`: the first version of this
analysis asked the dataset which files had content, but the packer keeps only
`evidence.MAX_FILES` of them. That counted "the model never saw this file" as
"the model saw it and chose wrong" and pointed at the opposite conclusion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ci_triage.agent import evidence as ev
from ci_triage.evaluation.headroom import analyse, classify, packed_file_paths
from ci_triage.evaluation.runner import Prediction
from ci_triage.miner.schema import CaseRecord

RED = "a" * 40


def _file(path: str, content: str) -> dict:
    return {
        "path": path,
        "ref": RED,
        "reasons": ["log_reference"],
        "content": content,
        "truncated": False,
    }


def _case(record_dict, *, gold: list[str], **input_overrides) -> CaseRecord:
    data = record_dict(**input_overrides)
    data["ground_truth"]["fix_window"]["files_changed"] = gold
    return CaseRecord.model_validate(data)


def _prediction(case: CaseRecord, files: list[str]) -> Prediction:
    return Prediction(
        case_id=case.case_id,
        system="test",
        latency_ms=1.0,
        diagnosis={
            "failure_type": "TEST_FAILURE",
            "root_cause": "x",
            "confidence": 0.5,
            "evidence": [],
            "affected_files": files,
        },
    )


def test_packed_file_paths_reports_only_what_the_packer_emitted(record_dict):
    """MAX_FILES files are sent, however many the dataset carries."""
    many = [_file(f"src/mod{i}.py", f"# content {i}\n" * 20) for i in range(5)]
    case = _case(record_dict, gold=["src/mod4.py"], relevant_files=many)
    pack = ev.pack(case.visible())

    sent = packed_file_paths(pack)

    assert len(sent) == ev.MAX_FILES
    assert sent == {"src/mod0.py", "src/mod1.py"}
    assert "src/mod4.py" not in sent  # in the dataset, never in the prompt


def test_file_whose_content_was_never_sent_is_name_only_not_reasoning(record_dict, tmp_path):
    """The bug: gold file known by name, content squeezed out -> a tool could fetch it."""
    files = [_file(f"src/mod{i}.py", "x = 1\n" * 50) for i in range(4)]
    case = _case(
        record_dict,
        gold=["src/mod3.py"],
        relevant_files=files,
        log_referenced_files=["src/mod3.py"],
    )
    analysis = classify(case, _prediction(case, ["src/mod0.py"]), tmp_path)

    assert analysis is not None
    assert analysis.bucket == "name_only"
    assert "src/mod3.py" not in analysis.content_sent


def test_file_whose_content_was_sent_is_a_reasoning_failure(record_dict, tmp_path):
    case = _case(
        record_dict,
        gold=["src/mod0.py"],
        relevant_files=[_file("src/mod0.py", "x = 1\n"), _file("src/mod1.py", "y = 2\n")],
    )
    analysis = classify(case, _prediction(case, ["src/mod1.py"]), tmp_path)

    assert analysis is not None
    assert analysis.bucket == "reasoning"
    assert "src/mod0.py" in analysis.content_sent


def test_correct_top_file_is_a_hit(record_dict, tmp_path):
    case = _case(record_dict, gold=["app.py"], relevant_files=[_file("app.py", "x = 1\n")])
    analysis = classify(case, _prediction(case, ["app.py", "other.py"]), tmp_path)

    assert analysis is not None
    assert analysis.bucket == "hit"


def test_path_absent_from_evidence_with_no_clone_is_impossible(record_dict, tmp_path):
    """Without a clone we cannot prove the file existed, so it must not be claimed
    as tool-reachable: the count that drives a build decision stays conservative."""
    case = _case(record_dict, gold=["deep/hidden.py"], relevant_files=[])
    analysis = classify(case, _prediction(case, ["app.py"]), tmp_path)

    assert analysis is not None
    assert analysis.bucket == "impossible"


def test_cases_without_a_gold_code_fix_are_skipped(record_dict, tmp_path):
    """Non-code fixes are excluded from localization, so they cannot bias the buckets."""
    data = record_dict()
    data["ground_truth"]["fix_status"] = "flaky_rerun"
    case = CaseRecord.model_validate(data)

    assert classify(case, _prediction(case, ["app.py"]), tmp_path) is None


def test_analyse_counts_buckets_and_reachable_share(record_dict, tmp_path):
    hit = _case(record_dict, gold=["app.py"], relevant_files=[_file("app.py", "x\n")])
    hit = CaseRecord.model_validate({**hit.model_dump(mode="json"), "case_id": "o__r__hit"})
    name_only = _case(
        record_dict,
        gold=["src/mod3.py"],
        relevant_files=[_file(f"src/mod{i}.py", "x = 1\n" * 50) for i in range(4)],
        log_referenced_files=["src/mod3.py"],
    )
    name_only = CaseRecord.model_validate(
        {**name_only.model_dump(mode="json"), "case_id": "o__r__name"}
    )
    predictions = [
        _prediction(hit, ["app.py"]),
        _prediction(name_only, ["src/mod0.py"]),
    ]
    for pred, case in zip(predictions, [hit, name_only], strict=True):
        pred.case_id = case.case_id

    result = analyse([hit, name_only], predictions, tmp_path)

    assert result["cases_scored"] == 2
    assert result["buckets"]["hit"] == 1
    assert result["buckets"]["name_only"] == 1
    assert result["tools_reachable"] == 1
    assert result["tools_reachable_share"] == 0.5
    assert result["ceiling_if_reachable_recovered"] == 1.0


@pytest.mark.parametrize("missing", [None, Path("no/such/dir")])
def test_missing_clone_directory_does_not_crash(record_dict, missing):
    case = _case(record_dict, gold=["deep/hidden.py"], relevant_files=[])
    analysis = classify(case, _prediction(case, []), missing or Path("also/missing"))

    assert analysis is not None
