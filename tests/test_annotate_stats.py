import pytest

from ci_triage.miner.annotate import build_queue, run_review, save_cases
from ci_triage.miner.schema import CaseRecord
from ci_triage.miner.stats import compute_stats, load_cases
from ci_triage.taxonomy import FailureCategory


@pytest.fixture
def make_case(record_dict):
    def make(case_id: str, status: str, category: str = "TEST_FAILURE") -> CaseRecord:
        data = record_dict()
        data["case_id"] = case_id
        data["labels"]["label_status"] = status
        data["labels"]["category"] = category
        data["labels"]["auto_category"] = category
        return CaseRecord.model_validate(data)

    return make


def test_queue_has_all_review_cases_and_a_reproducible_audit_sample(make_case):
    cases = [make_case(f"r{i}", "needs_review") for i in range(2)]
    cases += [make_case(f"a{i}", "auto_verified") for i in range(10)]
    queue = build_queue(cases, audit_size=3, seed=7)
    assert [cid for cid, why in queue if why == "review"] == ["r0", "r1"]
    audit = [cid for cid, why in queue if why == "audit"]
    assert len(audit) == 3
    assert audit == [cid for cid, why in build_queue(cases, 3, seed=7) if why == "audit"]


def test_review_session_updates_labels_and_audit_precision(tmp_path, make_case):
    path = tmp_path / "cases.jsonl"
    save_cases(
        path,
        [
            make_case("r0", "needs_review", "LINT_FAILURE"),
            make_case("a0", "auto_verified", "TEST_FAILURE"),
        ],
    )
    format_index = str(list(FailureCategory).index(FailureCategory.FORMAT_FAILURE))
    answers = iter(
        [
            "banana",  # invalid category -> asked again
            format_index,  # review case: human changes LINT -> FORMAT
            "black reformatted a file",
            "ran the formatter",
            "",
            "",  # audit case: Enter keeps the automatic category
            "assertion on add()",
            "restored addition",
            "",
        ]
    )
    output: list[str] = []
    reviewed = run_review(path, audit_size=1, input_fn=lambda _: next(answers), out=output.append)
    assert reviewed == 2
    assert any("invalid choice" in line for line in output)

    cases = {c.case_id: c for c in load_cases(path)}
    r0 = cases["r0"].labels
    assert (r0.category, r0.auto_category, r0.label_status) == (
        "FORMAT_FAILURE",
        "LINT_FAILURE",
        "human_verified",
    )
    assert r0.root_cause_text == "black reformatted a file" and not r0.audited
    assert cases["a0"].labels.audited

    stats = compute_stats(list(cases.values()), [{"reason": "no_green_found"}])
    assert stats["audit"] == {"audited": 1, "auto_label_agreed": 1, "precision": 1.0}
    assert stats["human_reviewed"] == 2
    assert stats["streaks_examined"] == 3
    assert stats["rejection_reasons"] == {"no_green_found": 1}


def test_quit_keeps_earlier_answers(tmp_path, make_case):
    path = tmp_path / "cases.jsonl"
    save_cases(path, [make_case("r0", "needs_review"), make_case("r1", "needs_review")])
    answers = iter(["", "cause", "fix", "", "q"])
    assert run_review(path, 0, input_fn=lambda _: next(answers), out=lambda _: None) == 1
    statuses = [c.labels.label_status for c in load_cases(path)]
    assert statuses == ["human_verified", "needs_review"]
