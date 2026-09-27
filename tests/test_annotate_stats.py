import pytest

from ci_triage.miner.annotate import (
    apply_reviews,
    build_queue,
    render_case,
    run_review,
    save_cases,
)
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
    assert stats["reviewed"] == {"human_verified": 2, "model_reviewed": 0}
    assert stats["streaks_examined"] == 3
    assert stats["rejection_reasons"] == {"no_green_found": 1}


def test_quit_keeps_earlier_answers(tmp_path, make_case):
    path = tmp_path / "cases.jsonl"
    save_cases(path, [make_case("r0", "needs_review"), make_case("r1", "needs_review")])
    answers = iter(["", "cause", "fix", "", "q"])
    assert run_review(path, 0, input_fn=lambda _: next(answers), out=lambda _: None) == 1
    statuses = [c.labels.label_status for c in load_cases(path)]
    assert statuses == ["human_verified", "needs_review"]


# ------------------------------------------------- model review, and the blind audit


def test_blind_render_hides_everything_the_auto_labeler_decided(make_case):
    """The audit sample measures how often the automatic label is right. A reviewer who
    has seen that label - or merely that this is an *audit* case, which means the labeler
    was confident - agrees with it more often, and the resulting precision figure then
    describes the anchoring instead of the labeler."""
    case = make_case("c1", "auto_verified", "LINT_FAILURE")

    blind = render_case(case, "1/1", "audit", blind=True)
    shown = render_case(case, "1/1", "audit", blind=False)

    for leak in ("automatic label", "LINT_FAILURE", "log signal", "audit", "auto rule"):
        assert leak not in blind, f"blind render leaked {leak!r}"
    assert "LINT_FAILURE" in shown and "automatic label" in shown
    # The evidence itself must survive, or there is nothing to review.
    assert "HINDSIGHT" in blind and case.case_id in blind


def test_applying_a_model_review_records_who_decided(make_case, tmp_path):
    path = tmp_path / "cases.jsonl"
    save_cases(path, [make_case("c1", "needs_review", "TEST_FAILURE")])

    result = apply_reviews(
        path,
        {"c1": {"category": "LINT_FAILURE", "root_cause": "ruff", "fix_text": "ran ruff"}},
        "model_reviewed",
    )

    (case,) = load_cases(path)
    assert result == {"applied": 1, "declined": 0, "unchanged": 0}
    assert case.labels.label_status == "model_reviewed"  # never "human_verified"
    assert case.labels.category == FailureCategory.LINT_FAILURE
    assert case.labels.auto_category == FailureCategory.TEST_FAILURE  # kept for precision
    assert case.labels.root_cause_text == "ruff"


def test_declining_a_case_keeps_it_for_a_human(make_case, tmp_path):
    """Forcing a category onto evidence that does not support one writes a wrong gold
    label, and every system is then scored against it. An unanswered case is cheaper."""
    path = tmp_path / "cases.jsonl"
    save_cases(path, [make_case("c1", "needs_review", "TEST_FAILURE")])

    result = apply_reviews(
        path,
        {"c1": {"category": None, "notes": "log and fix disagree, cannot tell"}},
        "model_reviewed",
    )

    (case,) = load_cases(path)
    assert result == {"applied": 0, "declined": 1, "unchanged": 0}
    assert case.labels.label_status == "needs_review"  # still queued for a person
    assert case.labels.category == FailureCategory.TEST_FAILURE  # untouched
    assert "cannot tell" in case.labels.reviewer_notes


def test_a_reviewed_case_leaves_the_queue_but_a_declined_one_does_not(make_case, tmp_path):
    path = tmp_path / "cases.jsonl"
    save_cases(
        path,
        [make_case("c1", "needs_review"), make_case("c2", "needs_review")],
    )
    apply_reviews(
        path,
        {"c1": {"category": "LINT_FAILURE"}, "c2": {"category": None, "notes": "unclear"}},
        "model_reviewed",
    )

    queue = build_queue(load_cases(path), audit_size=0)
    assert [cid for cid, _ in queue] == ["c2"]


def test_applying_an_unknown_case_id_is_refused(make_case, tmp_path):
    path = tmp_path / "cases.jsonl"
    save_cases(path, [make_case("c1", "needs_review")])

    with pytest.raises(ValueError, match="no such case"):
        apply_reviews(path, {"typo": {"category": "LINT_FAILURE"}}, "model_reviewed")

    assert load_cases(path)[0].labels.label_status == "needs_review"  # nothing written


def test_stats_never_merges_model_review_into_human_review(make_case):
    cases = [
        make_case("h1", "human_verified"),
        make_case("m1", "model_reviewed"),
        make_case("m2", "model_reviewed"),
        make_case("n1", "needs_review"),
    ]

    stats = compute_stats(cases, [])

    assert stats["reviewed"] == {"human_verified": 1, "model_reviewed": 2}
    assert stats["label_status"]["model_reviewed"] == 2
