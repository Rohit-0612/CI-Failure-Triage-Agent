import pytest

from ci_triage.diagnosis import Diagnosis, Evidence
from ci_triage.evaluation import metrics
from ci_triage.evaluation import runner as runner_mod
from ci_triage.evaluation.runner import Prediction, evaluate, gold_fix_files, run_system
from ci_triage.miner.schema import CaseRecord

# ----------------------------------------------------------------------------- metrics


def test_wilson_interval_known_values():
    assert metrics.wilson_interval(5, 10) == (0.237, 0.763)
    assert metrics.wilson_interval(0, 10) == (0.0, 0.278)
    assert metrics.wilson_interval(0, 0) is None


def test_localization_scores():
    scores = metrics.localization(["a.py", "b.py", "c.py"], {"b.py", "z.py"})
    assert scores == {"hit@1": 0.0, "hit@3": 1.0, "rr": 0.5, "recall@5": 0.5}
    assert metrics.localization([], {"a.py"})["rr"] == 0.0


def test_category_accuracy_and_confusion():
    result = metrics.category_accuracy([("A", "A"), ("A", "B"), ("B", "B")])
    assert (result["hits"], result["n"]) == (2, 3)
    assert result["confusion"] == {"A": {"A": 1, "B": 1}, "B": {"B": 1}}


# ----------------------------------------------------------------------------- runner


@pytest.fixture
def cases(record_dict):
    def case(case_id, category, status, fix_status="matched", files=("app.py",)):
        data = record_dict()
        data["case_id"] = case_id
        data["labels"]["category"] = category
        data["labels"]["label_status"] = status
        data["ground_truth"]["fix_status"] = fix_status
        data["ground_truth"]["fix_window"]["files_changed"] = list(files)
        return CaseRecord.model_validate(data)

    return [
        case("c1", "TEST_FAILURE", "auto_verified"),
        case("c2", "LINT_FAILURE", "needs_review", files=("docs/index.md",)),
        case("c3", "POLICY_CHECK_FAILURE", "needs_review"),
        case("c4", "FLAKY", "auto_verified", fix_status="flaky_rerun", files=()),
    ]


def diag(category, files, excerpt="FAILED t.py::test"):
    return Diagnosis(
        failure_type=category,
        root_cause="x",
        confidence=0.5,
        evidence=[Evidence(source="ci_log", location="log", excerpt=excerpt, explanation="")],
        affected_files=files,
    )


def test_gold_fix_files_excludes_non_code_fixes(cases):
    reasons = [gold_fix_files(c)[1] for c in cases]
    assert reasons == [
        "eligible",
        "fix_changes_only_docs",
        "non_code_category:POLICY_CHECK_FAILURE",
        "fix_status:flaky_rerun",
    ]


def test_evaluate_scores_categories_localization_and_grounding(cases):
    predictions = [
        Prediction(case_id="c1", system="s", diagnosis=diag("TEST_FAILURE", ["x.py", "app.py"]),
                   latency_ms=1.0),
        Prediction(case_id="c2", system="s", diagnosis=diag("FORMAT_FAILURE", []), latency_ms=2.0),
        Prediction(case_id="c3", system="s",
                   diagnosis=diag("POLICY_CHECK_FAILURE", [], excerpt="made up line"),
                   latency_ms=3.0),
        Prediction(case_id="c4", system="s", diagnosis=None, error="boom", latency_ms=4.0),
    ]  # fmt: skip
    report = evaluate(cases, predictions, "s", "dev")

    acc = report["category_accuracy"]
    assert (acc["all"]["hits"], acc["all"]["n"]) == (2, 4)
    assert (acc["auto_verified"]["hits"], acc["needs_review"]["hits"]) == (1, 1)
    assert report["errors"] == 1

    loc = report["localization"]
    assert loc["n"] == 1 and loc["mrr"] == 0.5 and loc["hit@3"]["hits"] == 1
    assert sum(loc["excluded"].values()) == 3

    grounding = report["evidence_grounding"]
    assert (grounding["items"], grounding["grounded"]) == (3, 2)
    assert grounding["cases_with_ungrounded_evidence"] == 1


def test_evaluate_refuses_missing_predictions(cases):
    with pytest.raises(ValueError, match="no prediction"):
        evaluate(cases, [], "s", "dev")


def test_run_system_reports_each_prediction_as_it_finishes(cases, monkeypatch):
    """Long local runs must survive interruption, so results are saved per case."""
    saved: list[str] = []
    monkeypatch.setitem(
        runner_mod.SYSTEMS,
        "stub",
        lambda trace_dir=None: runner_mod.SystemRun(
            "stub_v0", lambda view: diag("TEST_FAILURE", [])
        ),
    )
    run_system("stub", cases, on_prediction=lambda p: saved.append(p.case_id))
    assert saved == [c.case_id for c in cases]


def test_run_system_isolates_failures(cases, monkeypatch):
    def broken(view):
        assert not hasattr(view, "ground_truth")  # systems only ever get a CaseView
        raise RuntimeError("bad case")

    monkeypatch.setitem(
        runner_mod.SYSTEMS,
        "broken",
        lambda trace_dir=None: runner_mod.SystemRun("broken_v0", broken),
    )
    predictions = run_system("broken", cases)
    assert len(predictions) == 4
    assert all(p.diagnosis is None and "bad case" in p.error for p in predictions)
