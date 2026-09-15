import ast
from pathlib import Path

import pytest

from ci_triage.baseline import analyze as analyze_mod
from ci_triage.baseline.analyze import analyze
from ci_triage.baseline.classify import classify
from ci_triage.baseline.localize import localize
from ci_triage.evaluation.runner import grounding_corpus
from ci_triage.miner.schema import CaseRecord, CaseView
from ci_triage.taxonomy import FailureCategory as C

BASELINE_DIR = Path(analyze_mod.__file__).parent
DIFF = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a + b
+    return a - b
diff --git a/src/big.py b/src/big.py
--- a/src/big.py
+++ b/src/big.py
@@ -1 +1,3 @@
-x = 1
+x = 2
+y = 3
diff --git a/CHANGES.md b/CHANGES.md
--- a/CHANGES.md
+++ b/CHANGES.md
@@ -1 +1 @@
-old
+new
"""


@pytest.fixture
def view(record_dict):
    def make(**input_overrides) -> CaseView:
        return CaseRecord.model_validate(record_dict(**input_overrides)).visible()

    return make


def breaking(files):
    return {
        "base_sha": "c" * 40,
        "head_sha": "a" * 40,
        "commits": [{"sha": "a" * 40, "message": "refactor add"}],
        "diff": DIFF,
        "diff_truncated": False,
        "files_changed": files,
        "base_kind": "last_green",
    }


# ----------------------------------------------------------------------------- independence


def test_baseline_never_imports_labeling_or_ground_truth_code():
    """ADR-020: the baseline must not reuse the dataset labeler (circular evaluation)."""
    forbidden = (
        "ci_triage.miner.labeling",
        "ci_triage.miner.fix_matcher",
        "ci_triage.miner.pipeline",
        "ci_triage.evaluation",
    )
    for source in BASELINE_DIR.glob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                assert not name.startswith(forbidden), f"{source.name} imports {name}"


def test_case_view_has_no_answer_fields():
    assert set(CaseView.model_fields) == {"case_id", "repo", "run", "failure", "input"}


# ----------------------------------------------------------------------------- classify


def test_classify_returns_the_matching_log_line(view):
    line = "Please add '(#5247)' change line to CHANGES.md"
    result = classify(view(log_excerpt=f"setup\n{line}\n##[error]Process completed"))
    assert result.category == C.POLICY_CHECK_FAILURE
    assert result.evidence_line == line


def test_specific_cause_beats_generic_test_failure(view):
    excerpt = "E   ModuleNotFoundError: No module named 'foo'\nFAILED tests/test_a.py::test_x"
    assert classify(view(log_excerpt=excerpt)).category == C.IMPORT_ERROR


def test_step_name_is_only_a_low_confidence_fallback(view):
    result = classify(view(log_excerpt="##[error]Process completed with exit code 1."))
    assert (result.category, result.rule) == (C.TEST_FAILURE, "step_name_hint")
    assert result.confidence <= 0.3


# ----------------------------------------------------------------------------- localize


def test_file_in_log_and_diff_ranks_first_and_docs_are_ignored(view):
    case = view(
        log_referenced_files=["tests/test_app.py", "src/app.py"],
        breaking_window=breaking(["src/app.py", "src/big.py", "CHANGES.md"]),
        failing_tests=["tests/test_app.py::test_add"],
    )
    ranked = localize(case)
    paths = [s.path for s in ranked]
    assert paths[0] == "src/app.py"
    assert "CHANGES.md" not in paths
    assert set(paths) == {"src/app.py", "tests/test_app.py", "src/big.py"}
    assert "failing_test_file" in next(s for s in ranked if s.path == "tests/test_app.py").reasons


def test_bigger_change_ranks_higher_among_diff_only_files(view):
    paths = [s.path for s in localize(view(breaking_window=breaking(["src/app.py", "src/big.py"])))]
    assert paths == ["src/big.py", "src/app.py"]  # 3 changed lines vs 2


# ----------------------------------------------------------------------------- analyze


def test_analyze_returns_a_grounded_unverified_diagnosis(view):
    excerpt = "FAILED tests/test_app.py::test_add - assert -1 == 5\nE   assert -1 == 5"
    case = view(
        log_excerpt=excerpt,
        failing_tests=["tests/test_app.py::test_add"],
        log_referenced_files=["src/app.py"],
        breaking_window=breaking(["src/app.py"]),
    )
    diagnosis = analyze(case)
    assert diagnosis.failure_type == C.TEST_FAILURE
    assert diagnosis.affected_files[0] == "src/app.py"
    assert diagnosis.proposed_fix is None
    assert diagnosis.verification.status == "not_run"
    corpus = grounding_corpus(case)
    assert diagnosis.evidence and all(e.excerpt in corpus for e in diagnosis.evidence)
    assert {e.source for e in diagnosis.evidence} == {"ci_log", "git_diff"}
    assert "src/app.py" in diagnosis.root_cause
