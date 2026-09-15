from ci_triage.miner.labeling import (
    FixSignal,
    PyFileChange,
    Signal,
    combine,
    fix_signal,
    log_signal,
)
from ci_triage.taxonomy import FailedStage
from ci_triage.taxonomy import FailureCategory as C

# ----------------------------------------------------------------------------- log signal


def test_import_error_beats_generic_test_failure():
    lines = [
        "ImportError while loading conftest '/home/runner/work/r/r/tests/conftest.py'.",
        "FAILED tests/test_a.py::test_x",
    ]
    sig = log_signal(lines, ["tests/test_a.py::test_x"], FailedStage.TEST)
    assert sig == Signal(C.IMPORT_ERROR, "import_error")


def test_log_rules_for_tools():
    assert log_signal(["Found 3 errors."], [], FailedStage.LINT).category == C.LINT_FAILURE
    mypy = "src/a.py:10: error: Incompatible types in assignment  [assignment]"
    assert log_signal([mypy], [], FailedStage.TYPECHECK).category == C.TYPE_ERROR
    assert log_signal(["would reformat src/a.py"], [], FailedStage.FORMAT).category == (
        C.FORMAT_FAILURE
    )
    resolution = "ERROR: ResolutionImpossible: for help visit ..."
    assert log_signal([resolution], [], FailedStage.INSTALL).category == C.DEPENDENCY_FAILURE


def test_stage_fallback_and_unknown():
    assert log_signal(["exit 1"], [], FailedStage.LINT) == Signal(
        C.LINT_FAILURE, "stage_fallback:lint"
    )
    assert log_signal(["exit 1"], [], FailedStage.OTHER) == Signal(C.UNKNOWN, "no_rule_matched")


# ----------------------------------------------------------------------------- fix signal


def test_flaky_rerun_is_decisive():
    sig = fix_signal("flaky_rerun", [], "", [])
    assert sig.category == C.FLAKY and sig.decisive


def test_dependency_only_fixes():
    for path in ("requirements/tests.txt", "requirements-dev.txt", "uv.lock"):
        assert fix_signal("matched", [path], "", []).category == C.DEPENDENCY_FAILURE
    diff = "+++ b/pyproject.toml\n-  'werkzeug>=3.0'\n+  'werkzeug>=3.0,<3.2'\n"
    assert fix_signal("matched", ["pyproject.toml"], diff, []).category == C.DEPENDENCY_FAILURE


def test_pyproject_tool_config_is_configuration_not_dependency():
    diff = "+++ b/pyproject.toml\n+ignore = ['E501']\n"
    assert fix_signal("matched", ["pyproject.toml"], diff, []).category == C.CONFIGURATION_ERROR


def test_ci_only_and_docs_only():
    assert fix_signal("matched", [".github/workflows/ci.yml"], "", []).category == (
        C.CI_CONFIGURATION_FAILURE
    )
    docs = fix_signal("matched", ["docs/index.rst", "CHANGES.rst"], "", [])
    assert docs.category is None and not docs.compatible


def test_formatting_only_change_is_detected_by_ast():
    change = PyFileChange("a.py", "x = {'a':1}\n", 'x = {"a": 1}\n')
    assert fix_signal("matched", ["a.py"], "", [change]).rule == "fix_is_formatting_only"


def test_noqa_and_type_only_changes():
    noqa = PyFileChange("a.py", "import os\n", "import os  # noqa: F401\n")
    assert fix_signal("matched", ["a.py"], "", [noqa]).category == C.LINT_FAILURE
    typed = PyFileChange(
        "a.py",
        "def f(x):\n    return x\n",
        "from typing import Any\n\ndef f(x: Any) -> Any:\n    return x\n",
    )
    assert fix_signal("matched", ["a.py"], "", [typed]).category == C.TYPE_ERROR


def test_semantic_source_and_test_only_changes():
    real = PyFileChange(
        "app.py", "def add(a, b):\n    return a - b\n", "def add(a, b):\n    return a + b\n"
    )
    source = fix_signal("matched", ["app.py"], "", [real])
    assert source.category is None and source.rule == "fix_changes_source_code"
    tests_only = fix_signal("matched", ["tests/test_app.py"], "", [])
    assert tests_only.category == C.TEST_FAILURE


def test_mixed_change_kinds():
    sig = fix_signal("matched", ["app.py", "requirements.txt", ".github/workflows/ci.yml"], "", [])
    assert sig.rule == "fix_mixed_change_kinds"


# ----------------------------------------------------------------------------- combine


def fix(category, compatible, overrides=False, decisive=False, rule="r"):
    return FixSignal(category, frozenset(compatible), overrides, decisive, rule)


def test_agreement_is_high_confidence():
    d = combine(Signal(C.FORMAT_FAILURE, "formatter"), fix(C.FORMAT_FAILURE, {C.FORMAT_FAILURE}))
    assert (d.category, d.confidence, d.status) == (C.FORMAT_FAILURE, "high", "auto_verified")


def test_dependency_fix_refines_test_failure():
    d = combine(
        Signal(C.TEST_FAILURE, "test_failure"),
        fix(C.DEPENDENCY_FAILURE, {C.TEST_FAILURE}, overrides=True),
    )
    assert (d.category, d.confidence, d.rule) == (C.DEPENDENCY_FAILURE, "medium", "fix_refines_log")


def test_flaky_rerun_gives_high_flaky_label():
    d = combine(
        Signal(C.NETWORK_FAILURE, "network"),
        fix(C.FLAKY, {C.NETWORK_FAILURE}, overrides=True, decisive=True),
    )
    assert (d.category, d.confidence) == (C.FLAKY, "high")


def test_source_fix_is_consistent_with_test_failure():
    d = combine(Signal(C.TEST_FAILURE, "test_failure"), fix(None, {C.TEST_FAILURE}))
    assert (d.category, d.confidence, d.status) == (C.TEST_FAILURE, "medium", "auto_verified")


def test_stage_guess_is_not_verified_by_a_broad_fix():
    d = combine(Signal(C.TEST_FAILURE, "stage_fallback:test"), fix(None, {C.TEST_FAILURE}))
    assert (d.confidence, d.status, d.rule) == ("low", "needs_review", "weak_log_signal")


def test_stage_guess_matching_a_specific_fix_is_only_medium():
    d = combine(
        Signal(C.TEST_FAILURE, "stage_fallback:test"), fix(C.TEST_FAILURE, {C.TEST_FAILURE})
    )
    assert (d.confidence, d.status) == ("medium", "auto_verified")


def test_stage_guess_with_decisive_flaky_evidence_is_high():
    d = combine(
        Signal(C.TEST_FAILURE, "stage_fallback:test"),
        fix(C.FLAKY, {C.TEST_FAILURE}, overrides=True, decisive=True),
    )
    assert (d.category, d.confidence) == (C.FLAKY, "high")


def test_conflict_goes_to_review():
    d = combine(Signal(C.FORMAT_FAILURE, "formatter"), fix(C.FLAKY, {C.TEST_FAILURE}))
    assert (d.category, d.status) == (C.FORMAT_FAILURE, "needs_review")


def test_mixed_fix_goes_to_review():
    d = combine(
        Signal(C.TEST_FAILURE, "test_failure"),
        fix(None, set(C), rule="fix_mixed_change_kinds"),
    )
    assert (d.confidence, d.status) == ("low", "needs_review")
