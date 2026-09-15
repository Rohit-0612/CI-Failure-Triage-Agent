from ci_triage.miner import logs
from ci_triage.taxonomy import FailedStage

RAW_LOG = "﻿" + "\n".join(
    [
        "2026-09-01T10:00:01.1000000Z ##[group]Run actions/setup-python@v5",
        "2026-09-01T10:00:04.9000000Z Installed versions",
        "2026-09-01T10:00:05.1000000Z ##[group]Run pytest -q",
        "2026-09-01T10:00:05.2000000Z pytest -q",
        "2026-09-01T10:00:08.0000000Z FAILED tests/test_app.py::test_add - assert -1 == 5",
        "2026-09-01T10:00:08.1000000Z E   assert -1 == 5",
        "continuation line without timestamp",
        "2026-09-01T10:00:09.5000000Z ##[error]Process completed with exit code 1.",
        "2026-09-01T10:00:10.0000000Z Post job cleanup.",
    ]
)


def test_parse_log_strips_bom_and_inherits_timestamps():
    lines = logs.parse_log(RAW_LOG)
    assert lines[0].text.startswith("##[group]Run actions/setup-python")
    assert lines[6].text == "continuation line without timestamp"
    assert lines[6].ts == lines[5].ts


def test_slice_uses_group_marker_matching_step_name():
    sliced = logs.slice_failed_step(
        logs.parse_log(RAW_LOG), "Run pytest -q", "2026-09-01T10:00:04Z", "2026-09-01T10:00:09Z"
    )
    assert sliced.method == "group_marker"
    assert sliced.lines[0] == "##[group]Run pytest -q"
    assert sliced.lines[-1].startswith("##[error]Process completed")
    assert "Installed versions" not in sliced.lines
    assert "Post job cleanup." not in sliced.lines


def test_slice_by_timestamp_when_step_has_custom_name():
    sliced = logs.slice_failed_step(
        logs.parse_log(RAW_LOG), "Tests", "2026-09-01T10:00:05Z", "2026-09-01T10:00:09Z"
    )
    assert sliced.method == "timestamp"
    assert sliced.lines[0] == "##[group]Run pytest -q"


def test_slice_falls_back_to_tail_without_step_times():
    sliced = logs.slice_failed_step(logs.parse_log(RAW_LOG), None, None, None, tail_lines=2)
    assert sliced.method == "tail"
    assert len(sliced.lines) == 2


def test_build_excerpt_keeps_head_and_tail():
    lines = [f"line {i}" for i in range(1000)]
    text, truncated = logs.build_excerpt(lines, max_chars=500, head_lines=5)
    assert truncated
    assert text.startswith("line 0\n")
    assert text.endswith("line 999")
    assert "lines omitted" in text
    assert len(text) <= 500


def test_failing_tests_and_error_lines():
    lines = [
        "FAILED tests/test_app.py::test_add[1-2] - assert 1 == 2",
        "ERROR tests/test_io.py - ModuleNotFoundError: No module named 'x'",
        "FAILED tests/test_app.py::test_add[1-2] - assert 1 == 2",
        "collected 10 items",
    ]
    assert logs.extract_failing_tests(lines) == [
        "tests/test_app.py::test_add[1-2]",
        "tests/test_io.py",
    ]
    errors = logs.extract_error_lines(lines, max_lines=10)
    assert len(errors) == 2  # duplicates removed, "collected" line ignored


def test_resolve_repo_paths():
    repo_files = {"src/flask/app.py", "tests/conftest.py", "app.py", "a/util.py", "b/util.py"}
    candidates = [
        "/home/runner/work/flask/flask/tests/conftest.py",  # workspace prefix
        ".tox/py/lib/python3.12/site-packages/flask/app.py",  # installed from repo
        ".tox/py/lib/python3.12/site-packages/werkzeug/datastructures.py",  # third party
        "/opt/hostedtoolcache/Python/3.12/lib/python3.12/unittest/case.py",  # system
        "util.py",  # ambiguous bare name
        "app.py",
    ]
    assert logs.resolve_repo_paths(candidates, repo_files) == [
        "tests/conftest.py",
        "src/flask/app.py",
        "app.py",
    ]


def test_extract_file_references():
    lines = [
        '  File "/home/runner/work/r/r/app.py", line 2, in add',
        "tests/test_app.py:4: AssertionError",
        "FAILED tests/test_x.py::test_y - boom",
        "src/pkg/mod.py:10:5: F401 unused import",
    ]
    assert logs.extract_file_references(lines) == [
        "/home/runner/work/r/r/app.py",
        "tests/test_app.py",
        "tests/test_x.py",
        "src/pkg/mod.py",
    ]


def test_error_signature_prefers_pytest_lines():
    lines = [
        "Traceback (most recent call last):",
        "ValueError: outer",
        "E   DeprecationWarning: The 'ImmutableDict' class is deprecated",
    ]
    assert logs.extract_error_signature(lines) == (
        "DeprecationWarning",
        "The 'ImmutableDict' class is deprecated",
    )
    assert logs.extract_error_signature(["E   assert -1 == 5"]) == (
        "AssertionError",
        "assert -1 == 5",
    )
    assert logs.extract_error_signature(["ValueError: bad"]) == ("ValueError", "bad")
    assert logs.extract_error_signature(["all good"]) is None


def test_infer_stage():
    assert logs.infer_stage("Run mypy src", "lint") == FailedStage.TYPECHECK
    assert logs.infer_stage("Run pytest -q", "tests (3.12)") == FailedStage.TEST
    assert logs.infer_stage("Run pre-commit/action@v3", "pre-commit") == FailedStage.LINT
    assert logs.infer_stage("Run", "docs") == FailedStage.BUILD
    assert logs.infer_stage(None, "misc") == FailedStage.OTHER
