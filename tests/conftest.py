"""Shared test fixtures."""

import pytest

RED, GREEN = "a" * 40, "b" * 40


def _record(**input_overrides) -> dict:
    """A minimal, valid CaseRecord as a plain dict (edit it, then validate)."""
    return {
        "case_id": "o__r__1",
        "mined_at": "2026-09-15T00:00:00+00:00",
        "repo": {"full_name": "o/r", "default_branch": "main", "stars": 1},
        "run": {
            "run_id": 1,
            "run_attempt": 1,
            "workflow_name": "CI",
            "workflow_path": ".github/workflows/ci.yml",
            "event": "push",
            "head_repo": "o/r",
            "head_branch": "main",
            "html_url": "",
            "created_at": "2026-09-01T00:00:00Z",
            "streak_length": 1,
            "first_red_run_id": 1,
        },
        "failure": {
            "job_id": 5,
            "job_name": "test",
            "failed_step_number": 4,
            "failed_step_name": "Run pytest",
            "failed_stage": "test",
            "failure_timestamp": None,
            "other_failed_jobs": [],
        },
        "input": {
            "failed_commit": {"sha": RED, "message": "break"},
            "log_excerpt": "FAILED t.py::test",
            "log_excerpt_truncated": False,
            "log_slice_method": "timestamp",
            "error_lines": [],
            "full_log_path": "data/raw/logs/o__r__1.log.gz",
            "error_signature": None,
            "failing_tests": [],
            "log_referenced_files": [],
            "breaking_window": None,
            "relevant_files": [],
            **input_overrides,
        },
        "ground_truth": {
            "fix_status": "matched",
            "fix_confidence": "high",
            "fix_confidence_reasons": [],
            "fix_window": {
                "base_sha": RED,
                "head_sha": GREEN,
                "commits": [{"sha": GREEN, "message": "fix"}],
                "diff": "",
                "diff_truncated": False,
                "files_changed": ["app.py"],
                "green_run_id": 2,
                "green_run_url": "",
                "relation": "ahead",
                "hunks": [],
            },
            "candidate_fix_commit": GREEN,
            "same_job_passed_in_green": True,
        },
        "labels": {
            "category": "TEST_FAILURE",
            "log_signal": {"category": "TEST_FAILURE", "rule": "test_failure"},
            "fix_signal": {"category": None, "rule": "fix_changes_source_code"},
            "label_confidence": "medium",
            "label_status": "auto_verified",
        },
    }


@pytest.fixture
def record_dict():
    return _record
