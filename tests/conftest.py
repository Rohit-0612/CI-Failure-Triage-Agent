"""Shared test fixtures."""

import subprocess
from pathlib import Path

import pytest

RED, GREEN = "a" * 40, "b" * 40

# A marker that exists only in the commit that fixed the failure. Any tool output
# containing it has reached into the repository's future, which would put the answer
# into the investigator's input and invalidate every measurement in this project.
FIX_MARKER = "SECRET_FIX_TOKEN_THAT_ONLY_EXISTS_AFTER_THE_FIX"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_commit(cwd: Path, files: dict[str, str], message: str) -> str:
    for name, content in files.items():
        path = cwd / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    git(cwd, "add", "-A")
    git(cwd, "commit", "-q", "-m", message)
    return git(cwd, "rev-parse", "HEAD")


@pytest.fixture(scope="session")
def broken_project(tmp_path_factory):
    """A real repository: a commit that breaks a test, then the commit that fixes it.

    Everything a test needs comes back in the dict, including the commit helper:
    `tests/` is not a package, so test modules cannot import from conftest directly.
    """
    src = tmp_path_factory.mktemp("remote")
    git(src, "init", "-q", "-b", "main")
    git(src, "config", "uploadpack.allowAnySHA1InWant", "true")
    git(src, "config", "uploadpack.allowFilter", "true")
    red = git_commit(
        src,
        {
            "src/app.py": "def add(a, b):\n" + "    # padding\n" * 30 + "    return a - b\n",
            "tests/test_app.py": (
                "from src.app import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
            ),
            "docs/guide.md": "# Guide\n",
        },
        "break add",
    )
    green = git_commit(
        src,
        {"src/app.py": f"# {FIX_MARKER}\ndef add(a, b):\n    return a + b\n"},
        f"fix add {FIX_MARKER}",
    )
    return {
        "src": src,
        "red": red,
        "green": green,
        "fix_marker": FIX_MARKER,
        "commit": git_commit,
    }


@pytest.fixture
def toolbox(broken_project, tmp_path):
    """A Toolbox pinned to the broken commit, over a clone that also holds the fix."""
    from ci_triage.agent.tools import Toolbox
    from ci_triage.git_local import GitRepo

    repo = GitRepo.open_or_init(tmp_path, "o/r", remote_url=str(broken_project["src"]))
    repo.fetch([broken_project["red"], broken_project["green"]])
    return Toolbox(repo=repo, sha=broken_project["red"])


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
