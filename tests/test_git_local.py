"""GitRepo against a real local "remote" repository with known history shapes."""

import subprocess
from pathlib import Path

import pytest

from ci_triage.git_local import GitRepo, validate_sha


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit(cwd: Path, files: dict[str, str], message: str) -> str:
    for name, content in files.items():
        (cwd / name).write_text(content)
    git(cwd, "add", "-A")
    git(cwd, "commit", "-q", "--allow-empty", "-m", message)
    return git(cwd, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    """
    base -- red -- fix              (fix descends from red: "ahead")
        \\-- amended                 (same parent as red: "amend")
        \\-- other -- rebased        (neither: "diverged")
    red -- retrigger                (empty commit: no file changes)
    """
    src = tmp_path_factory.mktemp("remote")
    git(src, "init", "-q", "-b", "main")
    git(src, "config", "uploadpack.allowAnySHA1InWant", "true")
    git(src, "config", "uploadpack.allowFilter", "true")
    shas = {}
    shas["base"] = commit(src, {"app.py": "def add(a, b):\n    return a + b\n"}, "base")
    shas["red"] = commit(src, {"app.py": "def add(a, b):\n    return a - b\n"}, "break add")
    shas["fix"] = commit(src, {"app.py": "def add(a, b):\n    return a + b\n"}, "fix add")
    git(src, "checkout", "-q", "-b", "amend", shas["base"])
    shas["amended"] = commit(src, {"app.py": "def add(a, b):\n    return b + a\n"}, "break add")
    git(src, "checkout", "-q", "-b", "other", shas["base"])
    shas["other"] = commit(src, {"README.md": "hi\n"}, "docs")
    shas["rebased"] = commit(src, {"app.py": "def add(a, b):\n    return a + b\n"}, "fix")
    git(src, "checkout", "-q", "-b", "retrigger", shas["red"])
    shas["retrigger"] = commit(src, {}, "retrigger CI")
    return src, shas


@pytest.fixture
def repo(tmp_path, history):
    src, shas = history
    repo = GitRepo.open_or_init(tmp_path, "o/r", remote_url=src.as_uri())
    assert repo.fetch(list(shas.values())) == set(shas.values())
    return repo


def test_relation_shapes(repo, history):
    _, s = history
    assert repo.relation(s["red"], s["red"]) == "identical"
    assert repo.relation(s["red"], s["fix"]) == "ahead"
    assert repo.relation(s["red"], s["amended"]) == "amend"
    assert repo.relation(s["red"], s["rebased"]) == "diverged"


def test_diff_between_amended_commits_is_only_the_real_change(repo, history):
    _, s = history
    assert repo.changed_files(s["red"], s["amended"]) == ["app.py"]
    diff, truncated = repo.diff(s["red"], s["amended"], max_chars=10_000)
    assert "-    return a - b" in diff and "+    return b + a" in diff
    assert not truncated


def test_empty_retrigger_commit_changes_no_files(repo, history):
    _, s = history
    assert repo.changed_files(s["red"], s["retrigger"]) == []


def test_commits_between_and_hunks(repo, history):
    _, s = history
    commits = repo.commits_between(s["red"], s["fix"])
    assert [c.sha for c in commits] == [s["fix"]]
    assert commits[0].parents == (s["red"],)
    assert commits[0].message == "fix add"
    assert repo.hunk_headers(s["red"], s["fix"]) == ["app.py @@ -2 +2 @@ def add(a, b):"]


def test_read_file_and_truncation(repo, history):
    _, s = history
    assert repo.read_file(s["red"], "app.py", 1000) == ("def add(a, b):\n    return a - b\n", False)
    assert repo.read_file(s["red"], "app.py", 5) == ("def a", True)
    assert repo.read_file(s["red"], "missing.py", 1000) is None


def test_fetch_skips_unreachable_sha(tmp_path, history):
    src, s = history
    repo = GitRepo.open_or_init(tmp_path, "o/r", remote_url=src.as_uri())
    ghost = "0" * 40
    assert repo.fetch([s["red"], ghost]) == {s["red"]}


def test_validate_sha_rejects_non_sha_input():
    with pytest.raises(ValueError):
        validate_sha("HEAD; rm -rf /")
    with pytest.raises(ValueError):
        validate_sha("abc123")
