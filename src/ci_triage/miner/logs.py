"""Pure log processing for GitHub Actions job logs.

Every raw log line looks like `2026-09-08T16:41:14.8184792Z <text>`, and the jobs API
gives each step's started_at/completed_at (second precision). That lets us cut out the
failed step's lines by time, then tighten the start using the `##[group]Run ...` marker.

Log text is untrusted data. It is only matched against regexes here, never executed
or interpreted as instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from ci_triage.taxonomy import FailedStage

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?Z ?(.*)$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

_ERROR_LINE_RE = re.compile(
    r"##\[error\]"
    r"|^E\s{2,}"
    r"|^(?:FAILED|ERROR)\b"
    r"|Traceback \(most recent call last\)"
    r"|\b[A-Z]\w*(?:Error|Exception)\b:"
    r"|\berror:"
    r"|would reformat"
    r"|^Found \d+ errors?"
    r"|\.{3,}\s*Failed\b"
)
_FAILING_TEST_RE = re.compile(r"^(?:FAILED|ERROR) (\S+?\.py(?:::\S+)?)(?:\s|$)")
_TRACEBACK_FILE_RE = re.compile(r'File "([^"]+\.pyi?)", line \d+')
_PATH_LINE_RE = re.compile(r"(?<![\w/.\\-])((?:[\w.-]+[/\\])*[\w.-]+\.pyi?)(?::\d+|::)")
_WORKSPACE_RE = re.compile(r"^(?:/home/runner/work|/Users/runner/work|[A-Za-z]:/a)/[^/]+/[^/]+/")
_NON_REPO_DIRS = {".tox", ".nox", ".venv", "venv", "env", "__pypackages__", "node_modules"}
_SITE_DIRS = ("site-packages/", "dist-packages/")
_EXCEPTION_RE = re.compile(
    r"^(?:E\s+)?((?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception|Warning|Exit|Interrupt))"
    r"(?::\s*(.*))?$"
)
_PYTEST_ASSERT_RE = re.compile(r"^E\s+(assert .*)$")

SliceMethod = Literal["group_marker", "timestamp", "tail"]


@dataclass(frozen=True)
class LogLine:
    ts: datetime | None
    text: str


@dataclass(frozen=True)
class StepSlice:
    lines: list[str]
    method: SliceMethod


def parse_log(raw: str) -> list[LogLine]:
    """Split a raw log into (timestamp, text). Lines without a timestamp inherit one.

    ANSI color codes are removed: tools like pre-commit and pytest print colored
    output in CI, and `\\x1b[41mFailed\\x1b[m` would otherwise defeat every pattern.
    """
    lines: list[LogLine] = []
    last_ts: datetime | None = None
    for line in raw.lstrip("﻿").splitlines():
        match = _TS_RE.match(line)
        if match:
            last_ts = datetime.fromisoformat(match.group(1) + "+00:00")
            lines.append(LogLine(last_ts, _ANSI_RE.sub("", match.group(2))))
        else:
            lines.append(LogLine(last_ts, _ANSI_RE.sub("", line)))
    return lines


def slice_failed_step(
    lines: list[LogLine],
    step_name: str | None,
    started_at: str | None,
    completed_at: str | None,
    tail_lines: int = 400,
) -> StepSlice:
    """Return the failed step's lines, falling back to the log tail if we cannot tell."""
    if started_at and completed_at:
        start = _parse_api_time(started_at)
        # completed_at is truncated to the second, so include the whole last second.
        end = _parse_api_time(completed_at) + timedelta(seconds=1)
        idx = [i for i, ln in enumerate(lines) if ln.ts is not None and start <= ln.ts < end]
        if idx:
            first, last = idx[0], idx[-1]
            marker = f"##[group]{step_name}" if step_name else None
            for i in range(first, last + 1):
                if marker and lines[i].text.startswith(marker):
                    return StepSlice(_trim_end(_texts(lines[i : last + 1])), "group_marker")
            # Steps share boundary seconds; skip the previous step's tail if possible.
            for i in range(first, last + 1):
                if lines[i].text.startswith("##[group]Run ") and lines[i].ts == lines[first].ts:
                    return StepSlice(_trim_end(_texts(lines[i : last + 1])), "timestamp")
                if lines[i].ts != lines[first].ts:
                    break
            return StepSlice(_trim_end(_texts(lines[first : last + 1])), "timestamp")
    return StepSlice(_texts(lines[-tail_lines:]), "tail")


def _trim_end(texts: list[str]) -> list[str]:
    """Cut post-job steps that share the failed step's last second.

    A failing `run` step ends with `##[error]Process completed with exit code N.`;
    post-job steps start with `Post job cleanup.`. Nested `##[group]Run` lines are NOT
    boundaries: composite actions (e.g. pre-commit/action) print their own.
    """
    for j, text in enumerate(texts):
        if text.startswith("##[error]Process completed with exit code"):
            return texts[: j + 1]
        if j > 0 and text.startswith("Post job cleanup."):
            return texts[:j]
    return texts


def build_excerpt(lines: list[str], max_chars: int, head_lines: int = 30) -> tuple[str, bool]:
    """Head (the command) + tail (the failure summary) when the step output is too long."""
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text, False
    head = "\n".join(lines[:head_lines])
    budget = max_chars - len(head) - 64
    tail: list[str] = []
    used = 0
    for line in reversed(lines[head_lines:]):
        if used + len(line) + 1 > budget:
            break
        tail.append(line)
        used += len(line) + 1
    omitted = len(lines) - head_lines - len(tail)
    return f"{head}\n... [{omitted} lines omitted] ...\n" + "\n".join(reversed(tail)), True


def extract_error_lines(lines: list[str], max_lines: int) -> list[str]:
    seen: set[str] = set()
    found: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped not in seen and _ERROR_LINE_RE.search(stripped):
            seen.add(stripped)
            found.append(stripped[:500])
            if len(found) >= max_lines:
                break
    return found


def extract_failing_tests(lines: list[str]) -> list[str]:
    tests = [m.group(1) for line in lines if (m := _FAILING_TEST_RE.match(line.strip()))]
    return list(dict.fromkeys(tests))


def extract_file_references(lines: list[str]) -> list[str]:
    """Candidate file paths from tracebacks, `path.py:12` and pytest node ids (raw form)."""
    refs: list[str] = []
    for line in lines:
        refs.extend(_TRACEBACK_FILE_RE.findall(line))
        refs.extend(_PATH_LINE_RE.findall(line))
    return list(dict.fromkeys(refs))


def resolve_repo_paths(candidates: list[str], repo_files: set[str]) -> list[str]:
    """Map raw log paths to files that exist in the repository at the failed commit.

    Handles the runner workspace prefix and packages installed from the repo into
    site-packages (e.g. tox installs `src/flask` as `.../site-packages/flask`).
    Third-party and system paths do not resolve and are dropped.
    """
    resolved: list[str] = []
    for raw in candidates:
        path = _WORKSPACE_RE.sub("", raw.replace("\\", "/")).removeprefix("./")
        hit: str | None = None
        if path in repo_files:
            hit = path
        else:
            for site in _SITE_DIRS:
                if site in path:
                    hit = _unique_suffix_match(path.split(site, 1)[1], repo_files)
                    break
            else:
                parts = path.split("/")
                if not path.startswith("/") and not _NON_REPO_DIRS & set(parts):
                    hit = _unique_suffix_match(path, repo_files)
        if hit and hit not in resolved:
            resolved.append(hit)
    return resolved


def extract_error_signature(lines: list[str]) -> tuple[str, str] | None:
    """(exception type, message) of the root error; pytest `E` lines are preferred."""
    first_any: tuple[str, str] | None = None
    for line in lines:
        stripped = line.strip()
        if assert_match := _PYTEST_ASSERT_RE.match(stripped):
            return "AssertionError", assert_match.group(1)[:300]
        match = _EXCEPTION_RE.match(stripped)
        if not match:
            continue
        signature = (match.group(1), (match.group(2) or "")[:300])
        if stripped.startswith("E "):
            return signature
        first_any = first_any or signature
    return first_any


_STAGE_KEYWORDS: tuple[tuple[FailedStage, re.Pattern[str]], ...] = (
    (FailedStage.TYPECHECK, re.compile(r"\b(mypy|pyright|pytype|type[- ]?check(ing)?)\b")),
    (FailedStage.FORMAT, re.compile(r"\b(ruff format|black|isort|format(ting)?)\b")),
    (FailedStage.LINT, re.compile(r"\b(ruff|flake8|pylint|lint(ing)?|pre-commit|codespell)\b")),
    (FailedStage.BUILD, re.compile(r"\b(sphinx|docs?|mkdocs|build|wheel|sdist|twine)\b")),
    (FailedStage.TEST, re.compile(r"\b(pytest|tests?|tox|nox|coverage|unittest)\b")),
    (FailedStage.INSTALL, re.compile(r"\b(install|pip|uv sync|poetry|dependencies)\b")),
    (FailedStage.SETUP, re.compile(r"\b(checkout|set up|setup-python|setup-uv|cache)\b")),
)


def infer_stage(step_name: str | None, job_name: str) -> FailedStage:
    """Stage from the step name first, then the job name (e.g. job 'lint', step 'Run')."""
    for text in (step_name or "", job_name):
        lowered = text.lower()
        for stage, pattern in _STAGE_KEYWORDS:
            if pattern.search(lowered):
                return stage
    return FailedStage.OTHER


def _unique_suffix_match(path: str, repo_files: set[str]) -> str | None:
    if path in repo_files:
        return path
    if path.count("/") < 1:
        return None  # a bare filename like "app.py" is too ambiguous to map
    matches = [f for f in repo_files if f.endswith("/" + path)]
    return matches[0] if len(matches) == 1 else None


def _parse_api_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _texts(lines: list[LogLine]) -> list[str]:
    return [ln.text for ln in lines]
