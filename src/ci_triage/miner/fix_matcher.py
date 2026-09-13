"""Pure logic: decide how much we trust that a red -> green transition contains a fix.

A later green run is NOT proof of a fix. It may be a rerun (flaky), an unrelated
change, or the outside world recovering (a dependency release, a network outage).
So this module never says "ground truth"; it returns a status, a confidence level
and the list of reasons that produced them, so every label can be audited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ci_triage.git_local import CommitInfo, Relation

Confidence = Literal["high", "medium", "low"]
FixStatus = Literal["matched", "flaky_rerun", "ambiguous", "no_green_found"]

_LEVELS: tuple[Confidence, ...] = ("high", "medium", "low")
# Not ".txt": requirements.txt / constraints.txt are dependency files, not docs.
_DOC_SUFFIXES = (".md", ".rst")
_REVERT_RE = re.compile(r"This reverts commit ([0-9a-f]{7,40})")


@dataclass(frozen=True)
class FixEvidence:
    has_green: bool
    relation: Relation | None = None
    # Commits between last red and first green. For an amend this is [green commit].
    window_commits: list[CommitInfo] = field(default_factory=list)
    fix_files: list[str] = field(default_factory=list)
    log_referenced_files: list[str] = field(default_factory=list)
    # True: failed job passed in the green run; False: it ran but did not succeed;
    # None: no job with that name ran (path filters, renamed matrix entry...).
    same_job_passed_in_green: bool | None = None
    breaking_shas: list[str] = field(default_factory=list)
    max_window_commits: int = 5


@dataclass(frozen=True)
class FixAssessment:
    status: FixStatus
    confidence: Confidence | None
    candidate_fix_commit: str | None
    reasons: list[str]


def score_fix(ev: FixEvidence) -> FixAssessment:
    if not ev.has_green:
        return FixAssessment("no_green_found", None, None, ["no_success_run_after_streak"])
    if ev.relation == "identical":
        return FixAssessment("flaky_rerun", None, None, ["green_run_tested_same_commit"])
    if ev.relation not in ("ahead", "amend"):
        return FixAssessment("ambiguous", "low", None, [f"relation_{ev.relation}"])

    n = len(ev.window_commits)
    reasons = [f"relation_{ev.relation}", f"window_commits_{n}"]
    if n == 0:
        return FixAssessment("ambiguous", "low", None, [*reasons, "empty_fix_window"])
    if not ev.fix_files:
        # Code is byte-identical to the red commit (e.g. an empty "retrigger CI" commit).
        return FixAssessment("flaky_rerun", None, None, [*reasons, "fix_window_changes_no_files"])
    if n > ev.max_window_commits:
        return FixAssessment("ambiguous", "low", None, [*reasons, "fix_window_too_large"])

    level = 0 if n == 1 else 1 if n <= 3 else 2

    if ev.same_job_passed_in_green is None:
        level += 1
        reasons.append("failed_job_absent_in_green_run")
    elif ev.same_job_passed_in_green is False:
        level += 1
        reasons.append("failed_job_not_successful_in_green_run")

    if all(_is_doc_file(f) for f in ev.fix_files):
        level = 2
        reasons.append("fix_touches_only_docs")

    if set(ev.fix_files) & set(ev.log_referenced_files):
        reasons.append("fix_overlaps_log_referenced_files")

    candidate = ev.window_commits[0].sha if n == 1 else None
    revert = _find_revert_of(ev.window_commits, ev.breaking_shas)
    if revert is not None:
        reasons.append("revert_of_breaking_commit")
        candidate = revert.sha
        level = min(level, 1)

    confidence = _LEVELS[min(level, 2)]
    status: FixStatus = "matched" if confidence in ("high", "medium") else "ambiguous"
    return FixAssessment(status, confidence, candidate, reasons)


def _is_doc_file(path: str) -> bool:
    return path.startswith("docs/") or path.lower().endswith(_DOC_SUFFIXES)


def _find_revert_of(commits: list[CommitInfo], breaking_shas: list[str]) -> CommitInfo | None:
    for commit in commits:
        for match in _REVERT_RE.finditer(commit.message):
            prefix = match.group(1)
            if any(sha.startswith(prefix) for sha in breaking_shas):
                return commit
    return None
