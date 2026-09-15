"""Rank the files most likely responsible for a failure (fault localization).

Signals, all available at failure time:
- the file is referenced by the failed step's log (traceback frame, lint location)
- the file changed between the last passing state and the failure (breaking window)
- the file holds a failing test
- how much of the file changed (bigger edits are slightly more suspicious)

A file both referenced and changed scores both weights, which already puts it on top.
Dev-split ablation (42 cases, hit@1): log only 0.405, diff only 0.500, combined 0.571.
An extra "log and diff agree" bonus changed nothing on dev and was removed (ADR-021).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ci_triage.miner.schema import CaseView
from ci_triage.paths import is_doc_path

W_LOG = 2.0
W_LOG_ORDER_DECAY = 0.05  # earlier log references rank slightly higher
W_CHANGED = 1.5
W_FAILING_TEST = 0.5
W_CHANGE_SIZE_MAX = 0.5  # reached at 50 changed lines
MAX_SUSPECTS = 10


@dataclass
class Suspect:
    path: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def localize(case: CaseView) -> list[Suspect]:
    inp = case.input
    log_files = [f for f in inp.log_referenced_files if not is_doc_path(f)]
    window = inp.breaking_window
    changed = [f for f in (window.files_changed if window else []) if not is_doc_path(f)]
    changed_lines = _changed_line_counts(window.diff) if window else {}
    test_files = {t.split("::", 1)[0] for t in inp.failing_tests}

    suspects: dict[str, Suspect] = {}

    def get(path: str) -> Suspect:
        return suspects.setdefault(path, Suspect(path))

    for i, path in enumerate(log_files):
        s = get(path)
        s.score += W_LOG - min(i, 10) * W_LOG_ORDER_DECAY
        s.reasons.append("log_reference")
    for path in changed:
        s = get(path)
        s.score += W_CHANGED + min(changed_lines.get(path, 0), 50) / 50 * W_CHANGE_SIZE_MAX
        s.reasons.append("changed_before_failure")
    for path in test_files:
        if path in suspects:  # only boost files we already have evidence for
            suspects[path].score += W_FAILING_TEST
            suspects[path].reasons.append("failing_test_file")

    # sorted() is stable and dicts keep insertion order: ties go to first-seen (log first).
    ranked = sorted(suspects.values(), key=lambda s: -s.score)
    return ranked[:MAX_SUSPECTS]


def _changed_line_counts(diff: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else None
        # A tuple, not the string "+-": `"" in "+-"` is True and would count blank lines.
        elif current and line[:1] in ("+", "-") and not line.startswith("---"):
            counts[current] = counts.get(current, 0) + 1
    return counts
