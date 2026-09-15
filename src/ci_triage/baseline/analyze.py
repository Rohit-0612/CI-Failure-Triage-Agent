"""Rule baseline entry point: CaseView -> Diagnosis.

Every evidence excerpt is copied verbatim from the case input, so the evaluator's
grounding check can verify it. The root cause is a template over the classification and
the top suspect file: a hypothesis, never verified (verification.status = "not_run").
"""

from __future__ import annotations

from ci_triage.baseline.classify import Classification, classify
from ci_triage.baseline.localize import Suspect, localize
from ci_triage.diagnosis import MAX_EXCERPT_CHARS, Diagnosis, Evidence
from ci_triage.miner.schema import CaseView
from ci_triage.taxonomy import FailureCategory

C = FailureCategory
NAME = "rule_baseline_v1"

_WHAT_FAILED = {
    C.TEST_FAILURE: "Tests failed",
    C.SYNTAX_ERROR: "Python could not parse the code",
    C.TYPE_ERROR: "The type checker reported errors",
    C.IMPORT_ERROR: "A module could not be imported",
    C.DEPENDENCY_FAILURE: "Dependencies could not be resolved or installed",
    C.BUILD_FAILURE: "The build failed",
    C.LINT_FAILURE: "The linter reported violations",
    C.FORMAT_FAILURE: "The formatter check failed",
    C.CONFIGURATION_ERROR: "A tool configuration is invalid",
    C.ENVIRONMENT_FAILURE: "The runner environment failed",
    C.DOCKER_FAILURE: "The Docker step failed",
    C.CI_CONFIGURATION_FAILURE: "The workflow configuration is invalid",
    C.NETWORK_FAILURE: "A network operation failed",
    C.TIMEOUT: "The job timed out",
    C.COVERAGE_FAILURE: "Coverage fell below the required threshold",
    C.POLICY_CHECK_FAILURE: "A pull-request policy check failed (not a code problem)",
    C.FLAKY: "The failure looks non-deterministic",
    C.UNKNOWN: "The failure could not be classified",
}


def analyze(case: CaseView) -> Diagnosis:
    cls = classify(case)
    suspects = localize(case)
    evidence = _evidence(case, cls, suspects)
    confidence = cls.confidence if suspects else round(cls.confidence * 0.8, 3)
    return Diagnosis(
        failure_type=cls.category,
        root_cause=_root_cause(case, cls, suspects),
        confidence=confidence,
        evidence=evidence,
        affected_files=[s.path for s in suspects],
    )


def _root_cause(case: CaseView, cls: Classification, suspects: list[Suspect]) -> str:
    parts = [_WHAT_FAILED[cls.category]]
    sig = case.input.error_signature
    if sig:
        parts.append(f"({sig.exception_type}: {sig.message[:200]})")
    elif cls.evidence_line:
        parts.append(f"({cls.evidence_line[:200]})")
    if suspects and cls.category != C.POLICY_CHECK_FAILURE:
        top = suspects[0]
        parts.append(f"Most suspicious file: {top.path} [{', '.join(top.reasons)}].")
        window = case.input.breaking_window
        if window and top.path in window.files_changed and window.commits:
            first_line = window.commits[-1].message.splitlines()[0][:100]
            parts.append(f"It changed before the failure, e.g. in '{first_line}'.")
    return " ".join(parts)


def _evidence(case: CaseView, cls: Classification, suspects: list[Suspect]) -> list[Evidence]:
    stage = case.failure.failed_step_name or case.failure.job_name
    items: list[Evidence] = []
    if cls.evidence_line:
        items.append(
            Evidence(
                source="ci_log",
                location=f"failed step: {stage}",
                excerpt=cls.evidence_line[:MAX_EXCERPT_CHARS],
                explanation=f"matched baseline rule '{cls.rule}' -> {cls.category}",
            )
        )
    for line in case.input.log_excerpt.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED ") and stripped != cls.evidence_line:
            items.append(
                Evidence(
                    source="ci_log",
                    location=f"failed step: {stage}",
                    excerpt=stripped[:MAX_EXCERPT_CHARS],
                    explanation="first failing test reported by pytest",
                )
            )
            break
    if suspects:
        top = suspects[0].path
        hunk = (
            _diff_section(case.input.breaking_window.diff, top)
            if case.input.breaking_window
            else ""
        )
        if hunk:
            items.append(
                Evidence(
                    source="git_diff",
                    location=top,
                    excerpt=hunk[:MAX_EXCERPT_CHARS],
                    explanation="this file changed between the last passing state and the failure",
                )
            )
        log_line = next(
            (ln.strip() for ln in case.input.log_excerpt.splitlines() if top in ln), None
        )
        if log_line:
            items.append(
                Evidence(
                    source="ci_log",
                    location=f"failed step: {stage}",
                    excerpt=log_line[:MAX_EXCERPT_CHARS],
                    explanation=f"the log references {top}",
                )
            )
    return items


def _diff_section(diff: str, path: str, max_lines: int = 20) -> str:
    """The first lines of `path`'s section in a unified diff, verbatim."""
    lines = diff.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("diff --git ") and line.endswith(f" b/{path}"):
            section = []
            for follow in lines[i + 1 :]:
                if follow.startswith("diff --git "):
                    break
                section.append(follow)
                if len(section) >= max_lines:
                    break
            return "\n".join(section)
    return ""
