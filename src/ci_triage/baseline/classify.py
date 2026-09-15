"""Failure category from the failed step's log, by ordered first-match rules.

Unlike the dataset labeler, this sees only failure-time information and returns the log
line that triggered the rule, so every classification comes with quotable evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ci_triage.miner.schema import CaseView
from ci_triage.taxonomy import FailureCategory

C = FailureCategory


@dataclass(frozen=True)
class Classification:
    category: FailureCategory
    confidence: float
    rule: str
    evidence_line: str | None  # verbatim line from the case input, if any


# (rule id, category, confidence, pattern). Order is priority: the first rule with a
# matching line wins. Specific causes come before generic symptoms such as FAILED tests.
_RULES: tuple[tuple[str, FailureCategory, float, re.Pattern[str]], ...] = tuple(
    (rule, category, confidence, re.compile(pattern))
    for rule, category, confidence, pattern in (
        (
            "pr_policy",
            C.POLICY_CHECK_FAILURE,
            0.85,
            r"(add|missing).{0,40}(CHANGES|changelog|news)|PR checklist"
            r"|pull request (body|description)|skip news",
        ),
        ("job_timeout", C.TIMEOUT, 0.85, r"exceeded the maximum execution time|\+{5,} Timeout"),
        (
            "workflow_invalid",
            C.CI_CONFIGURATION_FAILURE,
            0.85,
            r"Invalid workflow file|Unable to resolve action|Can't find 'action\.ya?ml'",
        ),
        (
            "resolver",
            C.DEPENDENCY_FAILURE,
            0.8,
            r"ResolutionImpossible|No matching distribution found"
            r"|Could not find a version that satisfies|No solution found when resolving"
            r"|version solving failed|lockfile .*needs to be updated",
        ),
        ("syntax", C.SYNTAX_ERROR, 0.85, r"\b(SyntaxError|IndentationError|TabError)\b"),
        (
            "import",
            C.IMPORT_ERROR,
            0.8,
            r"\b(ModuleNotFoundError|ImportError)\b|cannot import name",
        ),
        (
            "network",
            C.NETWORK_FAILURE,
            0.6,
            r"Temporary failure in name resolution|Could not resolve host|Max retries exceeded"
            r"|Connection (reset|refused|timed out)|operation was aborted due to timeout"
            r"|Read timed out",
        ),
        (
            "runner_env",
            C.ENVIRONMENT_FAILURE,
            0.7,
            r"No space left on device|command not found|Unable to locate executable"
            r"|shutdown signal|Segmentation fault|\bMemoryError\b",
        ),
        (
            "coverage",
            C.COVERAGE_FAILURE,
            0.85,
            r"Coverage failure|fail[-_]under|Required test coverage",
        ),
        (
            "formatter",
            C.FORMAT_FAILURE,
            0.85,
            r"would reformat|would be reformatted"
            r"|(ruff format|black|isort|prettier|yamlfmt|end of files|trailing whitespace)"
            r".*\.{3,}\s*Failed",
        ),
        (
            "type_checker",
            C.TYPE_ERROR,
            0.85,
            r"\.pyi?:\d+(:\d+)?: error:|\.pyi?:\d+:\d+: (error|warning)\[[\w-]+\]"
            r"|Found \d+ errors? in \d+ files?|(mypy|pyright|ty check).*\.{3,}\s*Failed",
        ),
        (
            "linter",
            C.LINT_FAILURE,
            0.8,
            r"^Found \d+ errors?(\.| \()|\.pyi?:\d+:\d+: [A-Z]+\d+ "
            r"|(ruff|flake8|pylint).*\.{3,}\s*Failed",
        ),
        (
            "build",
            C.BUILD_FAILURE,
            0.7,
            r"Failed building wheel|subprocess-exited-with-error|treated as error"
            r"|build finished with problems",
        ),
        (
            "pytest",
            C.TEST_FAILURE,
            0.7,
            r"^FAILED |^ERROR \S+\.py|\bAssertionError\b|^E\s+assert |\b\d+ failed\b",
        ),
        ("hook_failed", C.LINT_FAILURE, 0.5, r"\.{4,}\s*Failed$"),
    )
)

# Last resort: what the step is called. A guess, so confidence is low.
_STEP_HINTS: tuple[tuple[FailureCategory, re.Pattern[str]], ...] = (
    (C.TYPE_ERROR, re.compile(r"\b(mypy|pyright|type[- ]?check)", re.I)),
    (C.FORMAT_FAILURE, re.compile(r"\b(format|black|isort)\b", re.I)),
    (C.LINT_FAILURE, re.compile(r"\b(lint|ruff|flake8|pre-commit)\b", re.I)),
    (C.BUILD_FAILURE, re.compile(r"\b(docs|build|wheel|sphinx)\b", re.I)),
    (C.TEST_FAILURE, re.compile(r"\b(test|tests|pytest|tox|nox)\b", re.I)),
)


def classify(case: CaseView) -> Classification:
    lines = [line.strip() for line in case.input.error_lines]
    lines += [line.strip() for line in case.input.log_excerpt.splitlines()]
    lines = [line for line in lines if line]
    for rule, category, confidence, pattern in _RULES:
        for line in lines:
            if pattern.search(line):
                return Classification(category, confidence, rule, line)
    if case.input.failing_tests:
        return Classification(C.TEST_FAILURE, 0.6, "failing_tests", None)
    step = f"{case.failure.failed_step_name or ''} {case.failure.job_name}"
    for category, pattern in _STEP_HINTS:
        if pattern.search(step):
            return Classification(category, 0.3, "step_name_hint", None)
    return Classification(C.UNKNOWN, 0.1, "no_rule", None)
