"""Pure logic: auto-label a case from two independent signals and cross-check them.

1. log_signal: what the failure *looks like* in the failed step's log (regex rules).
2. fix_signal: what the later fix *actually changed* (hindsight; never shown to agents).

If both name the same category the label is `high`/auto_verified. If the fix is
consistent with the log but points elsewhere or is broad, it is `medium`/auto_verified.
If they conflict, the case goes to the human review queue (`needs_review`).

Note for evaluation: labels that came from signal agreement favour any baseline that
uses similar log regexes. Baselines must not import this module, and category accuracy
should also be reported on human-reviewed cases only.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from ci_triage.taxonomy import FailedStage, FailureCategory, pick_primary

C = FailureCategory
ALL = frozenset(C)

# ----------------------------------------------------------------------------- log signal

_LOG_RULES: tuple[tuple[str, FailureCategory, re.Pattern[str]], ...] = tuple(
    (rule, category, re.compile(pattern))
    for rule, category, pattern in (
        (
            "timeout",
            C.TIMEOUT,
            r"exceeded the maximum execution time|\+{5,} Timeout \+{5,}|Timeout \(\d",
        ),
        (
            "invalid_workflow",
            C.CI_CONFIGURATION_FAILURE,
            r"Invalid workflow file|Unable to resolve action|Can't find 'action\.ya?ml'",
        ),
        (
            # PR-metadata gates (changelog entry, PR template checklist, labels). Not code.
            "policy_check",
            C.POLICY_CHECK_FAILURE,
            r"change line to CHANGES|changelog (entry|fragment)|news (entry|fragment)"
            r"|PR checklist|checkbox in the PR|in the pull request body|'ci: skip news'",
        ),
        (
            "docker",
            C.DOCKER_FAILURE,
            r"failed to solve:|docker: Error response from daemon"
            r"|Cannot connect to the Docker daemon",
        ),
        (
            "dependency_resolution",
            C.DEPENDENCY_FAILURE,
            r"ResolutionImpossible|No matching distribution found"
            r"|Could not find a version that satisfies|No solution found when resolving"
            r"|version solving failed|conflicting dependencies|lockfile .*needs to be updated",
        ),
        ("syntax_error", C.SYNTAX_ERROR, r"\b(SyntaxError|IndentationError|TabError): "),
        (
            "import_error",
            C.IMPORT_ERROR,
            r"\b(ModuleNotFoundError|ImportError): |ImportError while (loading conftest"
            r"|importing test module)|cannot import name ",
        ),
        (
            "network",
            C.NETWORK_FAILURE,
            r"Temporary failure in name resolution|Could not resolve host|ECONNRESET"
            r"|Connection reset by peer|Max retries exceeded with url|RemoteDisconnected"
            r"|\b50[234] (Service Unavailable|Bad Gateway|Gateway Time-?out)"
            r"|The operation was aborted due to timeout|Read timed out",
        ),
        (
            "environment",
            C.ENVIRONMENT_FAILURE,
            r"No space left on device|\bMemoryError\b|Segmentation fault|command not found"
            r"|Unable to locate executable file"
            r"|The runner has received a shutdown signal|lost communication with the server",
        ),
        (
            "tool_config",
            C.CONFIGURATION_ERROR,
            r"error: unrecognized arguments|INTERNALERROR|Unknown config (option|key)",
        ),
        (
            "formatter",
            C.FORMAT_FAILURE,
            r"would reformat|files? would be reformatted|Imports are incorrectly sorted"
            r"|(ruff[- ]format|black|isort|trailing[- ]whitespace|end[- ]of[- ]files?"
            r"|mixed[- ]line[- ]ending|prettier|yamlfmt|taplo|pyproject-fmt|blacken-docs)"
            r"\b.*\.{3,}\s*Failed",
        ),
        (
            "linter",
            C.LINT_FAILURE,
            # ruff summaries: "Found 1 error." / "Found 1 error (1 fixed, 0 remaining)."
            # (mypy's "Found 2 errors in 1 file" must NOT match; it is a type error.)
            r"^Found \d+ errors?(\.| \(\d+ fixed)|^\S+\.pyi?:\d+:\d+: [A-Z]{1,4}\d{2,4}\b"
            r"|Your code has been rated at"
            r"|(ruff|ruff-check|flake8|pylint|codespell|pyupgrade|typos|zizmor)\b.*\.{3,}\s*Failed",
        ),
        (
            "type_checker",
            C.TYPE_ERROR,
            r"^\S+\.pyi?:\d+(:\d+)?: error: .*\[[\w-]+\]\s*$|Found \d+ errors? in \d+ files?"
            r"|^\S+\.pyi?:\d+:\d+: (error|warning)\[[\w-]+\]"  # ty (Astral)
            r"|(mypy|pyright|pytype|ty check)\b.*\.{3,}\s*Failed|- error: .*\(report\w+\)",
        ),
        (
            "coverage_threshold",
            C.COVERAGE_FAILURE,
            r"Coverage failure: total of|Required test coverage of [\d.]+% not reached"
            r"|FAIL Required test coverage",
        ),
        (
            "build",
            C.BUILD_FAILURE,
            r"Failed building wheel|subprocess-exited-with-error|error: command '.*' failed"
            r"|Warning, treated as error|build finished with problems"
            r"|Aborted with \d+ warnings? in strict mode",
        ),
        (
            "test_failure",
            C.TEST_FAILURE,
            r"^FAILED |^=+ .*\b\d+ failed\b|^E\s+assert |\bAssertionError\b"
            r"|short test summary info",
        ),
    )
)

# Generic patterns, used only when no specific rule matched: they say *that* a tool
# failed but not *which kind* of failure, so they must never outrank a specific rule.
_FALLBACK_LOG_RULES: tuple[tuple[str, FailureCategory, re.Pattern[str]], ...] = (
    ("precommit_hook_failed", C.LINT_FAILURE, re.compile(r"^[\w .:/()-]+?\.{4,}\s*Failed$")),
)

_STAGE_FALLBACK = {
    FailedStage.TEST: C.TEST_FAILURE,
    FailedStage.LINT: C.LINT_FAILURE,
    FailedStage.FORMAT: C.FORMAT_FAILURE,
    FailedStage.TYPECHECK: C.TYPE_ERROR,
    FailedStage.BUILD: C.BUILD_FAILURE,
    FailedStage.INSTALL: C.DEPENDENCY_FAILURE,
}


@dataclass(frozen=True)
class Signal:
    category: FailureCategory | None
    rule: str


def log_signal(lines: list[str], failing_tests: list[str], stage: FailedStage) -> Signal:
    stripped = [line.strip() for line in lines]
    matched: dict[FailureCategory, str] = {}
    for line in stripped:
        for rule, category, pattern in _LOG_RULES:
            if category not in matched and pattern.search(line):
                matched[category] = rule
    if failing_tests:
        matched.setdefault(C.TEST_FAILURE, "pytest_failed_tests")
    if matched:
        primary = pick_primary(matched)
        return Signal(primary, matched[primary])
    for rule, category, pattern in _FALLBACK_LOG_RULES:
        if any(pattern.search(line) for line in stripped):
            return Signal(category, rule)
    if stage in _STAGE_FALLBACK:
        return Signal(_STAGE_FALLBACK[stage], f"stage_fallback:{stage}")
    return Signal(C.UNKNOWN, "no_rule_matched")


# ----------------------------------------------------------------------------- fix signal


@dataclass(frozen=True)
class PyFileChange:
    path: str
    old: str | None  # None: file added
    new: str | None  # None: file deleted


@dataclass(frozen=True)
class FixSignal:
    category: FailureCategory | None  # None: the fix implies no specific category
    compatible: frozenset[FailureCategory]  # log categories consistent with this fix
    overrides: bool  # root cause lies outside the code, so the fix category wins
    decisive: bool  # hindsight fact strong enough for `high` when consistent
    rule: str


# Covers both `requirements-dev.txt` and a `requirements/` directory (`requirements/tests.txt`).
_DEP_FILE_RE = re.compile(
    r"(^|/)(requirements[^/]*\.(txt|in)|requirements/[^/]+\.(txt|in)|constraints[^/]*\.txt"
    r"|uv\.lock|poetry\.lock|pdm\.lock|Pipfile(\.lock)?|environment\.ya?ml)$"
)
_DEP_MAYBE_FILES = {"pyproject.toml", "setup.py", "setup.cfg"}
_VERSION_SPEC_RE = re.compile(r"[A-Za-z0-9_.\-\[\]]+\s*(==|>=|<=|~=|!=|<|>)\s*\d")
_CONFIG_FILE_RE = re.compile(
    r"(^|/)(tox\.ini|setup\.cfg|pytest\.ini|mypy\.ini|\.flake8|\.?ruff\.toml|pyproject\.toml"
    r"|\.pre-commit-config\.yaml|noxfile\.py|\.coveragerc|\.?pylintrc|Makefile|codecov\.ya?ml)$"
)
_DOCKER_FILE_RE = re.compile(r"(^|/)(Dockerfile[^/]*|[^/]*\.dockerfile|docker-compose[^/]*)$")
_DOC_FILE_RE = re.compile(r"(^docs?/|\.(md|rst)$|(^|/)(CHANGES|CHANGELOG|HISTORY|NEWS)[^/]*$)")
_TEST_FILE_RE = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)(test_[^/]*|[^/]*_test)\.py$|conftest\.py$"
)
_TYPING_MODULES = {"typing", "typing_extensions", "collections.abc", "__future__"}

_FLAKY_COMPAT = frozenset(
    {C.TEST_FAILURE, C.NETWORK_FAILURE, C.TIMEOUT, C.ENVIRONMENT_FAILURE, C.DOCKER_FAILURE,
     C.DEPENDENCY_FAILURE, C.BUILD_FAILURE, C.UNKNOWN}
)  # fmt: skip
_CODE_COMPAT = frozenset(
    {C.TEST_FAILURE, C.IMPORT_ERROR, C.SYNTAX_ERROR, C.TYPE_ERROR, C.LINT_FAILURE,
     C.FORMAT_FAILURE, C.BUILD_FAILURE, C.CONFIGURATION_ERROR, C.COVERAGE_FAILURE, C.UNKNOWN}
)  # fmt: skip
# A policy check (changelog entry, PR checklist) is resolved by editing PR metadata, so
# no code change and no "same code passed later" may ever confirm or override it.
_NEVER_FROM_FIX = frozenset({C.FLAKY, C.POLICY_CHECK_FAILURE})


def fix_signal(
    fix_status: str, fix_files: list[str], fix_diff: str, py_changes: list[PyFileChange]
) -> FixSignal:
    if fix_status == "flaky_rerun":
        return FixSignal(C.FLAKY, _FLAKY_COMPAT, True, True, "same_code_passed_later")

    changed = _changed_lines_by_file(fix_diff)
    kinds = {_file_kind(f, changed.get(f, [])) for f in fix_files} - {"doc"}
    if not kinds:
        return FixSignal(None, frozenset(), False, False, "fix_touches_only_docs")
    if kinds == {"dep"} or (kinds == {"dep", "config"} and _has_dep_change(changed)):
        return FixSignal(
            C.DEPENDENCY_FAILURE,
            ALL - {C.SYNTAX_ERROR, C.FORMAT_FAILURE, C.CI_CONFIGURATION_FAILURE} - _NEVER_FROM_FIX,
            True,
            False,
            "fix_changes_only_dependencies",
        )
    if kinds == {"ci"}:
        return FixSignal(
            C.CI_CONFIGURATION_FAILURE,
            ALL
            - {C.SYNTAX_ERROR, C.FORMAT_FAILURE, C.LINT_FAILURE, C.TYPE_ERROR}
            - _NEVER_FROM_FIX,
            True,
            False,
            "fix_changes_only_ci_workflows",
        )
    if "docker" in kinds and kinds <= {"docker", "ci"}:
        return FixSignal(
            C.DOCKER_FAILURE,
            frozenset({C.DOCKER_FAILURE, C.BUILD_FAILURE, C.DEPENDENCY_FAILURE,
                       C.ENVIRONMENT_FAILURE, C.NETWORK_FAILURE, C.UNKNOWN}),
            True,
            False,
            "fix_changes_only_docker",
        )  # fmt: skip
    if kinds == {"config"}:
        return FixSignal(
            C.CONFIGURATION_ERROR,
            frozenset({C.CONFIGURATION_ERROR, C.LINT_FAILURE, C.TYPE_ERROR, C.FORMAT_FAILURE,
                       C.TEST_FAILURE, C.BUILD_FAILURE, C.DEPENDENCY_FAILURE, C.UNKNOWN}),
            False,
            False,
            "fix_changes_only_tool_config",
        )  # fmt: skip
    if kinds <= {"source", "test"}:
        py_kind = _python_change_kind(py_changes) if py_changes else "semantic"
        if py_kind == "format":
            return FixSignal(
                C.FORMAT_FAILURE, frozenset({C.FORMAT_FAILURE, C.LINT_FAILURE, C.UNKNOWN}),
                False, False, "fix_is_formatting_only",
            )  # fmt: skip
        if py_kind == "noqa":
            return FixSignal(
                C.LINT_FAILURE, frozenset({C.LINT_FAILURE, C.FORMAT_FAILURE, C.UNKNOWN}),
                False, False, "fix_adds_lint_suppressions_only",
            )  # fmt: skip
        if py_kind == "types":
            return FixSignal(
                C.TYPE_ERROR, frozenset({C.TYPE_ERROR, C.LINT_FAILURE, C.UNKNOWN}),
                False, False, "fix_changes_only_type_annotations",
            )  # fmt: skip
        if py_kind == "coverage":
            return FixSignal(
                C.COVERAGE_FAILURE, frozenset({C.COVERAGE_FAILURE, C.TEST_FAILURE, C.UNKNOWN}),
                False, False, "fix_adds_coverage_pragmas_only",
            )  # fmt: skip
        if kinds == {"test"}:
            return FixSignal(C.TEST_FAILURE, _CODE_COMPAT, False, False, "fix_changes_only_tests")
        return FixSignal(None, _CODE_COMPAT, False, False, "fix_changes_source_code")
    return FixSignal(None, ALL - _NEVER_FROM_FIX, False, False, "fix_mixed_change_kinds")


# ----------------------------------------------------------------------------- combine

LabelConfidence = Literal["high", "medium", "low"]
LabelStatus = Literal["auto_verified", "needs_review"]


@dataclass(frozen=True)
class LabelDecision:
    category: FailureCategory
    confidence: LabelConfidence
    status: LabelStatus
    rule: str


def is_weak_log_signal(log: Signal) -> bool:
    """A stage-name guess is not evidence from the log; no error pattern matched."""
    return log.rule.startswith("stage_fallback") or log.rule == "no_rule_matched"


def combine(log: Signal, fix: FixSignal) -> LabelDecision:
    log_cat = log.category or C.UNKNOWN
    weak = is_weak_log_signal(log)
    if fix.rule == "fix_mixed_change_kinds":
        # Mixed fixes are consistent with almost anything, so they verify nothing.
        return LabelDecision(log_cat, "low", "needs_review", "fix_too_mixed_to_verify")
    if fix.category is not None and log_cat == fix.category:
        confidence: LabelConfidence = "medium" if weak else "high"
        return LabelDecision(log_cat, confidence, "auto_verified", "signals_agree")
    if log_cat in fix.compatible:
        # Only fixes whose category says the cause was outside the code (dependency,
        # CI config, docker, flaky) may decide the label on their own.
        if fix.overrides:
            category = fix.category or log_cat
            confidence = "high" if fix.decisive else "medium"
            return LabelDecision(category, confidence, "auto_verified", "fix_refines_log")
        if weak:
            # Only a guess from the step name, and the fix does not name a category:
            # nothing independent confirms the label.
            return LabelDecision(log_cat, "low", "needs_review", "weak_log_signal")
        return LabelDecision(log_cat, "medium", "auto_verified", "fix_consistent_with_log")
    fallback = log_cat if log_cat != C.UNKNOWN else (fix.category or C.UNKNOWN)
    return LabelDecision(fallback, "low", "needs_review", "signals_conflict")


# ----------------------------------------------------------------------------- helpers


def is_doc_file(path: str) -> bool:
    return bool(_DOC_FILE_RE.search(path))


def _file_kind(path: str, changed_lines: list[str]) -> str:
    name = path.rsplit("/", 1)[-1]
    if path.startswith(".github/"):
        return "ci"
    if _DOC_FILE_RE.search(path):
        return "doc"
    if _DEP_FILE_RE.search(path):
        return "dep"
    if name in _DEP_MAYBE_FILES and any(_VERSION_SPEC_RE.search(ln) for ln in changed_lines):
        return "dep"
    if _DOCKER_FILE_RE.search(path):
        return "docker"
    if _CONFIG_FILE_RE.search(path):
        return "config"
    if _TEST_FILE_RE.search(path):
        return "test"
    return "source"


def _has_dep_change(changed: dict[str, list[str]]) -> bool:
    return any(_VERSION_SPEC_RE.search(ln) for lines in changed.values() for ln in lines)


def _changed_lines_by_file(diff: str) -> dict[str, list[str]]:
    changed: dict[str, list[str]] = {}
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("--- "):
            continue
        elif current and line[:1] in ("+", "-"):
            changed.setdefault(current, []).append(line[1:])
    return changed


# Weakest to strongest change; a set of per-file kinds reduces to its strongest member.
_PY_KIND_ORDER = ("format", "noqa", "coverage", "types", "semantic")


def _python_change_kind(changes: list[PyFileChange]) -> str:
    """format | noqa | coverage | types | semantic, over all changed Python files."""
    kinds = {_one_python_change(c) for c in changes}
    return max(kinds, key=_PY_KIND_ORDER.index)


def _one_python_change(change: PyFileChange) -> str:
    if change.old is None or change.new is None:
        return "semantic"
    try:
        old_tree, new_tree = ast.parse(change.old), ast.parse(change.new)
    except SyntaxError:
        return "semantic"
    if ast.dump(old_tree) == ast.dump(new_tree):
        added = _comments(change.new) - _comments(change.old)
        text = " ".join(added)
        if "type: ignore" in text or "pyright: ignore" in text:
            return "types"
        if "pragma: no cover" in text:
            return "coverage"
        if "noqa" in text or "pylint: disable" in text:
            return "noqa"
        return "format"
    if ast.dump(_strip_types(old_tree)) == ast.dump(_strip_types(new_tree)):
        return "types"
    return "semantic"


def _comments(source: str) -> Counter[str]:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return Counter(t.string for t in tokens if t.type == tokenize.COMMENT)
    except (tokenize.TokenError, SyntaxError):
        return Counter()


class _StripTypes(ast.NodeTransformer):
    """Remove annotations and typing-only imports so type-only edits compare equal."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.returns = None
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.annotation = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if node.value is None:
            return None
        return ast.Assign(targets=[node.target], value=node.value)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        return None if node.module in _TYPING_MODULES else node

    def visit_If(self, node: ast.If) -> ast.AST | None:
        test = node.test
        name = test.id if isinstance(test, ast.Name) else getattr(test, "attr", None)
        if name == "TYPE_CHECKING":
            return None
        self.generic_visit(node)
        return node


def _strip_types(tree: ast.Module) -> ast.Module:
    return _StripTypes().visit(tree)
