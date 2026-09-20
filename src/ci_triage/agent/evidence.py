"""Pack a case's failure-time evidence into a bounded, sanitized prompt section.

Two jobs:
1. **Budget.** A case carries 14-18k tokens of logs, diffs and files; a local 7B model
   with a 24k context cannot take all of it plus instructions plus output. Sections are
   filled in priority order and truncation is recorded, never silent.
2. **Containment.** Everything here is untrusted repository text. It goes inside one
   `<untrusted_evidence>` block, and any delimiter-like string in the text is defanged
   so the block cannot be closed early from inside.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ci_triage.miner.schema import CaseView

OPEN_TAG = "<untrusted_evidence>"
CLOSE_TAG = "</untrusted_evidence>"
# ~24k chars is roughly 7k tokens. Measured on a local 7B model: 24k and 12k chars both
# take ~90 s per call (generation-bound, not prefill-bound), so we keep the larger budget.
DEFAULT_BUDGET_CHARS = 24_000
# Used when the first attempt answers UNKNOWN and the graph retries with more evidence.
EXPANDED_BUDGET_CHARS = 40_000
# Used after a model timeout: a smaller prompt is the one lever we have on a machine
# that could not finish the call in time.
SHRUNK_BUDGET_CHARS = 12_000
MAX_FILE_CHARS = 6_000
# Two files, not four: with four, the log and the diff got squeezed to nothing (observed
# on real pydantic cases, where six of eight sections were truncated).
MAX_FILES = 2

_TAG_RE = re.compile(r"</?untrusted_evidence>", re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

FILE_SECTION = "file {path} (at the failed commit)"
_FILE_SECTION_RE = re.compile(r"^## file (?P<path>.+) \(at the failed commit\)$", re.MULTILINE)


def packed_paths(evidence_text: str) -> set[str]:
    """Paths whose *content* this evidence text actually carries.

    Read back from the text, not from the case: the packer keeps only `MAX_FILES` of
    the case's relevant files, so the case's own list overstates what the model saw.
    Getting this wrong once already inverted an analysis (see evaluation/headroom.py).
    """
    return {match.group("path") for match in _FILE_SECTION_RE.finditer(evidence_text)}


@dataclass
class EvidencePack:
    text: str
    used_chars: dict[str, int] = field(default_factory=dict)
    truncated: list[str] = field(default_factory=list)


def sanitize(text: str) -> str:
    """Defang delimiter strings and control characters in untrusted text."""
    return _CONTROL_RE.sub(" ", _TAG_RE.sub("[tag removed]", text))


def pack(case: CaseView, budget_chars: int = DEFAULT_BUDGET_CHARS) -> EvidencePack:
    inp = case.input
    pack_ = EvidencePack(text="")
    remaining = budget_chars
    parts: list[str] = []

    def add(name: str, body: str, share: float) -> None:
        nonlocal remaining
        if not body.strip():
            return
        cap = min(remaining, max(500, int(budget_chars * share)))
        if len(body) > cap:
            body = body[:cap] + f"\n[... {name} truncated ...]"
            pack_.truncated.append(name)
        parts.append(f"## {name}\n{sanitize(body)}")
        used = len(body)
        pack_.used_chars[name] = used
        remaining -= used

    signature = inp.error_signature
    facts = [
        f"repository: {case.repo.full_name}",
        f"workflow: {case.run.workflow_name} (event: {case.run.event})",
        f"failed job: {case.failure.job_name}",
        f"failed step: {case.failure.failed_step_name or 'unknown'}"
        f" (stage: {case.failure.failed_stage})",
        f"error signature: {signature.exception_type}: {signature.message}"
        if signature
        else "error signature: none extracted",
        f"failing tests: {', '.join(inp.failing_tests[:10]) or 'none reported'}",
        f"files mentioned in the log: {', '.join(inp.log_referenced_files[:10]) or 'none'}",
        f"failed commit: {inp.failed_commit.sha[:10]}"
        f" {inp.failed_commit.message.splitlines()[0] if inp.failed_commit.message else ''}",
    ]
    add("failure facts", "\n".join(facts), 0.08)
    add("error lines from the failed step", "\n".join(inp.error_lines[:60]), 0.20)
    add("failed step log (excerpt)", inp.log_excerpt, 0.28)

    window = inp.breaking_window
    if window:
        commits = "\n".join(
            f"- {c.sha[:10]} {c.message.splitlines()[0] if c.message else ''}"
            for c in window.commits[:10]
        )
        add(
            f"changes since the last passing run ({window.base_kind})",
            f"commits:\n{commits}\n\ndiff:\n{window.diff}",
            0.28,
        )

    files = [f for f in inp.relevant_files if f.content.strip()][:MAX_FILES]
    for file in files:
        body = file.content[:MAX_FILE_CHARS]
        add(FILE_SECTION.format(path=file.path), body, 0.16)

    pack_.text = f"{OPEN_TAG}\n" + "\n\n".join(parts) + f"\n{CLOSE_TAG}"
    return pack_


def brief(case: CaseView, shown_paths: set[str], budget_chars: int = 3_000) -> str:
    """A small summary for the gathering loop: what failed, and what can be fetched.

    Deciding which file to look at does not need the diff and the file bodies, and
    sending them is expensive twice over: the prompt grows with every tool result, and
    a local 7B model slows down and rambles as it grows. Measured on one real case,
    the three gathering turns took 122 s, 148 s and 171 s with the full pack attached.

    The listed paths matter more than the size, though. That same run spent all three
    lookups re-reading the two files it had already been given, while the file that
    actually broke - known to the case, dropped by the packer's two-file limit - was
    never asked for. The model was being asked to infer what it had not been shown;
    here it is simply told.
    """
    inp = case.input
    signature = inp.error_signature
    missing = sorted(_candidate_paths(case) - shown_paths)
    facts = [
        f"repository: {case.repo.full_name}",
        f"failed job: {case.failure.job_name} / step: {case.failure.failed_step_name or '?'}",
        f"error signature: {signature.exception_type}: {signature.message}"
        if signature
        else "error signature: none extracted",
        f"failing tests: {', '.join(inp.failing_tests[:10]) or 'none reported'}",
    ]
    sections = [
        "## what failed\n" + "\n".join(facts),
        "## error lines from the failed step\n" + "\n".join(inp.error_lines[:25]),
        "## files you have already been given in full (do not ask for these again)\n"
        + ("\n".join(f"- {path}" for path in sorted(shown_paths)) or "- none"),
        "## paths named in this failure that you have NOT seen the contents of\n"
        + ("\n".join(f"- {path}" for path in missing[:25]) or "- none"),
    ]
    body = sanitize("\n\n".join(section for section in sections if section.strip()))
    if len(body) > budget_chars:
        body = body[:budget_chars] + "\n[... brief truncated ...]"
    return f"{OPEN_TAG}\n{body}\n{CLOSE_TAG}"


def _candidate_paths(case: CaseView) -> set[str]:
    """Paths this case names at failure time. No hindsight: all of it is input."""
    inp = case.input
    paths = set(inp.log_referenced_files)
    paths.update(f.path for f in inp.relevant_files)
    if inp.breaking_window:
        paths.update(inp.breaking_window.files_changed)
    paths.update(test.split("::", 1)[0] for test in inp.failing_tests)
    return {path for path in paths if path.strip()}


def with_sections(evidence_text: str, sections: list[str]) -> str:
    """Append tool results *inside* the untrusted block.

    Putting them after the closing tag would quietly promote whatever a tool read out
    of the repository to the same standing as our own instructions - which is exactly
    the boundary this block exists to draw. The sections are already sanitized by the
    toolbox; re-running it here is cheap and keeps the invariant local.
    """
    if not sections:
        return evidence_text
    body = "\n\n".join(sanitize(section) for section in sections)
    if evidence_text.endswith(CLOSE_TAG):
        head = evidence_text[: -len(CLOSE_TAG)].rstrip("\n")
        return f"{head}\n\n{body}\n{CLOSE_TAG}"
    return f"{evidence_text}\n\n{body}"
