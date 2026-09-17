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
MAX_FILE_CHARS = 6_000
# Two files, not four: with four, the log and the diff got squeezed to nothing (observed
# on real pydantic cases, where six of eight sections were truncated).
MAX_FILES = 2

_TAG_RE = re.compile(r"</?untrusted_evidence>", re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


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
        add(f"file {file.path} (at the failed commit)", body, 0.16)

    pack_.text = f"{OPEN_TAG}\n" + "\n\n".join(parts) + f"\n{CLOSE_TAG}"
    return pack_
