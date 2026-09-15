"""Structured output of any failure investigator (rule baseline now, LLM agent later).

Free-form text is not accepted: every investigator must return a `Diagnosis`, which the
evaluation harness can score field by field.

The schema keeps observation and interpretation apart:
- `evidence`      observed facts, each quoting an excerpt that exists in the case input
- `root_cause`    the hypothesis built from that evidence
- `verification`  whether the hypothesis/fix was checked by running something
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ci_triage.taxonomy import FailureCategory

MAX_EXCERPT_CHARS = 1_000


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(_Model):
    source: Literal["ci_log", "git_diff", "source_file", "commit"]
    location: str  # e.g. "failed step log", "app.py", "commit 3f2a1b0"
    excerpt: str = Field(min_length=1, max_length=MAX_EXCERPT_CHARS)
    explanation: str


class ProposedFix(_Model):
    description: str
    patch: str | None = None  # unified diff; never applied without human approval


class Verification(_Model):
    status: Literal["verified", "failed", "not_run"] = "not_run"
    tests_run: list[str] = []
    result: str = ""


class Diagnosis(_Model):
    failure_type: FailureCategory
    root_cause: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence]
    affected_files: list[str]  # ranked: most suspicious first
    proposed_fix: ProposedFix | None = None
    verification: Verification = Verification()
