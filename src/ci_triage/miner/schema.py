"""Dataset record schema (one JSONL line per case).

The record is split into:
- `input`:        what an investigator could know at failure time. Agents and
                  baselines may read only this part.
- `ground_truth`: information from the future (the fix). Only evaluation reads it.
- `labels`:       category / root cause, with how each label was obtained.

A validator refuses records where fix information leaks into `input`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ci_triage.git_local import Relation
from ci_triage.miner.fix_matcher import Confidence, FixStatus
from ci_triage.taxonomy import FailedStage, FailureCategory

SCHEMA_VERSION = "1.0"

Sha = str  # validated as 40-hex where it is produced (git_local.validate_sha)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepoInfo(_Model):
    full_name: str
    default_branch: str
    stars: int


class RunMeta(_Model):
    run_id: int
    run_attempt: int
    workflow_name: str
    workflow_path: str
    event: str
    head_repo: str
    head_branch: str
    html_url: str
    created_at: str
    streak_length: int = Field(ge=1)
    first_red_run_id: int


class FailedJob(_Model):
    job_id: int
    job_name: str
    failed_step_number: int | None
    failed_step_name: str | None
    failed_stage: FailedStage
    failure_timestamp: str | None
    other_failed_jobs: list[str]


class CommitRef(_Model):
    sha: Sha
    message: str
    author_date: str = ""
    parents: list[Sha] = []


class DiffWindow(_Model):
    base_sha: Sha
    head_sha: Sha
    commits: list[CommitRef]
    diff: str
    diff_truncated: bool
    files_changed: list[str]


class BreakingWindow(DiffWindow):
    # last_green: previous green run on the same streak key
    # pr_merge_base: PR branch point on the default branch (no earlier green run)
    # first_parent: the failed commit's own change (fallback)
    base_kind: Literal["last_green", "pr_merge_base", "first_parent"]


class ErrorSignature(_Model):
    exception_type: str
    message: str


class RelevantFile(_Model):
    path: str
    ref: Sha
    reasons: list[Literal["log_reference", "breaking_diff"]]
    content: str
    truncated: bool


class CaseInput(_Model):
    failed_commit: CommitRef
    log_excerpt: str
    log_excerpt_truncated: bool
    log_slice_method: Literal["group_marker", "timestamp", "tail"]
    error_lines: list[str]
    full_log_path: str
    error_signature: ErrorSignature | None
    failing_tests: list[str]
    log_referenced_files: list[str]
    breaking_window: BreakingWindow | None
    relevant_files: list[RelevantFile]


class FixWindow(DiffWindow):
    green_run_id: int
    green_run_url: str
    relation: Relation
    hunks: list[str]


class GroundTruth(_Model):
    fix_status: FixStatus
    fix_confidence: Confidence | None
    fix_confidence_reasons: list[str]
    fix_window: FixWindow
    candidate_fix_commit: Sha | None
    same_job_passed_in_green: bool | None


class SignalResult(_Model):
    category: FailureCategory | None
    rule: str


class Labels(_Model):
    category: FailureCategory
    log_signal: SignalResult
    fix_signal: SignalResult
    label_confidence: Literal["high", "medium", "low"]
    label_status: Literal["auto_verified", "needs_review", "human_verified"]
    # Why the auto-labeler decided this (labeling.combine rule, e.g. "signals_conflict").
    label_rule: str | None = None
    # The automatic category, kept after human review so audit precision is measurable.
    auto_category: FailureCategory | None = None
    audited: bool = False
    root_cause_text: str | None = None
    fix_text: str | None = None
    reviewer_notes: str | None = None


class CaseView(_Model):
    """What an investigator (baseline or agent) is given: failure-time information only.

    There is no `ground_truth` or `labels` field, so a system under evaluation cannot
    read the answer even by accident.
    """

    case_id: str
    repo: RepoInfo
    run: RunMeta
    failure: FailedJob
    input: CaseInput


class CaseRecord(_Model):
    case_id: str
    schema_version: str = SCHEMA_VERSION
    mined_at: str
    repo: RepoInfo
    run: RunMeta
    failure: FailedJob
    input: CaseInput
    ground_truth: GroundTruth
    labels: Labels

    def visible(self) -> CaseView:
        return CaseView(
            case_id=self.case_id,
            repo=self.repo,
            run=self.run,
            failure=self.failure,
            input=self.input,
        )

    @model_validator(mode="after")
    def _no_fix_leak_into_input(self) -> CaseRecord:
        """Future SHAs (the green commit and fix-window commits) must not appear in input."""
        window = self.ground_truth.fix_window
        future = {window.head_sha} | {c.sha for c in window.commits}
        future.discard(self.input.failed_commit.sha)  # amend/identical edge cases
        visible = self.input.model_dump_json()
        leaked = [sha[:10] for sha in future if sha in visible]
        if leaked:
            raise ValueError(f"fix information leaked into input: {leaked}")
        return self
