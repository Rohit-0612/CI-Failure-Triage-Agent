"""Mining pipeline: repos -> failed runs -> red streaks -> evidence -> validated records.

All decisions are delegated to pure modules (runs, fix_matcher, logs, labeling); this
module only fetches data, wires it together, and writes three outputs:
- data/processed/cases.jsonl      accepted, schema-validated cases
- data/processed/rejected.jsonl   every examined streak that was not accepted, with why
- data/processed/mining_report.json   funnel counters for this run
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ci_triage.git_local import CommitInfo, GitError, GitRepo
from ci_triage.github_client import GitHubClient, GitHubError
from ci_triage.miner import labeling, logs
from ci_triage.miner.config import MinerSettings
from ci_triage.miner.fix_matcher import FixEvidence, score_fix
from ci_triage.miner.runs import RunInfo, StreakKey, Transition, find_transitions
from ci_triage.miner.schema import (
    BreakingWindow,
    CaseInput,
    CaseRecord,
    CommitRef,
    ErrorSignature,
    FailedJob,
    FixWindow,
    GroundTruth,
    Labels,
    RelevantFile,
    RepoInfo,
    RunMeta,
    SignalResult,
)

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = (".py", ".pyi", ".toml", ".cfg", ".ini", ".txt", ".yml", ".yaml", ".json")
_MAX_AST_FILES = 20
_MAX_AST_CHARS = 400_000
_MAX_WINDOW_COMMITS_STORED = 20


class CaseRejected(Exception):
    def __init__(self, reason: str, **details: Any):
        super().__init__(reason)
        self.reason = reason
        self.details = details


class Miner:
    def __init__(
        self,
        settings: MinerSettings,
        gh: GitHubClient,
        *,
        now: datetime | None = None,
        git_factory: Callable[[Path, str], GitRepo] = GitRepo.open_or_init,
    ):
        self.s = settings
        self.gh = gh
        self.now = now or datetime.now(UTC)
        self.git_factory = git_factory
        self.funnel: Counter[str] = Counter()
        self.cases_path = settings.processed_dir / "cases.jsonl"
        self.rejected_path = settings.processed_dir / "rejected.jsonl"
        self.report_path = settings.processed_dir / "mining_report.json"
        self.seen: set[str] = set()
        self.per_repo: Counter[str] = Counter()
        self.accepted = 0
        self.flaky = 0
        self._load_existing()

    # ------------------------------------------------------------------ driver

    def run(self) -> dict[str, Any]:
        for repo in self.s.repos:
            if self._target_reached():
                break
            try:
                self._mine_repo(repo)
            except (GitHubError, GitError) as exc:
                logger.error("repo %s aborted: %s", repo, exc)
                self.funnel["repo_errors"] += 1
        report = {
            "generated_at": self.now.isoformat(),
            "total_cases": self.accepted,
            "flaky_cases": self.flaky,
            "cases_per_repo": dict(self.per_repo),
            "funnel_this_run": dict(self.funnel),
            "api_requests_this_run": self.gh.request_count,
            "api_cache_hits_this_run": self.gh.cache_hits,
            "settings": self.s.model_dump(mode="json", exclude={"repos"}),
        }
        self.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _mine_repo(self, repo: str) -> None:
        if self.per_repo[repo] >= self.s.max_cases_per_repo:
            return
        meta = self.gh.get_json(f"/repos/{repo}")
        since = (self.now - timedelta(days=self.s.lookback_days)).date().isoformat()
        keys: dict[StreakKey, None] = {}
        for raw in self.gh.paginate(
            f"/repos/{repo}/actions/runs",
            "workflow_runs",
            {"status": "failure", "created": f">={since}"},
            max_pages=3,
        ):
            run = RunInfo.from_api(raw)
            self.funnel["failed_runs_seen"] += 1
            if run.event not in self.s.events or not run.head_branch:
                self.funnel["failed_runs_skipped_event"] += 1
                continue
            keys.setdefault(run.streak_key, None)
        logger.info("%s: %d streak keys from failed runs", repo, len(keys))

        git = self.git_factory(self.s.repos_dir, repo)
        for key in list(keys)[: self.s.max_keys_per_repo]:
            for transition in find_transitions(self._key_history(repo, key, since)):
                if self._repo_done(repo):
                    return
                self._process_transition(meta, git, transition)

    def _process_transition(self, meta: dict[str, Any], git: GitRepo, t: Transition) -> None:
        repo = meta["full_name"]
        case_id = f"{repo.replace('/', '__')}__{t.last_red.run_id}"
        self.funnel["streaks_examined"] += 1
        if case_id in self.seen:
            self.funnel["already_mined"] += 1
            return
        self.seen.add(case_id)
        try:
            record = self._build_case(meta, git, t, case_id)
        except CaseRejected as rej:
            self._reject(case_id, repo, t, rej.reason, rej.details)
            return
        except (GitHubError, GitError, ValidationError) as exc:
            self._reject(case_id, repo, t, f"error:{type(exc).__name__}", {"error": str(exc)[:300]})
            return
        self._accept(record)

    def _key_history(self, repo: str, key: StreakKey, since: str) -> list[RunInfo]:
        workflow_id, event, head_repo, branch = key
        runs = self.gh.paginate(
            f"/repos/{repo}/actions/workflows/{workflow_id}/runs",
            "workflow_runs",
            {"branch": branch, "event": event, "created": f">={since}"},
            max_pages=3,
        )
        # The branch filter matches the name only; forks can share names like "main".
        return [r for r in map(RunInfo.from_api, runs) if r.head_repo == head_repo]

    # ------------------------------------------------------------------ one case

    def _build_case(
        self, meta: dict[str, Any], git: GitRepo, t: Transition, case_id: str
    ) -> CaseRecord:
        repo = meta["full_name"]
        # The investigated failure is the LAST red run of the streak: its code is what
        # the fix window (last red -> first green) actually repaired.
        red, green = t.last_red, t.next_green
        if green is None:
            raise CaseRejected("no_green_found")

        jobs = list(self.gh.paginate(f"/repos/{repo}/actions/runs/{red.run_id}/jobs", "jobs"))
        failed_jobs = sorted(
            (j for j in jobs if j.get("conclusion") == "failure"),
            key=lambda j: (j.get("started_at") or "", j["id"]),
        )
        if not failed_jobs:
            raise CaseRejected("no_failed_job")
        job = failed_jobs[0]
        step = next((s for s in job.get("steps") or [] if s.get("conclusion") == "failure"), None)

        available = git.fetch([red.head_sha, green.head_sha])
        if {red.head_sha, green.head_sha} - available:
            raise CaseRejected("commit_unavailable")
        relation = git.relation(red.head_sha, green.head_sha)
        window = _fix_window_commits(git, relation, red.head_sha, green.head_sha)
        fix_files = (
            [] if relation == "identical" else git.changed_files(red.head_sha, green.head_sha)
        )
        green_jobs = list(
            self.gh.paginate(f"/repos/{repo}/actions/runs/{green.run_id}/jobs", "jobs")
        )
        same_job = _same_job_passed(job["name"], green_jobs)
        breaking = self._breaking_window(meta, git, t)
        breaking_shas = [c.sha for c in breaking.commits] if breaking else []

        def assess(log_files: list[str]):
            return score_fix(
                FixEvidence(
                    has_green=True,
                    relation=relation,
                    window_commits=window,
                    fix_files=fix_files,
                    log_referenced_files=log_files,
                    same_job_passed_in_green=same_job,
                    breaking_shas=breaking_shas,
                    max_window_commits=self.s.max_fix_window_commits,
                )
            )

        # Score before downloading logs: most rejections need no log at all.
        assessment = assess([])
        if assessment.status not in ("matched", "flaky_rerun"):
            raise CaseRejected(f"fix_{assessment.status}", reasons=assessment.reasons)
        if assessment.status == "flaky_rerun" and self.flaky >= self.s.max_flaky_cases:
            raise CaseRejected("flaky_cap_reached")

        raw_log = self.gh.get_job_log(repo, job["id"])
        if raw_log is None:
            raise CaseRejected("log_unavailable")
        step_slice = logs.slice_failed_step(
            logs.parse_log(raw_log),
            step.get("name") if step else None,
            step.get("started_at") if step else None,
            step.get("completed_at") if step else None,
        )
        lines = step_slice.lines
        excerpt, excerpt_truncated = logs.build_excerpt(lines, self.s.max_log_excerpt_chars)
        failing_tests = logs.extract_failing_tests(lines)
        repo_files = set(git.list_files(red.head_sha))
        log_files = logs.resolve_repo_paths(logs.extract_file_references(lines), repo_files)
        signature = logs.extract_error_signature(lines)
        assessment = assess(log_files)  # same status; adds the log-overlap reason

        fix_diff, fix_diff_truncated = git.diff(red.head_sha, green.head_sha, self.s.max_diff_chars)
        stage = logs.infer_stage(step.get("name") if step else None, job["name"])
        log_sig = labeling.log_signal(lines, failing_tests, stage)
        fix_sig = labeling.fix_signal(
            assessment.status,
            fix_files,
            fix_diff,
            _python_changes(git, red.head_sha, green.head_sha, fix_files),
        )
        decision = labeling.combine(log_sig, fix_sig)

        log_path = self.s.raw_logs_dir / f"{case_id}.log.gz"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(gzip.compress(raw_log.encode("utf-8"), mtime=0))

        failed_commit = git.commit_info(red.head_sha)
        return CaseRecord(
            case_id=case_id,
            mined_at=self.now.isoformat(),
            repo=RepoInfo(
                full_name=repo,
                default_branch=meta.get("default_branch", ""),
                stars=meta.get("stargazers_count", 0),
            ),
            run=RunMeta(
                run_id=red.run_id,
                run_attempt=red.run_attempt,
                workflow_name=red.workflow_name,
                workflow_path=red.workflow_path,
                event=red.event,
                head_repo=red.head_repo,
                head_branch=red.head_branch,
                html_url=red.html_url,
                created_at=red.created_at,
                streak_length=len(t.reds),
                first_red_run_id=t.first_red.run_id,
            ),
            failure=FailedJob(
                job_id=job["id"],
                job_name=job["name"],
                failed_step_number=step.get("number") if step else None,
                failed_step_name=step.get("name") if step else None,
                failed_stage=stage,
                failure_timestamp=job.get("completed_at"),
                other_failed_jobs=[j["name"] for j in failed_jobs[1:21]],
            ),
            input=CaseInput(
                failed_commit=_commit_ref(failed_commit),
                log_excerpt=excerpt,
                log_excerpt_truncated=excerpt_truncated,
                log_slice_method=step_slice.method,
                error_lines=logs.extract_error_lines(lines, self.s.max_error_lines),
                full_log_path=log_path.as_posix(),
                error_signature=ErrorSignature(exception_type=signature[0], message=signature[1])
                if signature
                else None,
                failing_tests=failing_tests,
                log_referenced_files=log_files,
                breaking_window=breaking,
                relevant_files=self._relevant_files(
                    git, red.head_sha, log_files, breaking.files_changed if breaking else []
                ),
            ),
            ground_truth=GroundTruth(
                fix_status=assessment.status,
                fix_confidence=assessment.confidence,
                fix_confidence_reasons=assessment.reasons,
                fix_window=FixWindow(
                    base_sha=red.head_sha,
                    head_sha=green.head_sha,
                    commits=[_commit_ref(c) for c in window[:_MAX_WINDOW_COMMITS_STORED]],
                    diff=fix_diff,
                    diff_truncated=fix_diff_truncated,
                    files_changed=fix_files,
                    green_run_id=green.run_id,
                    green_run_url=green.html_url,
                    relation=relation,
                    hunks=git.hunk_headers(red.head_sha, green.head_sha)
                    if relation != "identical"
                    else [],
                ),
                candidate_fix_commit=assessment.candidate_fix_commit,
                same_job_passed_in_green=same_job,
            ),
            labels=Labels(
                category=decision.category,
                log_signal=SignalResult(category=log_sig.category, rule=log_sig.rule),
                fix_signal=SignalResult(category=fix_sig.category, rule=fix_sig.rule),
                label_confidence=decision.confidence,
                label_status=decision.status,
                label_rule=decision.rule,
                auto_category=decision.category,
            ),
        )

    def _breaking_window(
        self, meta: dict[str, Any], git: GitRepo, t: Transition
    ) -> BreakingWindow | None:
        """What changed before the failure: agent-visible, so it must use only past SHAs."""
        red = t.last_red.head_sha
        base: str | None = None
        kind = "first_parent"
        if t.prev_green is not None:
            base, kind = t.prev_green.head_sha, "last_green"
        elif t.last_red.event == "pull_request":
            try:
                compare = self.gh.get_json(
                    f"/repos/{meta['full_name']}/compare/{meta['default_branch']}...{red}"
                )
                base, kind = compare["merge_base_commit"]["sha"], "pr_merge_base"
            except (GitHubError, KeyError):
                base = None
        if base is not None and base not in git.fetch([base]):
            base = None
        if base is None:
            parents = git.commit_info(red).parents
            if not parents:
                return None
            base, kind = parents[0], "first_parent"
            if base not in git.fetch([base]):
                return None
        if base == red:
            return None
        if git.is_ancestor(base, red):
            commits = git.commits_between(base, red)[-_MAX_WINDOW_COMMITS_STORED:]
        else:
            commits = [git.commit_info(red)]
        diff, truncated = git.diff(base, red, self.s.max_diff_chars)
        return BreakingWindow(
            base_sha=base,
            head_sha=red,
            commits=[_commit_ref(c) for c in commits],
            diff=diff,
            diff_truncated=truncated,
            files_changed=git.changed_files(base, red),
            base_kind=kind,
        )

    def _relevant_files(
        self, git: GitRepo, sha: str, log_files: list[str], breaking_files: list[str]
    ) -> list[RelevantFile]:
        reasons: dict[str, list[str]] = {}
        for path in log_files:
            reasons.setdefault(path, []).append("log_reference")
        for path in breaking_files:
            if path.endswith(_TEXT_SUFFIXES):
                reasons.setdefault(path, []).append("breaking_diff")
        files: list[RelevantFile] = []
        for path, why in reasons.items():
            if len(files) >= self.s.max_relevant_files:
                break
            content = git.read_file(sha, path, self.s.max_file_chars)
            if content is None:
                continue  # deleted at this commit, or binary
            text, truncated = content
            files.append(
                RelevantFile(path=path, ref=sha, reasons=why, content=text, truncated=truncated)
            )
        return files

    # ------------------------------------------------------------------ bookkeeping

    def _accept(self, record: CaseRecord) -> None:
        self.cases_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cases_path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        self.accepted += 1
        self.per_repo[record.repo.full_name] += 1
        if record.ground_truth.fix_status == "flaky_rerun":
            self.flaky += 1
        self.funnel["accepted"] += 1
        logger.info(
            "ACCEPT %s fix=%s/%s label=%s (%s, %s)",
            record.case_id,
            record.ground_truth.fix_status,
            record.ground_truth.fix_confidence,
            record.labels.category,
            record.labels.label_confidence,
            record.labels.label_status,
        )

    def _reject(
        self, case_id: str, repo: str, t: Transition, reason: str, details: dict[str, Any]
    ) -> None:
        self.rejected_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "case_id": case_id,
            "repo": repo,
            "run_id": t.last_red.run_id,
            "reason": reason,
            "details": details,
            "mined_at": self.now.isoformat(),
        }
        with self.rejected_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self.funnel[f"rejected:{reason}"] += 1
        logger.info("REJECT %s %s %s", case_id, reason, details or "")

    def _load_existing(self) -> None:
        """Resume support: never re-mine a case that is already accepted or rejected."""
        if self.cases_path.exists():
            for line in self.cases_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.seen.add(row["case_id"])
                self.per_repo[row["repo"]["full_name"]] += 1
                self.accepted += 1
                if row["ground_truth"]["fix_status"] == "flaky_rerun":
                    self.flaky += 1
        if self.rejected_path.exists():
            for line in self.rejected_path.read_text(encoding="utf-8").splitlines():
                self.seen.add(json.loads(line)["case_id"])

    def _target_reached(self) -> bool:
        return self.accepted >= self.s.target_cases

    def _repo_done(self, repo: str) -> bool:
        return self._target_reached() or self.per_repo[repo] >= self.s.max_cases_per_repo


# ---------------------------------------------------------------------- helpers


def _fix_window_commits(git: GitRepo, relation: str, red: str, green: str) -> list[CommitInfo]:
    if relation == "ahead":
        return git.commits_between(red, green)
    if relation == "amend":
        return [git.commit_info(green)]
    return []


def _same_job_passed(name: str, green_jobs: list[dict[str, Any]]) -> bool | None:
    matching = [j for j in green_jobs if j.get("name") == name]
    if not matching:
        return None
    return any(j.get("conclusion") == "success" for j in matching)


def _python_changes(
    git: GitRepo, red: str, green: str, fix_files: list[str]
) -> list[labeling.PyFileChange]:
    """Old/new source of changed .py files, for AST-level fix classification.

    Returns [] (meaning: treat as a semantic change) when the fix is not Python-only
    or too large to analyse.
    """
    code = [f for f in fix_files if not labeling.is_doc_file(f)]
    if not code or len(code) > _MAX_AST_FILES or not all(f.endswith(".py") for f in code):
        return []
    changes = []
    for path in code:
        old = git.read_file(red, path, _MAX_AST_CHARS)
        new = git.read_file(green, path, _MAX_AST_CHARS)
        if (old and old[1]) or (new and new[1]):
            return []  # truncated: cannot compare ASTs reliably
        changes.append(
            labeling.PyFileChange(path, old[0] if old else None, new[0] if new else None)
        )
    return changes


def _commit_ref(commit: CommitInfo) -> CommitRef:
    return CommitRef(
        sha=commit.sha,
        message=commit.message[:2000],
        author_date=commit.author_date,
        parents=list(commit.parents),
    )
