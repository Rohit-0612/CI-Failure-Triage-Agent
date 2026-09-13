"""Pure logic: turn a workflow's run history into red -> green transitions.

A "streak" is a sequence of consecutive failed runs of the same workflow, for the same
event, on the same branch of the same (possibly forked) repository. One streak becomes
at most one dataset case: its first red run is the failure we investigate, its last
red run is where the fix window starts, and the next green run is where it ends.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# Only these conclusions carry a pass/fail signal. cancelled/skipped/neutral/timed_out/
# startup_failure/action_required runs are ignored: they neither start nor end a streak.
RED = "failure"
GREEN = "success"

StreakKey = tuple[int, str, str, str]  # (workflow_id, event, head_repo, head_branch)


@dataclass(frozen=True)
class RunInfo:
    run_id: int
    workflow_id: int
    workflow_name: str
    workflow_path: str
    event: str
    head_repo: str
    head_branch: str
    head_sha: str
    conclusion: str | None
    created_at: str
    run_attempt: int
    html_url: str

    @classmethod
    def from_api(cls, run: dict[str, Any]) -> RunInfo:
        head_repo = (run.get("head_repository") or {}).get("full_name") or ""
        return cls(
            run_id=run["id"],
            workflow_id=run["workflow_id"],
            workflow_name=run.get("name") or "",
            workflow_path=run.get("path") or "",
            event=run["event"],
            head_repo=head_repo,
            head_branch=run.get("head_branch") or "",
            head_sha=run["head_sha"],
            conclusion=run.get("conclusion"),
            created_at=run["created_at"],
            run_attempt=run.get("run_attempt") or 1,
            html_url=run.get("html_url") or "",
        )

    @property
    def streak_key(self) -> StreakKey:
        return (self.workflow_id, self.event, self.head_repo, self.head_branch)


@dataclass
class Transition:
    key: StreakKey
    reds: list[RunInfo] = field(default_factory=list)
    prev_green: RunInfo | None = None
    next_green: RunInfo | None = None

    @property
    def first_red(self) -> RunInfo:
        return self.reds[0]

    @property
    def last_red(self) -> RunInfo:
        return self.reds[-1]


def find_transitions(runs: list[RunInfo]) -> list[Transition]:
    """Group runs by streak key and return every red streak, oldest first.

    `prev_green` is the last success before the streak (None if the streak starts at
    the beginning of the history we fetched); `next_green` is the first success after
    it (None if the branch is still red or we did not fetch far enough).
    """
    by_key: dict[StreakKey, list[RunInfo]] = defaultdict(list)
    for run in runs:
        if run.conclusion in (RED, GREEN):
            by_key[run.streak_key].append(run)

    transitions: list[Transition] = []
    for key, key_runs in by_key.items():
        # created_at is ISO-8601 UTC, so string order is time order; run_id breaks ties.
        key_runs.sort(key=lambda r: (r.created_at, r.run_id))
        last_green: RunInfo | None = None
        current: Transition | None = None
        for run in key_runs:
            if run.conclusion == RED:
                if current is None:
                    current = Transition(key=key, prev_green=last_green)
                current.reds.append(run)
            else:
                if current is not None:
                    current.next_green = run
                    transitions.append(current)
                    current = None
                last_green = run
        if current is not None:
            transitions.append(current)

    transitions.sort(key=lambda t: (t.first_red.created_at, t.first_red.run_id))
    return transitions
