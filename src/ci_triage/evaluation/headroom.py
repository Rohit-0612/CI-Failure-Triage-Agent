"""Error analysis: why did fault localization miss, and what could fix it?

Localization hit@1 tells us *how often* a system is wrong. It does not tell us *why*,
and therefore not what to build next. This splits every miss by the information the
investigator actually held at the time:

    reasoning    the gold file's content was in the prompt; the system chose otherwise
    name_only    the gold path appeared (log line, diff header) but its content was never
                 sent - the evidence packer has to guess which files to include, and it
                 guessed wrong
    blind        the gold path did not appear at all, though the file exists at the
                 failed commit - it could only be found by searching the repository
    impossible   the file did not exist at the failed commit (the fix created it)

`name_only` and `blind` are the buckets a tool-using agent can address; `reasoning` is
not fixed by more data, and `impossible` is not fixed by anything.

Run before building retrieval or tools, and again afterwards to see which bucket moved.
"""

from __future__ import annotations

import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ci_triage.agent import evidence as ev
from ci_triage.agent.graph import known_paths
from ci_triage.evaluation.runner import Prediction, gold_fix_files
from ci_triage.miner.schema import CaseRecord, CaseView

BUCKETS = ("hit", "reasoning", "name_only", "blind", "impossible")

BUCKET_MEANING = {
    "hit": "hit@1 was already correct",
    "reasoning": "gold file's content was in the prompt; the model chose another file",
    "name_only": "gold path was visible as a name, its content was never sent",
    "blind": "gold path was absent from the evidence; the file exists in the repository",
    "impossible": "gold file did not exist at the failed commit",
}
# Buckets a tool-using or retrieval-based agent could plausibly recover.
REACHABLE = ("name_only", "blind")


@dataclass
class CaseAnalysis:
    case_id: str
    bucket: str
    gold_files: list[str]
    predicted: list[str]
    content_sent: list[str]


def packed_file_paths(pack: ev.EvidencePack) -> set[str]:
    """Paths whose *content* the packer actually emitted.

    One definition, shared with the agent (`evidence.packed_paths`): the agent asks the
    same question to decide what is worth fetching, and two parsers of the same section
    header would be two chances to answer it differently.
    """
    return ev.packed_paths(pack.text)


def visible_paths(view: CaseView, evidence_text: str, gold: set[str]) -> set[str]:
    """Gold paths the investigator could have named from what it was given."""
    structured = known_paths(view)
    normalized = evidence_text.replace("\\", "/")
    return {path for path in gold if path in structured or path in normalized}


def file_exists_at(repos_dir: Path, repo_full_name: str, sha: str, path: str) -> bool:
    """Did `path` exist at the failed commit? False if the clone is unavailable.

    Never lazily fetches (BUG-005): a missing object must fail, not start a download.
    """
    repo = repos_dir / repo_full_name.replace("/", "__")
    if not (repo / "HEAD").exists():
        return False
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{sha}:{path}"],
        capture_output=True,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        check=False,
    )
    return result.returncode == 0


def classify(
    case: CaseRecord, prediction: Prediction | None, repos_dir: Path
) -> CaseAnalysis | None:
    """Put one case in a bucket, or None if it does not count for localization."""
    gold, _ = gold_fix_files(case)
    if gold is None:
        return None

    view = case.visible()
    pack = ev.pack(view)
    with_content = packed_file_paths(pack)
    predicted = prediction.diagnosis.affected_files if prediction and prediction.diagnosis else []
    row = CaseAnalysis(
        case_id=case.case_id,
        bucket="hit",
        gold_files=sorted(gold),
        predicted=predicted[:3],
        content_sent=sorted(with_content),
    )
    if predicted and predicted[0] in gold:
        return row

    visible = visible_paths(view, pack.text, gold)
    if visible & with_content:
        row.bucket = "reasoning"
    elif visible:
        row.bucket = "name_only"
    else:
        sha = view.input.failed_commit.sha
        exists = any(file_exists_at(repos_dir, view.repo.full_name, sha, p) for p in gold)
        row.bucket = "blind" if exists else "impossible"
    return row


def analyse(
    cases: list[CaseRecord], predictions: list[Prediction], repos_dir: Path
) -> dict[str, Any]:
    by_id = {p.case_id: p for p in predictions}
    rows = [
        analysis
        for case in cases
        if (analysis := classify(case, by_id.get(case.case_id), repos_dir)) is not None
    ]
    counts = Counter(row.bucket for row in rows)
    total = len(rows)
    reachable = sum(counts[bucket] for bucket in REACHABLE)
    return {
        "cases_scored": total,
        "buckets": {bucket: counts[bucket] for bucket in BUCKETS},
        "share": {
            bucket: round(counts[bucket] / total, 3) if total else None for bucket in BUCKETS
        },
        "tools_reachable": reachable,
        "tools_reachable_share": round(reachable / total, 3) if total else None,
        "hit_at_1": round(counts["hit"] / total, 3) if total else None,
        "ceiling_if_reachable_recovered": round((counts["hit"] + reachable) / total, 3)
        if total
        else None,
        "per_case": [
            {
                "case_id": row.case_id,
                "bucket": row.bucket,
                "gold_files": row.gold_files,
                "predicted": row.predicted,
                "content_sent": row.content_sent,
            }
            for row in rows
        ],
    }


def format_report(result: dict[str, Any], system: str, split: str) -> str:
    total = result["cases_scored"]
    lines = [f"{system} on {split}: {total} localization-eligible cases", ""]
    for bucket in BUCKETS:
        count = result["buckets"][bucket]
        lines.append(f"  {count:3d} ({count / total:6.1%})  {bucket:<11} {BUCKET_MEANING[bucket]}")
    lines += [
        "",
        f"  hit@1                         {result['hit_at_1']:.1%}",
        f"  reachable by tools/retrieval  {result['tools_reachable_share']:.1%}"
        f"  ({' + '.join(REACHABLE)})",
        f"  ceiling if all recovered      {result['ceiling_if_reachable_recovered']:.1%}",
    ]
    return "\n".join(lines)
