"""Human review queue for dataset labels.

The queue holds:
- every `needs_review` case (the two auto-label signals conflicted or were weak), and
- a seeded random audit sample of `auto_verified` cases, so we can measure how often
  the automatic labels are right instead of assuming it.

The original automatic category is kept in `labels.auto_category`, so audit precision
= fraction of audited cases where the human kept the automatic category.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, get_args

from ci_triage.miner.schema import REVIEWED_STATUSES, CaseRecord
from ci_triage.miner.stats import load_cases
from ci_triage.taxonomy import FailureCategory

CATEGORIES = list(FailureCategory)
ReviewStatus = Literal["human_verified", "model_reviewed"]
# One definition of what counts as reviewed. Adding a status in the schema and forgetting
# it here would silently make it unwritable, so the two are checked against each other.
assert set(get_args(ReviewStatus)) == set(REVIEWED_STATUSES)


def build_queue(cases: list[CaseRecord], audit_size: int, seed: int = 0) -> list[tuple[str, str]]:
    """(case_id, "review" | "audit") pairs; the audit sample is reproducible via `seed`."""
    review = [c for c in cases if c.labels.label_status == "needs_review"]
    candidates = [
        c for c in cases if c.labels.label_status == "auto_verified" and not c.labels.audited
    ]
    done = sum(c.labels.audited for c in cases)
    k = max(0, min(audit_size - done, len(candidates)))
    audit = random.Random(seed).sample(candidates, k)
    return [(c.case_id, "review") for c in review] + [(c.case_id, "audit") for c in audit]


def apply_review(
    case: CaseRecord,
    category: FailureCategory,
    root_cause: str,
    fix_text: str,
    notes: str,
    audit: bool,
    status: ReviewStatus = "human_verified",
) -> CaseRecord:
    """Record a review. `status` says who did it, and it is not decoration: evaluation
    splits accuracy by label_status, so writing "human_verified" for a model's judgement
    would silently overstate every category number derived from it."""
    labels = case.labels.model_copy(
        update={
            "auto_category": case.labels.auto_category or case.labels.category,
            "category": category,
            "label_status": status,
            "label_confidence": "high",
            "root_cause_text": root_cause.strip() or None,
            "fix_text": fix_text.strip() or None,
            "reviewer_notes": notes.strip() or None,
            "audited": case.labels.audited or audit,
        }
    )
    return case.model_copy(update={"labels": labels})


def render_case(case: CaseRecord, position: str, why: str, *, blind: bool = False) -> str:
    """The evidence for one case, and (unless blind) what the auto-labeler thought.

    `blind` exists for the audit sample, whose whole purpose is to measure how often the
    automatic label is right. A reviewer who has already seen that label - or even just
    that the case is an *audit* case, which means the labeler was confident - will agree
    with it more often, and the resulting precision figure measures the anchoring rather
    than the labeler. So blind mode hides the label, its confidence, the rule that
    produced it, and whether this case is a review or an audit.
    """
    i, gt, lb = case.input, case.ground_truth, case.labels
    sig = i.error_signature
    fix_diff = "\n".join(gt.fix_window.diff.splitlines()[:60])
    heading = (
        f"[{position}] {case.case_id}"
        if blind
        else f"[{position}] {case.case_id}   ({why}; auto rule: {lb.label_rule})"
    )
    automatic = (
        []
        if blind
        else [
            "--- automatic label ---",
            f"category: {lb.category}  ({lb.label_confidence}, {lb.label_status})",
            f"log signal: {lb.log_signal.category} [{lb.log_signal.rule}]   "
            f"fix signal: {lb.fix_signal.category} [{lb.fix_signal.rule}]",
        ]
    )
    return "\n".join(
        [
            "=" * 88,
            heading,
            f"{case.repo.full_name} | {case.run.workflow_name} | job: {case.failure.job_name}",
            f"failed step: {case.failure.failed_step_name}  (stage: {case.failure.failed_stage})",
            f"run: {case.run.html_url}",
            f"error signature: {sig.exception_type + ': ' + sig.message if sig else '-'}",
            f"failing tests: {', '.join(i.failing_tests[:5]) or '-'}",
            "--- error lines (from the failed step) ---",
            *[f"  {line[:160]}" for line in i.error_lines[:15]],
            "--- HINDSIGHT: what fixed it ---",
            f"fix: {gt.fix_status}/{gt.fix_confidence}  relation: {gt.fix_window.relation}  "
            f"reasons: {', '.join(gt.fix_confidence_reasons)}",
            *[f"  commit: {c.message.splitlines()[0][:100]}" for c in gt.fix_window.commits[:5]],
            f"files: {', '.join(gt.fix_window.files_changed[:10])}",
            fix_diff,
            *automatic,
        ]
    )


def apply_reviews(
    path: Path, decisions: dict[str, dict[str, Any]], status: ReviewStatus
) -> dict[str, int]:
    """Apply reviews from a file instead of an interactive session.

    A decision may deliberately *decline* to label: give it `"category": null` and it
    keeps `needs_review` and only records the notes. Forcing a category onto evidence
    that does not support one would write a wrong gold label, and every system's score
    would then be measured against it - a wrong answer is more expensive here than an
    unanswered one.
    """
    cases = load_cases(path)
    by_id = {c.case_id: c for c in cases}
    unknown = sorted(set(decisions) - set(by_id))
    if unknown:
        raise ValueError(f"no such case(s) in {path}: {unknown[:5]}")

    applied = declined = 0
    for case_id, decision in decisions.items():
        case = by_id[case_id]
        category = decision.get("category")
        notes = str(decision.get("notes", "")).strip()
        if category is None:
            by_id[case_id] = case.model_copy(
                update={"labels": case.labels.model_copy(update={"reviewer_notes": notes or None})}
            )
            declined += 1
            continue
        by_id[case_id] = apply_review(
            case,
            FailureCategory(category),
            str(decision.get("root_cause", "")),
            str(decision.get("fix_text", "")),
            notes,
            audit=bool(decision.get("audit", False)) or case.labels.audited,
            status=status,
        )
        applied += 1

    save_cases(path, [by_id[c.case_id] for c in cases])
    return {"applied": applied, "declined": declined, "unchanged": len(cases) - len(decisions)}


def save_cases(path: Path, cases: list[CaseRecord]) -> None:
    """Atomic rewrite, so a crash mid-review never leaves a half-written dataset."""
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(c.model_dump_json() + "\n" for c in cases), encoding="utf-8")
    os.replace(tmp, path)


def run_review(
    path: Path,
    audit_size: int,
    *,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
    seed: int = 0,
) -> int:
    cases = load_cases(path)
    by_id = {c.case_id: c for c in cases}
    queue = build_queue(cases, audit_size, seed)
    menu = "  ".join(f"{n}={cat.value}" for n, cat in enumerate(CATEGORIES))
    out(f"{len(queue)} cases in queue. Categories: {menu}")
    reviewed = 0
    for n, (case_id, why) in enumerate(queue, start=1):
        case = by_id[case_id]
        out(render_case(case, f"{n}/{len(queue)}", why))
        category = _ask_category(case, input_fn, out)
        if category == "quit":
            break
        if category is None:
            continue
        root_cause = input_fn("root cause (one line): ")
        fix_text = input_fn("what the fix did (one line): ")
        notes = input_fn("notes (optional): ")
        by_id[case_id] = apply_review(case, category, root_cause, fix_text, notes, why == "audit")
        save_cases(path, [by_id[c.case_id] for c in cases])
        reviewed += 1
    out(f"reviewed {reviewed} case(s)")
    return reviewed


def _ask_category(
    case: CaseRecord, input_fn: Callable[[str], str], out: Callable[[str], None]
) -> FailureCategory | str | None:
    while True:
        answer = input_fn("category [Enter=keep, number, s=skip, q=quit]: ").strip().lower()
        if answer == "q":
            return "quit"
        if answer == "s":
            return None
        if answer == "":
            return case.labels.category
        if answer.isdigit() and int(answer) < len(CATEGORIES):
            return CATEGORIES[int(answer)]
        out(f"invalid choice: {answer!r}")
