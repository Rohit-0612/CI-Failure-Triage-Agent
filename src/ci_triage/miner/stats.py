"""Dataset statistics computed from the JSONL files (never typed in by hand)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ci_triage.miner.schema import CaseRecord


def load_cases(path: Path) -> list[CaseRecord]:
    if not path.exists():
        return []
    return [
        CaseRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_rejections(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compute_stats(cases: list[CaseRecord], rejections: list[dict[str, Any]]) -> dict[str, Any]:
    def count(values) -> dict[str, int]:
        return dict(Counter(values).most_common())

    matched = [c for c in cases if c.ground_truth.fix_status == "matched"]
    audited = [c for c in cases if c.labels.audited and c.labels.auto_category is not None]
    audit_agree = sum(c.labels.category == c.labels.auto_category for c in audited)
    reviewed = [c for c in cases if c.labels.label_status == "human_verified"]

    return {
        "cases": len(cases),
        "repositories": len({c.repo.full_name for c in cases}),
        "streaks_examined": len(cases) + len(rejections),
        "cases_per_repo": count(c.repo.full_name for c in cases),
        "events": count(c.run.event for c in cases),
        "fix_status": count(c.ground_truth.fix_status for c in cases),
        "matched_fix_confidence": count(c.ground_truth.fix_confidence for c in matched),
        "matched_with_single_candidate_commit": sum(
            c.ground_truth.candidate_fix_commit is not None for c in matched
        ),
        "fix_relation": count(c.ground_truth.fix_window.relation for c in cases),
        "failed_stage": count(c.failure.failed_stage for c in cases),
        "log_slice_method": count(c.input.log_slice_method for c in cases),
        "category_all": count(c.labels.category for c in cases),
        "category_by_label_status": {
            status: count(c.labels.category for c in cases if c.labels.label_status == status)
            for status in ("auto_verified", "needs_review", "human_verified")
        },
        "label_status": count(c.labels.label_status for c in cases),
        "label_confidence": count(c.labels.label_confidence for c in cases),
        "log_rule": count(c.labels.log_signal.rule for c in cases),
        "fix_rule": count(c.labels.fix_signal.rule for c in cases),
        "human_reviewed": len(reviewed),
        "audit": {
            "audited": len(audited),
            "auto_label_agreed": audit_agree,
            "precision": round(audit_agree / len(audited), 3) if audited else None,
        },
        "rejection_reasons": count(r["reason"] for r in rejections),
    }


def format_stats(stats: dict[str, Any]) -> str:
    lines = []
    for key, value in stats.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)
