"""Run a system over a dataset split, store predictions, and score them.

Systems only ever receive `CaseRecord.visible()` (a `CaseView`); ground truth and labels
are read here, in the evaluator, and nowhere else.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ci_triage.baseline import analyze as baseline
from ci_triage.diagnosis import Diagnosis
from ci_triage.evaluation import metrics
from ci_triage.llm import Usage
from ci_triage.miner.schema import CaseRecord, CaseView
from ci_triage.paths import is_doc_path
from ci_triage.taxonomy import FailureCategory

logger = logging.getLogger(__name__)

SystemFn = Callable[[CaseView], Diagnosis]


@dataclass
class SystemRun:
    """A system ready to be evaluated.

    `name` identifies what produced the predictions and includes the model for LLM
    systems, so a report can never be mistaken for one from a different model.
    `usage_of` returns the last case's token/call counts, if the system tracks them.
    """

    name: str
    analyze: SystemFn
    usage_of: Callable[[], Usage] | None = None


def _baseline_system(trace_dir: Path | None = None) -> SystemRun:
    return SystemRun(baseline.NAME, baseline.analyze)


def _agent_system(trace_dir: Path | None = None) -> SystemRun:
    # Imported lazily: the baseline must stay usable without langgraph or a model server.
    from ci_triage.agent import run as agent_run

    investigator = agent_run.from_env(trace_dir)
    return SystemRun(investigator.name, investigator.analyze, lambda: investigator.last_usage)


SYSTEMS: dict[str, Callable[[Path | None], SystemRun]] = {
    "baseline": _baseline_system,
    "agent": _agent_system,
}

# Failures whose "fix window" is not a code repair (a metadata edit, a rerun, the network
# recovering): their changed files say nothing about where the fault was.
NON_CODE_CATEGORIES = frozenset(
    {
        FailureCategory.POLICY_CHECK_FAILURE,
        FailureCategory.FLAKY,
        FailureCategory.NETWORK_FAILURE,
        FailureCategory.ENVIRONMENT_FAILURE,
        FailureCategory.TIMEOUT,
    }
)


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    system: str
    diagnosis: Diagnosis | None
    error: str | None = None
    latency_ms: float
    # Cost accounting fields exist now so the LLM agent reports them in the same shape.
    tool_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def run_system(
    system: str, cases: list[CaseRecord], *, trace_dir: Path | None = None
) -> list[Prediction]:
    run = SYSTEMS[system](trace_dir)
    predictions = []
    for index, case in enumerate(cases, start=1):
        view = case.visible()
        started = time.perf_counter()
        try:
            diagnosis, error = run.analyze(view), None
        except Exception as exc:  # one broken case must not abort the whole evaluation
            diagnosis, error = None, f"{type(exc).__name__}: {exc}"[:500]
        usage = run.usage_of() if run.usage_of else Usage(0, 0, 0)
        predictions.append(
            Prediction(
                case_id=case.case_id,
                system=run.name,
                diagnosis=diagnosis,
                error=error,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                llm_calls=usage.calls,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
        )
        logger.info(
            "[%d/%d] %s %s in %.1fs",
            index,
            len(cases),
            case.case_id,
            error or (diagnosis.failure_type if diagnosis else "?"),
            predictions[-1].latency_ms / 1000,
        )
    return predictions


def gold_fix_files(case: CaseRecord) -> tuple[set[str] | None, str]:
    """Code files the fix changed, or (None, why the case is excluded from localization)."""
    gt = case.ground_truth
    if gt.fix_status != "matched":
        return None, f"fix_status:{gt.fix_status}"
    if case.labels.category in NON_CODE_CATEGORIES:
        return None, f"non_code_category:{case.labels.category}"
    files = {f for f in gt.fix_window.files_changed if not is_doc_path(f)}
    if not files:
        return None, "fix_changes_only_docs"
    return files, "eligible"


def grounding_corpus(view: CaseView) -> str:
    """All text the investigator was given; evidence must quote from it."""
    inp = view.input
    parts = [inp.log_excerpt, *inp.error_lines, inp.failed_commit.message]
    if inp.breaking_window:
        parts.append(inp.breaking_window.diff)
        parts.extend(c.message for c in inp.breaking_window.commits)
    parts.extend(f.content for f in inp.relevant_files)
    return "\n".join(parts)


def evaluate(
    cases: list[CaseRecord], predictions: list[Prediction], system: str, split: str
) -> dict[str, Any]:
    by_id = {p.case_id: p for p in predictions}
    missing = [c.case_id for c in cases if c.case_id not in by_id]
    if missing:
        raise ValueError(f"no prediction for {len(missing)} case(s), e.g. {missing[:3]}")

    per_case: list[dict[str, Any]] = []
    pairs_by_status: dict[str, list[tuple[str, str]]] = {"all": []}
    loc_scores: list[dict[str, float]] = []
    excluded: Counter[str] = Counter()
    evidence_total = evidence_grounded = cases_with_ungrounded = 0

    for case in cases:
        pred = by_id[case.case_id]
        diag = pred.diagnosis
        gold = str(case.labels.category)
        guess = str(diag.failure_type) if diag else "ERROR"
        pairs_by_status["all"].append((gold, guess))
        pairs_by_status.setdefault(case.labels.label_status, []).append((gold, guess))

        row: dict[str, Any] = {
            "case_id": case.case_id,
            "gold": gold,
            "gold_status": case.labels.label_status,
            "pred": guess,
            "correct": gold == guess,
        }
        gold_files, why = gold_fix_files(case)
        if gold_files is None:
            excluded[why] += 1
        else:
            scores = metrics.localization(diag.affected_files if diag else [], gold_files)
            loc_scores.append(scores)
            row["localization"] = scores
            row["gold_files"] = sorted(gold_files)
            row["top_files"] = diag.affected_files[:3] if diag else []

        if diag:
            corpus = grounding_corpus(case.visible())
            ungrounded = [e.excerpt[:80] for e in diag.evidence if e.excerpt not in corpus]
            evidence_total += len(diag.evidence)
            evidence_grounded += len(diag.evidence) - len(ungrounded)
            if ungrounded:
                cases_with_ungrounded += 1
                row["ungrounded_evidence"] = ungrounded
        per_case.append(row)

    latencies = [p.latency_ms for p in predictions]
    predicted = [p.diagnosis for p in predictions if p.diagnosis]
    return {
        "system": predictions[0].system if predictions else system,
        "split": split,
        "cases": len(cases),
        "errors": sum(p.error is not None for p in predictions),
        "category_accuracy": {
            status: metrics.category_accuracy(pairs)
            for status, pairs in sorted(pairs_by_status.items())
        },
        "abstention_rate": round(
            sum(d.failure_type == FailureCategory.UNKNOWN for d in predicted) / len(cases), 3
        )
        if cases
        else None,
        "localization": {
            **metrics.summarize_localization(loc_scores),
            "excluded": dict(excluded.most_common()),
        },
        "evidence_grounding": {
            "items": evidence_total,
            "grounded": evidence_grounded,
            "rate": round(evidence_grounded / evidence_total, 3) if evidence_total else None,
            "cases_with_ungrounded_evidence": cases_with_ungrounded,
        },
        "operational": {
            "latency_ms_mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "latency_ms_p95": metrics.percentile(latencies, 0.95),
            "tool_calls_total": sum(p.tool_calls for p in predictions),
            "llm_calls_total": sum(p.llm_calls for p in predictions),
            "cost_usd_total": round(sum(p.cost_usd for p in predictions), 4),
            "tokens": {
                "input": sum(p.input_tokens for p in predictions),
                "output": sum(p.output_tokens for p in predictions),
            },
        },
        "per_case": per_case,
    }


def write_jsonl(path: Path, predictions: list[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(p.model_dump_json() + "\n" for p in predictions), encoding="utf-8")


def read_jsonl(path: Path) -> list[Prediction]:
    return [
        Prediction.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
