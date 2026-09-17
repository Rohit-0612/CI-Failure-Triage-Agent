"""CLI: python -m ci_triage.evaluation run --system baseline --split dev"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from ci_triage.evaluation.runner import (
    SYSTEMS,
    evaluate,
    read_jsonl,
    run_system,
    write_jsonl,
    write_report,
)
from ci_triage.miner.schema import CaseRecord
from ci_triage.miner.stats import load_cases


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ci_triage.evaluation")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "run a system over a split, then score it"),
        ("score", "re-score existing predictions (e.g. after labels were reviewed)"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--system", choices=sorted(SYSTEMS), default="baseline")
        cmd.add_argument("--split", choices=["dev", "test"], default="dev")
        # Local models take ~1-3 minutes per case, so partial runs must be possible.
        cmd.add_argument("--limit", type=int, help="only the first N cases of the split")
        cmd.add_argument("--cases", nargs="+", help="only these case ids")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cases = load_cases(args.data_dir / "processed" / args.split / "cases.jsonl")
    if not cases:
        raise SystemExit(f"no cases found for split {args.split!r}")
    out_dir = args.data_dir / "eval" / args.split / args.system
    predictions_path = out_dir / "predictions.jsonl"

    if args.command == "run":
        selected = select_cases(cases, args.cases, args.limit)
        predictions = run_system(args.system, selected, trace_dir=out_dir / "traces")
        if len(selected) < len(cases):
            # Partial run: keep predictions for cases we did not re-run this time.
            existing = read_jsonl(predictions_path) if predictions_path.exists() else []
            fresh = {p.case_id for p in predictions}
            predictions = [p for p in existing if p.case_id not in fresh] + predictions
        write_jsonl(predictions_path, predictions)

    predictions = read_jsonl(predictions_path)
    scored = [c for c in cases if c.case_id in {p.case_id for p in predictions}]
    report = evaluate(scored, predictions, args.system, args.split)
    report["scored_cases_of_split"] = f"{len(scored)}/{len(cases)}"
    write_report(out_dir / "report.json", report)
    print(format_summary(report))


def select_cases(
    cases: list[CaseRecord], case_ids: list[str] | None, limit: int | None
) -> list[CaseRecord]:
    selected = cases
    if case_ids:
        wanted = set(case_ids)
        selected = [c for c in selected if c.case_id in wanted]
        missing = wanted - {c.case_id for c in selected}
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")
    if limit:
        selected = selected[:limit]
    return selected


def format_summary(report: dict[str, Any]) -> str:
    def fmt(r: dict[str, Any]) -> str:
        if not r.get("n"):
            return "n=0"
        return (
            f"{r['rate']:.1%} ({r['hits']}/{r['n']}, 95% CI {r['ci95'][0]:.0%}-{r['ci95'][1]:.0%})"
        )

    lines = [
        f"{report['system']} on {report['split']}: {report['cases']} cases "
        f"({report.get('scored_cases_of_split', '')} of split), {report['errors']} errors"
    ]
    for status, acc in report["category_accuracy"].items():
        lines.append(f"  category accuracy [{status}]: {fmt(acc)}")
    loc = report["localization"]
    if loc.get("n"):
        lines.append(f"  localization hit@1: {fmt(loc['hit@1'])}")
        lines.append(f"  localization hit@3: {fmt(loc['hit@3'])}")
        lines.append(f"  localization MRR: {loc['mrr']}  recall@5: {loc['recall@5']}")
    lines.append(f"  localization excluded: {loc['excluded']}")
    grounding = report["evidence_grounding"]
    lines.append(
        f"  evidence grounding: {grounding['grounded']}/{grounding['items']} items grounded"
    )
    lines.append(f"  abstention (UNKNOWN): {report['abstention_rate']}")
    ops = report["operational"]
    lines.append(
        f"  latency mean {ops['latency_ms_mean']} ms, p95 {ops['latency_ms_p95']} ms; "
        f"tool calls {ops['tool_calls_total']}, LLM calls {ops['llm_calls_total']}, "
        f"cost ${ops['cost_usd_total']}"
    )
    tokens = ops.get("tokens")
    if tokens:
        lines.append(f"  tokens: {tokens['input']} in, {tokens['output']} out")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
