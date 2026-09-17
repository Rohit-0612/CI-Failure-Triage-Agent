"""CLI: python -m ci_triage.evaluation run --system baseline --split dev"""

from __future__ import annotations

import argparse
import json
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
        cmd.add_argument(
            "--resume",
            action="store_true",
            help="skip cases that already have a prediction (same system and split)",
        )

    compare = sub.add_parser("compare", help="table of every system's report for a split")
    compare.add_argument("--split", choices=["dev", "test"], default="dev")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "compare":
        print(compare_systems(args.data_dir / "eval" / args.split))
        return

    cases = load_cases(args.data_dir / "processed" / args.split / "cases.jsonl")
    if not cases:
        raise SystemExit(f"no cases found for split {args.split!r}")
    out_dir = args.data_dir / "eval" / args.split / args.system
    predictions_path = out_dir / "predictions.jsonl"

    if args.command == "run":
        kept = read_jsonl(predictions_path) if predictions_path.exists() else []
        selected = select_cases(cases, args.cases, args.limit)
        if args.resume:
            done = {p.case_id for p in kept}
            selected = [c for c in selected if c.case_id not in done]
            print(f"resuming: {len(done)} already done, {len(selected)} to go")
        fresh = {c.case_id for c in selected}
        kept = [p for p in kept if p.case_id not in fresh]

        # Save after every case: a local run takes over an hour and may be interrupted.
        def save(prediction) -> None:
            kept.append(prediction)
            write_jsonl(predictions_path, kept)

        run_system(args.system, selected, trace_dir=out_dir / "traces", on_prediction=save)
        write_jsonl(predictions_path, kept)

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


def compare_systems(split_dir: Path) -> str:
    """Markdown table of every report under data/eval/<split>/*/report.json.

    Rows are metrics, columns are systems, so a report can be pasted into the README
    without retyping numbers (and without mixing up which model produced them).
    """
    reports = [json.loads(p.read_text()) for p in sorted(split_dir.glob("*/report.json"))]
    if not reports:
        raise SystemExit(f"no reports under {split_dir}")

    def pct(value: dict[str, Any] | None) -> str:
        if not value or not value.get("n"):
            return "-"
        low, high = value["ci95"]
        return f"{value['rate']:.1%} ({value['hits']}/{value['n']}, CI {low:.0%}-{high:.0%})"

    rows: list[tuple[str, list[str]]] = [
        ("cases scored", [str(r.get("scored_cases_of_split", r["cases"])) for r in reports]),
        (
            "category accuracy (auto labels)",
            [pct(r["category_accuracy"].get("all")) for r in reports],
        ),
        ("localization hit@1", [pct(r["localization"].get("hit@1")) for r in reports]),
        ("localization hit@3", [pct(r["localization"].get("hit@3")) for r in reports]),
        ("localization MRR", [str(r["localization"].get("mrr", "-")) for r in reports]),
        (
            "evidence grounded",
            [
                f"{r['evidence_grounding']['grounded']}/{r['evidence_grounding']['items']}"
                for r in reports
            ],
        ),
        ("abstention (UNKNOWN)", [str(r["abstention_rate"]) for r in reports]),
        ("errors", [str(r["errors"]) for r in reports]),
        (
            "mean latency",
            [f"{(r['operational']['latency_ms_mean'] or 0) / 1000:.1f}s" for r in reports],
        ),
        ("LLM calls", [str(r["operational"]["llm_calls_total"]) for r in reports]),
        ("cost", [f"${r['operational']['cost_usd_total']}" for r in reports]),
    ]
    header = ["metric", *(r["system"] for r in reports)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += [f"| {name} | " + " | ".join(values) + " |" for name, values in rows]
    return "\n".join(lines)


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
