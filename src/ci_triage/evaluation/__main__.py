"""CLI: python -m ci_triage.evaluation run --system baseline --split dev"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args(argv)

    cases = load_cases(args.data_dir / "processed" / args.split / "cases.jsonl")
    if not cases:
        raise SystemExit(f"no cases found for split {args.split!r}")
    out_dir = args.data_dir / "eval" / args.split / args.system
    predictions_path = out_dir / "predictions.jsonl"
    if args.command == "run":
        write_jsonl(predictions_path, run_system(args.system, cases))
    report = evaluate(cases, read_jsonl(predictions_path), args.system, args.split)
    write_report(out_dir / "report.json", report)
    print(format_summary(report))


def format_summary(report: dict[str, Any]) -> str:
    def fmt(r: dict[str, Any]) -> str:
        if not r.get("n"):
            return "n=0"
        return (
            f"{r['rate']:.1%} ({r['hits']}/{r['n']}, 95% CI {r['ci95'][0]:.0%}-{r['ci95'][1]:.0%})"
        )

    lines = [f"{report['system']} on {report['split']}: {report['cases']} cases, "
             f"{report['errors']} errors"]  # fmt: skip
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
    return "\n".join(lines)


if __name__ == "__main__":
    main()
