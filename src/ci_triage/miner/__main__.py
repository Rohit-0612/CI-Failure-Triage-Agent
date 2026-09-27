"""CLI for the dataset miner.

python -m ci_triage.miner mine [--repos owner/name ...] [--target N]
python -m ci_triage.miner stats [--json]
python -m ci_triage.miner annotate [--audit N]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from ci_triage.github_client import GitHubClient
from ci_triage.miner.annotate import apply_reviews, build_queue, render_case, run_review
from ci_triage.miner.config import load_settings
from ci_triage.miner.pipeline import Miner
from ci_triage.miner.schema import REVIEWED_STATUSES
from ci_triage.miner.stats import compute_stats, format_stats, load_cases, load_rejections


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx logs every request URL at INFO, including signed log-download URLs that
    # act as temporary credentials. GitHubClient logs its own sanitized request lines.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ci_triage.miner")
    parser.add_argument("--config", type=Path, default=Path("data/repos.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)

    mine = sub.add_parser("mine", help="mine failed CI runs into data/processed/cases.jsonl")
    mine.add_argument("--repos", nargs="+", help="override the repo list from the config")
    mine.add_argument("--target", type=int, help="stop after this many accepted cases")
    mine.add_argument("--max-per-repo", type=int, help="cap accepted cases per repo")

    stats = sub.add_parser("stats", help="print dataset statistics")
    stats.add_argument("--json", action="store_true", help="print JSON instead of text")

    annotate = sub.add_parser("annotate", help="review needs_review cases + an audit sample")
    annotate.add_argument("--audit", type=int, default=5, help="audit sample size (total)")
    annotate.add_argument("--seed", type=int, default=0, help="audit sampling seed")
    annotate.add_argument(
        "--export",
        type=Path,
        help="write the review queue to a file and exit, instead of prompting",
    )
    annotate.add_argument(
        "--blind",
        action="store_true",
        help="hide the automatic label, its rule, and review/audit status (use for --export, "
        "so an audit measures the labeler rather than the reviewer's agreement with it)",
    )
    annotate.add_argument(
        "--apply",
        type=Path,
        help="apply decisions from a JSON file {case_id: {category, root_cause, fix_text, "
        "notes}} instead of prompting; a null category records notes and keeps needs_review",
    )
    annotate.add_argument(
        "--status",
        choices=sorted(REVIEWED_STATUSES),
        default="human_verified",
        help="who did the review being applied (--apply only)",
    )

    args = parser.parse_args(argv)
    configure_logging()
    load_dotenv()

    if args.command == "mine":
        settings = load_settings(
            args.config,
            repos=args.repos,
            target_cases=args.target,
            max_cases_per_repo=args.max_per_repo,
        )
        gh = GitHubClient.from_env(cache_dir=settings.cache_dir)
        try:
            report = Miner(settings, gh).run()
        finally:
            gh.close()
        print(json.dumps(report, indent=2))
        return

    settings = load_settings(args.config)
    cases_path = settings.processed_dir / "cases.jsonl"
    if args.command == "stats":
        result = compute_stats(
            load_cases(cases_path), load_rejections(settings.processed_dir / "rejected.jsonl")
        )
        print(json.dumps(result, indent=2) if args.json else format_stats(result))
    elif args.command == "annotate":
        if args.apply:
            decisions = json.loads(args.apply.read_text(encoding="utf-8"))
            # Which cases are the audit sample is decided here, not in the decisions
            # file: audit precision needs the flag set, but a reviewer who knew a case
            # was an audit case would know the auto-labeler was confident about it.
            # Computed after the decisions are written, and before they are applied -
            # the queue is derived from label_status, which applying will change.
            audit_ids = {
                cid
                for cid, why in build_queue(load_cases(cases_path), args.audit, args.seed)
                if why == "audit"
            }
            for case_id in decisions.keys() & audit_ids:
                decisions[case_id]["audit"] = True
            result = apply_reviews(cases_path, decisions, args.status)
            result["audit_cases_flagged"] = len(decisions.keys() & audit_ids)
            print(json.dumps(result, indent=2))
        elif args.export:
            queue = build_queue(load_cases(cases_path), args.audit, args.seed)
            by_id = {c.case_id: c for c in load_cases(cases_path)}
            rendered = [
                render_case(by_id[cid], f"{n}/{len(queue)}", why, blind=args.blind)
                for n, (cid, why) in enumerate(queue, start=1)
            ]
            args.export.write_text("\n\n".join(rendered), encoding="utf-8")
            print(f"wrote {len(queue)} case(s) to {args.export}")
        else:
            run_review(cases_path, args.audit, seed=args.seed)


if __name__ == "__main__":
    main()
