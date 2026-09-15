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
from ci_triage.miner.annotate import run_review
from ci_triage.miner.config import load_settings
from ci_triage.miner.pipeline import Miner
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
        run_review(cases_path, args.audit, seed=args.seed)


if __name__ == "__main__":
    main()
