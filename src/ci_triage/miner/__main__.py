"""CLI: `python -m ci_triage.miner mine [--repos owner/name ...] [--target N]`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from ci_triage.github_client import GitHubClient
from ci_triage.miner.config import load_settings
from ci_triage.miner.pipeline import Miner


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
    sub = parser.add_subparsers(dest="command", required=True)

    mine = sub.add_parser("mine", help="mine failed CI runs into data/processed/cases.jsonl")
    mine.add_argument("--config", type=Path, default=Path("data/repos.yaml"))
    mine.add_argument("--repos", nargs="+", help="override the repo list from the config")
    mine.add_argument("--target", type=int, help="stop after this many accepted cases")
    mine.add_argument("--max-per-repo", type=int, help="cap accepted cases per repo")

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


if __name__ == "__main__":
    main()
