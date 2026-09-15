"""Miner settings, loaded from a YAML file (default: data/repos.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MinerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # dev: cases we inspected while building rules. test: held-out cases from repos that
    # were never looked at, so evaluation numbers are not tuned to the data (ADR-018).
    split: Literal["dev", "test"] = "dev"
    repos: list[str] = Field(min_length=1)
    events: list[str] = ["push", "pull_request"]
    # Actions logs are deleted after ~90 days by default; stay safely inside that.
    lookback_days: int = Field(default=80, ge=1, le=90)
    target_cases: int = Field(default=50, ge=1)
    max_cases_per_repo: int = Field(default=4, ge=1)
    # Rerun-passed cases are useful but say nothing about fixes; don't let them dominate.
    max_flaky_cases: int = Field(default=10, ge=0)
    max_keys_per_repo: int = Field(default=60, ge=1)
    max_fix_window_commits: int = Field(default=5, ge=1)
    max_log_excerpt_chars: int = 20_000
    max_error_lines: int = 150
    max_diff_chars: int = 40_000
    max_file_chars: int = 12_000
    max_relevant_files: int = 6
    data_dir: Path = Path("data")

    @field_validator("repos")
    @classmethod
    def _repo_names(cls, repos: list[str]) -> list[str]:
        for name in repos:
            owner, _, repo = name.partition("/")
            if not owner or not repo or "/" in repo:
                raise ValueError(f"repo must look like 'owner/name': {name!r}")
        return repos

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed" / self.split

    @property
    def raw_logs_dir(self) -> Path:
        # Shared by all splits: case ids are unique, and logs are raw evidence only.
        return self.data_dir / "raw" / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"


def load_settings(path: Path, **overrides: object) -> MinerSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return MinerSettings.model_validate(raw)
