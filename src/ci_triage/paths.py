"""Neutral path helpers shared by the miner, the baseline and the evaluator."""

from __future__ import annotations

import re

# Documentation and changelog files: never the locus of a code failure.
_DOC_PATH_RE = re.compile(r"(^docs?/|\.(md|rst)$|(^|/)(CHANGES|CHANGELOG|HISTORY|NEWS)[^/]*$)")


def is_doc_path(path: str) -> bool:
    return bool(_DOC_PATH_RE.search(path))
