"""Pure metric functions. No I/O, no knowledge of datasets or systems."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion.

    Preferred over the normal approximation because it behaves at small n and at 0% or
    100%, which are exactly the cases a 50-case benchmark produces.
    """
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def rate(successes: int, n: int) -> dict[str, object]:
    return {
        "n": n,
        "hits": successes,
        "rate": round(successes / n, 3) if n else None,
        "ci95": wilson_interval(successes, n),
    }


def category_accuracy(pairs: Iterable[tuple[str, str]]) -> dict[str, object]:
    """pairs: (gold, predicted). Returns accuracy with CI and a confusion matrix."""
    pairs = list(pairs)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for gold, pred in pairs:
        confusion[gold][pred] += 1
    correct = sum(g == p for g, p in pairs)
    return {
        **rate(correct, len(pairs)),
        "confusion": {g: dict(c.most_common()) for g, c in sorted(confusion.items())},
    }


def localization(ranked: list[str], gold: set[str]) -> dict[str, float]:
    """File-level fault localization scores for one case.

    hit@k: a correct file is among the top k. rr: reciprocal rank of the first correct
    file (0 if none). recall@5: share of gold files found in the top 5.
    """
    first = next((i for i, path in enumerate(ranked) if path in gold), None)
    return {
        "hit@1": float(first is not None and first < 1),
        "hit@3": float(first is not None and first < 3),
        "rr": 1.0 / (first + 1) if first is not None else 0.0,
        "recall@5": len(set(ranked[:5]) & gold) / len(gold) if gold else 0.0,
    }


def summarize_localization(per_case: list[dict[str, float]]) -> dict[str, object]:
    n = len(per_case)
    if n == 0:
        return {"n": 0}
    hits1 = int(sum(c["hit@1"] for c in per_case))
    hits3 = int(sum(c["hit@3"] for c in per_case))
    return {
        "n": n,
        "hit@1": rate(hits1, n),
        "hit@3": rate(hits3, n),
        "mrr": round(sum(c["rr"] for c in per_case) / n, 3),
        "recall@5": round(sum(c["recall@5"] for c in per_case) / n, 3),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return round(ordered[index], 3)
