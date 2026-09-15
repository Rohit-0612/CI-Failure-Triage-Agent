"""Deterministic, rule-based failure analysis: the floor every smarter system must beat.

Independence rule (ADR-020): this package must not import `ci_triage.miner.labeling` or
read ground truth. It receives a `CaseView`, which has no ground truth fields at all.
"""
