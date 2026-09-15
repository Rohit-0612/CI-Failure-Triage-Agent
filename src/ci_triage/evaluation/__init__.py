"""Evaluation harness: run any investigator over a dataset split and score it.

Contract: a system turns each `CaseView` into a `Diagnosis`; predictions are stored as
JSONL; the evaluator compares them with ground truth it alone can read.
"""
