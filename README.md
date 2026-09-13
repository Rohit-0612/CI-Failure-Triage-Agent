# CI-Failure-Triage-Agent

An AI-assisted developer tool that investigates failed GitHub Actions CI runs: it collects
evidence (logs, diffs, history), forms and verifies a root-cause hypothesis, and proposes a
fix that a human must approve before anything is applied.

## Status

This project is built incrementally. What exists today:

| Phase | Component | Status |
|---|---|---|
| 0 | Repository foundation (uv, ruff, pytest, CI) | done |
| 1 | Real CI-failure dataset miner | in progress |
| 2+ | Baseline analyzer, LangGraph agent, tools, retrieval, fix verification, UI | not started |

Nothing beyond the table above is implemented yet.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # create .venv and install dependencies
cp .env.example .env    # then set GITHUB_TOKEN (read-only access to public repos)
uv run pytest           # run the test suite
uv run ruff check .     # lint
```
