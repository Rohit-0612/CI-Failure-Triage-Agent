# CI-Failure-Triage-Agent

An AI-assisted developer tool that investigates failed GitHub Actions CI runs: it collects
evidence (logs, diffs, history), forms and verifies a root-cause hypothesis, and proposes a
fix that a human must approve before anything is applied.

## Status

This project is built incrementally. What exists today:

| Phase | Component | Status |
|---|---|---|
| 0 | Repository foundation (uv, ruff, pytest, CI) | done |
| 1 | Real CI-failure dataset miner + 50-case dataset | done (human label review pending) |
| 2+ | Baseline analyzer, LangGraph agent, tools, retrieval, fix verification, UI | not started |

Nothing beyond the table above is implemented yet: there is no LLM, agent, API, database,
UI or tracing in this repository today.

## Setup

Requires Python 3.11+, git and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # create .venv and install dependencies
cp .env.example .env    # then set GITHUB_TOKEN (read-only access to public repos)
uv run pytest           # run the test suite
uv run ruff check .     # lint
```

## Phase 1: CI failure dataset miner

The miner collects **real** failed GitHub Actions runs from public Python repositories and
pairs each failure with the code change that made CI green again.

```bash
uv run python -m ci_triage.miner mine        # mine using data/repos.yaml
uv run python -m ci_triage.miner stats       # dataset statistics (computed from the JSONL)
uv run python -m ci_triage.miner annotate    # human review queue + audit sample
```

### How it works

1. For each repo in `data/repos.yaml`, list failed workflow runs (last 80 days; Actions logs
   expire after ~90).
2. Group runs into **red streaks** by `(workflow, event, head repository, branch)`, so forks
   that share a branch name like `main` never mix.
3. For each streak, take the last red run and the next green run on the same key. The code
   between them is the **fix window**.
4. Compute the fix window with a local blobless git clone (`git diff red green`), not the
   compare API. On amended/force-pushed PR commits the compare API diffs from the merge base
   and reports the whole PR as the "fix".
5. Score how much the fix window can be trusted (`matched` / `flaky_rerun` / `ambiguous` /
   `no_green_found`) with a recorded list of reasons. Only trusted cases are kept.
6. Slice the failed step out of the job log (step timestamps + runner markers), extract
   error lines, failing tests, file references and the error signature.
7. Label the failure from two independent signals: what the log shows, and what the fix
   actually changed (file kinds + AST comparison: formatting-only, lint suppression,
   annotation-only...). Agreement → `auto_verified`; conflict or weak evidence →
   `needs_review` for a human.

Each record separates what an investigator could know **at failure time** (`input`) from
information from the future (`ground_truth`: the fix). Schema validation rejects any record
where a fix commit SHA appears in `input`.

### Dataset (first milestone)

Computed by `python -m ci_triage.miner stats` on `data/processed/cases.jsonl`:

| | |
|---|---|
| Cases | 50 |
| Repositories | 14 |
| Red streaks examined | 246 (187 had no green run on the same branch, 7 ambiguous fixes, 2 without a failed job) |
| Events | 43 pull_request, 7 push |
| Fix status | 44 matched (all single-commit, 35 descendant + 9 amended commits), 6 same-commit reruns |
| Labels | 32 `auto_verified`, 18 `needs_review`, 0 human-reviewed yet |

Category distribution (automatic labels; not yet human-audited):

| Category | Cases | | Category | Cases |
|---|---|---|---|---|
| FORMAT_FAILURE | 9 | | COVERAGE_FAILURE | 5 |
| TEST_FAILURE | 8 | | DEPENDENCY_FAILURE | 4 (all `needs_review`) |
| POLICY_CHECK_FAILURE | 7 (all `needs_review`) | | CI_CONFIGURATION_FAILURE | 3 |
| TYPE_ERROR | 6 | | BUILD / NETWORK / SYNTAX | 1 each |
| LINT_FAILURE | 5 | | | |

### Known limitations

- **A green run is not proof of a causal fix.** Confidence `high` means structural evidence
  (a single commit between red and green, and the same job passing), not that the change is
  semantically confirmed. Example: some pip PR-template failures went green after an
  unrelated commit, while the real fix was editing the PR description.
- **Automatic labels describe the failure class, not always the deepest cause.** A test
  failure caused by an upstream deprecation can be labelled `TEST_FAILURE` when both signals
  agree. Label precision will be measured with a human audit sample before any evaluation
  relies on it.
- **Selection bias.** Popular, well-maintained, mostly pure-Python repos; only fixes that
  land on the same branch inside the 80-day window; single-commit fixes dominate. 76% of examined streaks
  never went green on the same branch (abandoned or superseded PRs) and are excluded.
- **Small sample.** 50 cases across 16 categories; several categories have 0-1 examples.
- **Log coverage.** When the error is printed by an earlier step than the one that fails
  (e.g. a separate "error out" step), the excerpt misses it.
- For `pull_request` runs GitHub tests a merge commit with the base branch; base-branch
  changes between two runs are not part of the fix diff.

### Data provenance

All data comes from public GitHub repositories and their public Actions logs, fetched through
the GitHub REST API. Code excerpts and logs remain under their original projects' licenses.
Full job logs are kept gzipped in `data/raw/logs/` because GitHub deletes them after ~90 days.
