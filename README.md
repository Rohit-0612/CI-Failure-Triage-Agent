# CI-Failure-Triage-Agent

An AI-assisted developer tool that investigates failed GitHub Actions CI runs: it collects
evidence (logs, diffs, history), forms and verifies a root-cause hypothesis, and proposes a
fix that a human must approve before anything is applied.

## Status

This project is built incrementally. What exists today:

| Phase | Component | Status |
|---|---|---|
| 0 | Repository foundation (uv, ruff, pytest, CI) | done |
| 1 | Real CI-failure dataset miner: 50-case dev split + 25-case held-out test split | done (human label review pending) |
| 2 | Deterministic rule baseline + evaluation harness | done |
| 3+ | LangGraph agent, tools, retrieval, fix verification, UI | not started |

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
uv run python -m ci_triage.miner mine                                # dev split (data/repos.yaml)
uv run python -m ci_triage.miner --config data/repos_test.yaml mine  # held-out test split
uv run python -m ci_triage.miner stats                               # statistics from the JSONL
uv run python -m ci_triage.miner annotate                            # human review + audit sample
```

### How it works

1. For each repo, list failed workflow runs (last 80 days; Actions logs expire after ~90).
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

### Dataset

| | Dev split | Held-out test split |
|---|---|---|
| File | `data/processed/dev/cases.jsonl` | `data/processed/test/cases.jsonl` |
| Cases / repositories | 50 / 14 | 25 / 7 (disjoint from dev) |
| Red streaks examined | 246 | 132 |
| Fix status | 44 matched (all single-commit), 6 same-commit reruns | 25 matched (18 high, 7 medium confidence) |
| Labels | 32 `auto_verified`, 18 `needs_review` | 19 `auto_verified`, 6 `needs_review` |
| Human-reviewed labels | 0 | 0 |

Dev category distribution (automatic labels): FORMAT 9, TEST 8, POLICY_CHECK 7, TYPE 6,
LINT 5, COVERAGE 5, DEPENDENCY 4, CI_CONFIGURATION 3, BUILD / NETWORK / SYNTAX 1 each.
Test: TEST 8, LINT 5, UNKNOWN 5, TYPE 4, BUILD 2, CI_CONFIGURATION 1.

The dev split is the data the labeling and baseline rules were developed on. The test split
comes from repositories that were never inspected while writing rules; it was mined and
evaluated only after the baseline was committed.

## Phase 2: rule baseline and evaluation harness

A deterministic investigator (no LLM) that sets the floor later systems must beat, plus the
harness that scores any investigator the same way.

```bash
uv run python -m ci_triage.evaluation run --system baseline --split dev
uv run python -m ci_triage.evaluation score --system baseline --split test   # re-score only
```

- Systems receive a `CaseView` (failure-time information only; the type has no ground-truth
  fields) and must return a structured `Diagnosis`: category, root-cause hypothesis,
  confidence, verbatim evidence, ranked suspect files, verification status.
- The baseline classifies with ordered log rules (independent of the dataset labeler,
  enforced by an import-guard test) and ranks suspect files from log references and the
  code changed since the last passing state.
- Metrics: category accuracy per label status with Wilson 95% intervals; file-level fault
  localization (hit@1, hit@3, MRR) against the files the real fix changed, excluding
  non-code fixes; evidence grounding (every quoted excerpt must exist in the input);
  abstention; latency and cost.

### Results (rule baseline v1)

| Metric | Dev (50) | Held-out test (25) |
|---|---|---|
| Localization hit@1 | 57.1% (24/42, CI 42-71%) | **43.5%** (10/23, CI 26-63%) |
| Localization hit@3 | 78.6% (CI 64-88%) | **56.5%** (CI 37-74%) |
| Localization MRR | 0.682 | **0.505** |
| Category accuracy vs automatic labels | 94.0% | 80.0% |
| Evidence grounding | 121/121 | 62/62 |
| Abstention (UNKNOWN) | 0% | 16% |
| Mean latency / cost | 2.4 ms / $0 | 2.3 ms / $0 |

How to read this:

- **Fault localization is the meaningful number**: it is scored against the real fix and
  does not depend on labels. It drops on held-out repositories, so the dev numbers were
  optimistic.
- **Category accuracy is not yet trustworthy.** The gold labels are automatic and partly
  come from similar log patterns, so agreement is inflated (100% on dev `needs_review`
  cases). Of the 5 test disagreements, 2 look like label errors, 2 are baseline errors and 1
  depends on hindsight. Category accuracy becomes meaningful only after the human review
  (`annotate`), followed by `evaluation score`.
- Known baseline v1 bugs found in the held-out error analysis, deliberately **not** fixed in
  v1 so the test numbers stay clean: the coverage rule also matches the coverage *success*
  message ("Required test coverage of 99.0% reached"), and coverage is checked before test
  failures although failing tests are the usual cause of low coverage. A fixed v2 has to be
  evaluated on new held-out data.
- Dev-split ablation for localization (hit@1): log references only 0.405, changed files only
  0.500, combined 0.571.

### Known limitations

- **A green run is not proof of a causal fix.** Confidence `high` means structural evidence
  (a single commit between red and green, and the same job passing), not that the change is
  semantically confirmed. Example: some pip PR-template failures went green after an
  unrelated commit, while the real fix was editing the PR description.
- **Automatic labels describe the failure class, not always the deepest cause**, and the
  labeler was developed on the dev split (5 test cases are `UNKNOWN`). Label precision will
  be measured with a human audit sample.
- **Selection bias.** Popular, well-maintained, mostly pure-Python repos; only fixes that
  land on the same branch inside the 80-day window; single-commit fixes dominate. Most
  examined streaks never went green on the same branch and are excluded.
- **Small samples.** 50 + 25 cases across 16 categories; several categories have 0-1
  examples, and confidence intervals are wide.
- **Log coverage.** When the error is printed by an earlier step than the one that fails
  (e.g. a separate "error out" step), the excerpt misses it.
- For `pull_request` runs GitHub tests a merge commit with the base branch; base-branch
  changes between two runs are not part of the fix diff.
- The baseline's `confidence` is a fixed per-rule number, not a calibrated probability.

### Data provenance

All data comes from public GitHub repositories and their public Actions logs, fetched through
the GitHub REST API. Code excerpts and logs remain under their original projects' licenses.
Full job logs are kept gzipped in `data/raw/logs/` because GitHub deletes them after ~90 days.
