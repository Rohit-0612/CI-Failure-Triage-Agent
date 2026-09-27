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
| 3 | LangGraph investigation agent (local model) + prompt-injection defence | done |
| 4 | Repository tools for the agent, pinned to the failed commit | done — measured, and they did not help this model |
| 5+ | Retrieval, fix generation, isolated verification, approval UI, webhooks | not started |

Nothing beyond the table above is implemented yet: there is no API, database, UI or hosted
tracing in this repository today.

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
uv run python -m ci_triage.miner annotate                            # review queue + audit sample
uv run python -m ci_triage.miner annotate --export q.txt --blind     # queue without the auto label
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
| Labels after review | 27 `auto_verified`, 22 `model_reviewed`, 1 `needs_review` | 19 `auto_verified`, 6 `needs_review` |
| Reviewed labels | 22 model-reviewed, 1 still open | 0 |
| Human-reviewed labels | **0** | **0** |

Dev category distribution after the review below: FORMAT 10, TEST 10, POLICY_CHECK 7,
TYPE 6, COVERAGE 5, LINT 4, CI_CONFIGURATION 3, BUILD / DEPENDENCY / ENVIRONMENT / NETWORK /
TIMEOUT 1 each. Test (automatic labels): TEST 8, LINT 5, UNKNOWN 5, TYPE 4, BUILD 2,
CI_CONFIGURATION 1.

The dev split is the data the labeling and baseline rules were developed on. The test split
comes from repositories that were never inspected while writing rules; it was mined and
evaluated only after the baseline was committed.

### Label review

The automatic labels on the dev split's review queue - 18 `needs_review` cases plus a 5-case
audit sample - were re-decided from the failure evidence and the real fix, **blind**: the
automatic label, its confidence, the rule behind it, and whether a case was a review or an
audit case were all hidden. Anchoring is the whole risk in auditing your own labeler, and a
precision figure produced by a reviewer who already saw the label measures the anchoring.

22 of the 23 were decided; 1 was declined and left open, because its excerpt carries a
truncated `ExceptionGroup` and no error class, and two categories are equally defensible on
that evidence. Forcing a category there would write a wrong gold label, and every system
would then be scored against it.

| | Result |
|---|---|
| Audit sample (`auto_verified` cases) | **5/5** kept the automatic label |
| `needs_review` cases corrected | **7 of 18** |
| Cases declined, still open | 1 |

Three of the seven corrections were the automatic labeler matching a substring: a
`SyntaxError` raised by pytest's own match-expression parser is not "source could not be
parsed", and pipx's test suite always logs `No matching distribution found for pycowsay`,
which is noise rather than a dependency failure.

**These labels are `model_reviewed`, not `human_verified`.** The distinction is in the data,
and every report splits category accuracy by it, so no number here can quietly borrow the
stronger claim. The circularity is reduced - the gold labels no longer come from the same
log regexes the baseline uses - but not removed: the labels are still a language model's
judgement. The test split's queue (11 cases) has not been reviewed at all.

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
| Localization hit@1 | 57.5% (23/40, CI 42-72%) | **43.5%** (10/23, CI 26-63%) |
| Localization hit@3 | 77.5% (CI 62-88%) | **56.5%** (CI 37-74%) |
| Localization MRR | 0.683 | **0.505** |
| Category accuracy (dev: reviewed labels) | 80.0% | 80.0% |
| Evidence grounding | 121/121 | 62/62 |
| Abstention (UNKNOWN) | 0% | 16% |
| Mean latency / cost | 2.4 ms / $0 | 2.3 ms / $0 |

How to read this:

- **Fault localization is the meaningful number**: it is scored against the real fix and
  does not depend on labels. It drops on held-out repositories, so the dev numbers were
  optimistic.
- **Category accuracy was inflated, and the review shows by how much.** On dev it was
  94.0% against purely automatic labels; against the reviewed labels it is 80.0%, and the
  per-status split says where the difference lives: **92.6% on `auto_verified` cases against
  63.6% on the reviewed ones**. Scoring a regex system against regex-derived labels measured
  agreement between two implementations of the same idea. The test split has not been
  reviewed, so its 80.0% is still the inflated kind of number.
- Known baseline v1 bugs found in the held-out error analysis, deliberately **not** fixed in
  v1 so the test numbers stay clean: the coverage rule also matches the coverage *success*
  message ("Required test coverage of 99.0% reached"), and coverage is checked before test
  failures although failing tests are the usual cause of low coverage. A fixed v2 has to be
  evaluated on new held-out data.
- Dev-split ablation for localization (hit@1), measured before the label review: log
  references only 0.405, changed files only 0.500, combined 0.571.

## Phase 3: LangGraph investigation agent

An LLM investigator that receives the same `CaseView` and returns the same `Diagnosis` as the
baseline, so the harness scores both identically.

```bash
# needs a local model server: ollama serve; ollama pull qwen2.5-coder:7b
uv run python -m ci_triage.evaluation run --system agent --split dev --resume
uv run python -m ci_triage.evaluation compare --split test
```

```
pack_evidence -> propose -> validate -+-> finalize          (valid)
                   ^                  +-> repair -> validate (validator found problems)
                   +-- expand --------+                      (answered UNKNOWN)
```

- **The critic is code, not another model call.** The validator checks that every quote appears
  verbatim in the evidence, that file paths are supported by it, that the category is in the
  taxonomy, and that the answer does not restate the instructions. Its exact complaints go into
  the repair prompt. Abstention (`UNKNOWN`) is allowed and triggers a retry with more evidence.
- **Ungrounded output cannot survive.** Quotes are located in the evidence to derive their source;
  anything not found is dropped, as are unsupported file paths.
- **Prompt-injection defence** (repository text is attacker-controlled in principle): all untrusted
  text sits in one block with its delimiters defanged, the instruction hierarchy is explicit, and
  answers that echo the instructions are rejected. `tests/test_injection.py` pins containment, the
  rejection path and the absence of instruction leakage, plus an opt-in test against the real model
  (`pytest -m slow`). It does **not** claim the model can never be talked into the wrong category.
- Structured output is enforced by JSON-schema-constrained decoding, not by asking nicely.
- Every case writes a trace (`data/eval/<split>/agent/traces/<case_id>.json`) with nodes, timings,
  tokens, validator problems and the raw model output.

### Results: agent vs baseline

Model: `qwen2.5-coder:7b` running locally through Ollama (chosen for zero cost, not for quality).
Reports name the model, e.g. `agent_v1_ollama-qwen2.5-coder-7b`.

| Metric | dev: agent | dev: baseline | **test: agent** | **test: baseline** |
|---|---|---|---|---|
| Localization hit@1 | **65.0%** (26/40) | 57.5% (23/40) | **43.5%** (10/23) | **43.5%** (10/23) |
| Localization hit@3 | 72.5% | 77.5% | 52.2% | 56.5% |
| Localization MRR | 0.688 | 0.683 | 0.478 | 0.505 |
| Category accuracy (dev: reviewed labels) | 54.0% | 80.0% | 72.0% | 80.0% |
| Evidence grounded | 93/93 | 121/121 | 45/45 | 62/62 |
| Abstention (UNKNOWN) | 18% | 0% | 20% | 16% |
| Mean latency / cost | 137 s / $0 | ~0 s / $0 | 131 s / $0 | ~0 s / $0 |

**The honest headline: on held-out repositories the local 7B agent does not beat the rule
baseline.** It matches it on hit@1 and is slightly behind on hit@3 and MRR, while taking ~131
seconds per case instead of milliseconds. The dev-split advantage (65.0% vs 57.5%) did not
generalise — the same lesson the held-out split taught in Phase 2.

The dev category numbers are now scored against the reviewed labels, which moved both systems:
the agent up (48.0% to 54.0%) and the baseline down (94.0% to 80.0%), because the review removed
the advantage a regex system gets from regex-derived labels. On the cases a reviewer decided, the
gap is 45.5% for the agent against 63.6% for the baseline. The held-out test split has **not** been
reviewed, so its category column is still agreement with the automatic labeler.

The remaining agent "mistakes" are largely taxonomy-boundary calls (a formatter failure reported as
a lint failure) and honest abstentions. The labels are `model_reviewed`, not `human_verified` — see
[Label review](#label-review).

What this chapter is really worth is the harness around the model: a deterministic validation loop,
grounding that cannot be bypassed, injection containment, per-case traces, and token/cost
accounting. Swapping in a hosted model is a second implementation of the `LLMClient` protocol plus
an environment variable; the measured cost fields are already in every prediction.

### Known limitations

- Two of 75 cases failed outright: one where the model quoted a long traceback and hit the output
  token cap mid-JSON, and one that exceeded the 300 s model timeout. Both are counted as errors in
  the reports rather than hidden.
- A 7B local model is the weakest link here; these numbers should not be read as what an LLM agent
  can do on this benchmark, only as what this model does.
- **A green run is not proof of a causal fix.** Confidence `high` means structural evidence
  (a single commit between red and green, and the same job passing), not that the change is
  semantically confirmed. Example: some pip PR-template failures went green after an
  unrelated commit, while the real fix was editing the PR description.
- **Labels describe the failure class, not always the deepest cause.** On dev, a blind audit
  put the automatic labeler at 5/5 on the cases it was confident about, and 7 of 18 wrong on
  the ones it flagged itself. The labeler was developed on the dev split, and the test split's
  labels are still entirely automatic (5 of its cases are `UNKNOWN`). No label in this dataset
  has been checked by a person.
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

## Phase 4: repository tools

### Deciding what to build, before building it

`hit@1` says how often localization is wrong. It does not say *why*, so it cannot say what to
build next. This splits every miss by the information the investigator actually held:

```bash
uv run python -m ci_triage.evaluation headroom --system agent --split dev --detail
```

| Bucket | dev, agent v1 | Meaning |
|---|---|---|
| hit | 64.3% | hit@1 was already correct |
| reasoning | 4.8% | the gold file's content was in the prompt; the model chose another |
| **name_only** | **16.7%** | the gold path was visible as a name, its content was never sent |
| **blind** | **11.9%** | the path was absent entirely, though the file exists at that commit |
| impossible | 2.4% | the file did not exist at the failed commit |

The two middle rows are what tools or retrieval can reach: **28.6%**. The threshold for
building them (15%) was written down before the numbers were, and the largest bucket points at
a specific flaw — the evidence packer must guess two files in advance and guesses wrong in one
case out of six. A `read_file` tool replaces that guess with a request.

An earlier version of this analysis read "which files had content" from the dataset rather than
from the pack the model received, counting files the packer had dropped as files the model saw
and rejected. It reported 14.3% reachable and 19% reasoning failures — the opposite conclusion,
and below the threshold. `tests/test_headroom.py` pins the corrected behaviour.

### The tools

```bash
uv run python -m ci_triage.evaluation run --system agent_tools --split dev --limit 15
```

A gathering loop runs before the diagnosis: `decide -> act -> decide`, up to three lookups,
then the normal propose/validate/repair cycle. `read_file`, `search_code` and `list_files` are
the whole allowlist; `run_tests` belongs with Phase 7's isolated environment.

- **Every tool reads the failed commit, and the ref is not a parameter.** The local clone also
  contains the commit that fixed the failure, so a tool that could be pointed elsewhere would
  put the answer into the input and invalidate every measurement here. There is no argument to
  point it with; tests assert that `ref`/`sha`/`commit`/`branch` arguments are ignored and that
  no tool output contains the fix.
- **Tool arguments are untrusted**, since they are written by a model that has just read
  attacker-controlled repository text: absolute paths, `..` and `.git` are refused, and searches
  are fixed-string rather than regex.
- **Tool output is untrusted too** — it is repository text arriving through a channel the model
  chose — so it is sanitised and capped like any other evidence, and appended *inside* the
  untrusted block.
- A refused call is a result the model can read, not a lost case; a repeated call ends the
  gathering loop, because the same request cannot return anything new.

Searching needs file contents, and the clones are blobless. Fetching them one lazy read at a
time costs ~0.9 s per file; the commit is instead hydrated once with a size-limited filter,
measured at 1.5 s, after which `git grep` is local.

### Results: the tools did not help this model

A 15-case pilot on the dev split, both systems run on the same cases (a seeded random sample
of the localization-eligible cases, so the sample cannot favour the buckets tools were built
for). The rule for what to do next was written down before the run: scale up only if
`agent_tools` beats the tool-free agent.

| | agent v2 (no tools) | agent_tools | baseline |
|---|---|---|---|
| Localization hit@1 | 10/14 | **10/14** | 8/14 |
| Cases won by tools | — | **0** | — |
| Cases lost to tools | — | **0** | — |
| Mean latency | 98 s | **207 s** | ~0 s |
| LLM calls | 20 | **63** | 0 |
| Tool calls | 0 | 42 | 0 |
| Errors | 0 | 0 | 0 |

**Paired on all 14 cases the two systems could be compared on, the outcome is identical.**
Not "similar" — the same cases right and the same cases wrong, for 2.1x the wall clock and
3.2x the model calls. So the pilot was not scaled to the full split, and the tools are not
claimed as an improvement.

The traces say why, and it is not that the tools failed. On three of the four misses the
model searched the repository for text that was never code:

| Missed case | What it searched for |
|---|---|
| fastapi coverage gate | `coverage report --fail-under=100` — a command line from the log |
| fastapi pre-commit | `pre-commit` — then read `.pre-commit-config.yaml` and blamed it |
| pydantic windows test | `Process completed with exit code 1` — a GitHub Actions log marker |

On the fourth it searched correctly, `read_file`'d the exact file the real fix changed, and
still answered with two files it had been handed in the original evidence.

So the gap is not reach, it is aim: a 7B model does not reliably distinguish "text from the
log" from "a symbol in the source", and giving it more reach does not teach it to. The
honest reading at n=14 is narrow — this says nothing about what a stronger model would do
with the same tools, and the confidence intervals here overlap almost completely. What it
does say is that the next thing to build is not another tool.

One caveat on the headroom command when pointed at a tool-using system: its `name_only` and
`blind` buckets describe what the *evidence packer* sent, so they do not move when a tool
later fetches the file. It is a before-you-build diagnostic, not a scorecard for these runs.

### Data provenance

All data comes from public GitHub repositories and their public Actions logs, fetched through
the GitHub REST API. Code excerpts and logs remain under their original projects' licenses.
Full job logs are kept gzipped in `data/raw/logs/` because GitHub deletes them after ~90 days.
