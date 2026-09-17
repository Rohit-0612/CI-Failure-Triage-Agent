"""The investigation graph.

    pack_evidence -> propose -> validate -+-> finalize        (valid)
                       ^                 |
                       |                 +-> repair  -> validate   (validator found problems)
                       |                 |
                       +-- expand -------+                         (answered UNKNOWN)

The validator is deterministic code, not another model call: it checks that quotes exist
verbatim in the evidence, that file paths were not invented, that the category is in the
taxonomy, and that the answer does not restate the instructions. That is what makes the
loop worth having - the model gets concrete, checkable feedback instead of "are you sure?".

LLM calls are capped (default 3) because a local 7B model needs ~90 s per call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ci_triage.agent import evidence as ev
from ci_triage.agent.prompts import (
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    build_repair_prompt,
    build_user_prompt,
)
from ci_triage.diagnosis import MAX_EXCERPT_CHARS, Diagnosis, Evidence
from ci_triage.llm import LLMClient, LLMError, Usage
from ci_triage.miner.schema import CaseView
from ci_triage.taxonomy import FailureCategory

logger = logging.getLogger(__name__)

MAX_LLM_CALLS = 3
MAX_EVIDENCE_ITEMS = 6
MAX_AFFECTED_FILES = 10
# Phrases from our own instructions; if they come back in the answer, the model is
# echoing the prompt (usually because injected text told it to).
_INSTRUCTION_MARKERS = ("untrusted_evidence", "Security rules", "system prompt")


class InvestigationState(TypedDict, total=False):
    case: CaseView
    budget: int
    evidence: str
    proposal: dict[str, Any]
    problems: list[str]
    previous_problems: list[str]
    llm_calls: int
    usage: Usage
    trace: list[dict[str, Any]]
    diagnosis: Diagnosis
    error: str


def build_graph(client: LLMClient, max_llm_calls: int = MAX_LLM_CALLS):
    """Compile the investigation graph for one LLM client."""

    def record(state: InvestigationState, node: str, **detail: Any) -> list[dict[str, Any]]:
        return [*state.get("trace", []), {"node": node, "at": time.time(), **detail}]

    def pack_evidence(state: InvestigationState) -> InvestigationState:
        budget = state.get("budget") or ev.DEFAULT_BUDGET_CHARS
        pack = ev.pack(state["case"], budget_chars=budget)
        return {
            "evidence": pack.text,
            "budget": budget,
            "trace": record(
                state,
                "pack_evidence",
                chars=len(pack.text),
                sections=list(pack.used_chars),
                truncated=pack.truncated,
            ),
        }

    def _call(state: InvestigationState, prompt: str, node: str) -> InvestigationState:
        started = time.perf_counter()
        try:
            proposal, usage = client.complete_json(
                system=SYSTEM_PROMPT, user=prompt, schema=OUTPUT_SCHEMA
            )
        except LLMError as exc:
            return {
                "error": str(exc),
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": record(state, node, error=str(exc), raw=exc.raw_output[:500]),
            }
        return {
            "proposal": proposal,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "usage": state.get("usage", Usage(0, 0, 0)) + usage,
            "trace": record(
                state,
                node,
                seconds=round(time.perf_counter() - started, 1),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                answer=proposal.get("failure_type"),
            ),
        }

    def propose(state: InvestigationState) -> InvestigationState:
        return _call(state, build_user_prompt(state["evidence"]), "propose")

    def repair(state: InvestigationState) -> InvestigationState:
        prompt = build_repair_prompt(state["evidence"], state.get("problems", []))
        return _call(state, prompt, "repair")

    def expand(state: InvestigationState) -> InvestigationState:
        """Answer was UNKNOWN: re-pack with a bigger budget and ask once more."""
        pack = ev.pack(state["case"], budget_chars=ev.EXPANDED_BUDGET_CHARS)
        return {
            "evidence": pack.text,
            "budget": ev.EXPANDED_BUDGET_CHARS,
            "trace": record(state, "expand_evidence", chars=len(pack.text)),
        }

    def validate(state: InvestigationState) -> InvestigationState:
        if state.get("error"):
            return {"problems": [state["error"]]}
        problems = _validate(state["case"], state["evidence"], state.get("proposal") or {})
        return {
            "problems": problems,
            "previous_problems": state.get("problems", []),
            "trace": record(state, "validate", problems=problems),
        }

    def finalize(state: InvestigationState) -> InvestigationState:
        proposal = state.get("proposal")
        if proposal is None:
            return {"trace": record(state, "finalize", ok=False)}
        diagnosis = _to_diagnosis(state["case"], state["evidence"], proposal)
        return {
            "diagnosis": diagnosis,
            "trace": record(
                state,
                "finalize",
                failure_type=str(diagnosis.failure_type),
                evidence_items=len(diagnosis.evidence),
                files=diagnosis.affected_files[:3],
            ),
        }

    def route(state: InvestigationState) -> str:
        proposal = state.get("proposal") or {}
        calls_left = state.get("llm_calls", 0) < max_llm_calls
        if not proposal:
            return END  # transport failure with nothing to fall back on
        problems = state.get("problems") or []
        if problems:
            # Stop if the repair produced exactly the same complaints: the model is not
            # going to fix it, and each local call costs ~90-180 s.
            if problems == state.get("previous_problems"):
                return "finalize"
            # Retry while we can; otherwise finalize best-effort - the mapping step drops
            # ungrounded quotes and invented paths anyway.
            return "repair" if calls_left else "finalize"
        if (
            proposal.get("failure_type") == FailureCategory.UNKNOWN.value
            and calls_left
            and state.get("budget", 0) < ev.EXPANDED_BUDGET_CHARS
        ):
            return "expand"
        return "finalize"

    graph = StateGraph(InvestigationState)
    graph.add_node("pack_evidence", pack_evidence)
    graph.add_node("propose", propose)
    graph.add_node("validate", validate)
    graph.add_node("repair", repair)
    graph.add_node("expand", expand)
    graph.add_node("finalize", finalize)
    graph.set_entry_point("pack_evidence")
    graph.add_edge("pack_evidence", "propose")
    graph.add_edge("propose", "validate")
    graph.add_edge("repair", "validate")
    graph.add_edge("expand", "propose")
    graph.add_conditional_edges(
        "validate",
        route,
        {"repair": "repair", "expand": "expand", "finalize": "finalize", END: END},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


# ----------------------------------------------------------------------- validation


def known_paths(case: CaseView) -> set[str]:
    """Paths the case data names explicitly (resolved by the miner)."""
    inp = case.input
    paths = set(inp.log_referenced_files)
    paths.update(f.path for f in inp.relevant_files)
    if inp.breaking_window:
        paths.update(inp.breaking_window.files_changed)
    paths.update(test.split("::", 1)[0] for test in inp.failing_tests)
    return paths


def path_is_supported(path: str, case: CaseView, evidence_text: str) -> bool:
    """A file path is acceptable if the evidence mentions it at all.

    Not only the miner's resolved list: a log can print a path the miner could not map
    (e.g. Windows `tests\\pkg\\test_x.py`), and blaming the model for reading it is wrong.
    """
    if path in known_paths(case):
        return True
    candidate = path.replace("\\", "/").strip()
    return bool(candidate) and candidate in evidence_text.replace("\\", "/")


def _validate(case: CaseView, evidence_text: str, proposal: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    category = proposal.get("failure_type")
    if category not in {c.value for c in FailureCategory}:
        problems.append(f"failure_type {category!r} is not one of the allowed categories")

    root_cause = (proposal.get("root_cause") or "").strip()
    if not root_cause:
        problems.append("root_cause is empty")
    if any(marker in root_cause for marker in _INSTRUCTION_MARKERS):
        problems.append("root_cause repeats the instructions instead of describing the failure")

    quotes = [str(item.get("quote", "")) for item in proposal.get("evidence") or []]
    ungrounded = [q for q in quotes if q and q not in evidence_text]
    abstained = category == FailureCategory.UNKNOWN.value
    if not quotes and not abstained:
        # Abstention with no quotes is a legitimate answer; the graph responds by
        # offering more evidence (expand), not by asking for quotes that do not exist.
        problems.append("evidence is empty: quote at least one line from the evidence block")
    elif quotes and len(ungrounded) == len(quotes):
        problems.append(
            "none of the quotes appear verbatim in the evidence; the first was "
            f"{ungrounded[0][:120]!r}"
        )
    elif ungrounded:
        problems.append(
            f"{len(ungrounded)} quote(s) are not verbatim, e.g. {ungrounded[0][:120]!r}"
        )

    invented = [
        p
        for p in proposal.get("suspect_files") or []
        if not path_is_supported(p, case, evidence_text)
    ]
    if invented:
        problems.append(f"these file paths do not appear in the evidence: {invented[:3]}")
    return problems


# ----------------------------------------------------------------------- output mapping


def _locate(case: CaseView, quote: str) -> tuple[str, str] | None:
    """Where does this quote come from? Also serves as the grounding check."""
    inp = case.input
    if quote in inp.log_excerpt or any(quote in line for line in inp.error_lines):
        step = case.failure.failed_step_name or case.failure.job_name
        return "ci_log", f"failed step: {step}"
    window = inp.breaking_window
    if window:
        if quote in window.diff:
            return "git_diff", f"changes since {window.base_sha[:10]}"
        for commit in window.commits:
            if quote in commit.message:
                return "commit", commit.sha[:10]
    for file in inp.relevant_files:
        if quote in file.content:
            return "source_file", file.path
    return None


def _to_diagnosis(case: CaseView, evidence_text: str, proposal: dict[str, Any]) -> Diagnosis:
    items: list[Evidence] = []
    for entry in proposal.get("evidence") or []:
        quote = str(entry.get("quote", "")).strip()
        located = _locate(case, quote) if quote else None
        if located is None:
            continue  # ungrounded quotes are dropped, never reported as evidence
        source, location = located
        items.append(
            Evidence(
                source=source,
                location=location,
                excerpt=quote[:MAX_EXCERPT_CHARS],
                explanation=str(entry.get("why", ""))[:500],
            )
        )
        if len(items) >= MAX_EVIDENCE_ITEMS:
            break

    files = [
        p.replace("\\", "/")
        for p in proposal.get("suspect_files") or []
        if path_is_supported(p, case, evidence_text)
    ][:MAX_AFFECTED_FILES]
    category = proposal.get("failure_type")
    if category not in {c.value for c in FailureCategory}:
        category = FailureCategory.UNKNOWN.value
    confidence = proposal.get("confidence")
    confidence = float(confidence) if isinstance(confidence, int | float) else 0.5
    return Diagnosis(
        failure_type=FailureCategory(category),
        root_cause=(proposal.get("root_cause") or "no root cause given").strip()[:2000],
        confidence=min(1.0, max(0.0, confidence)),
        evidence=items,
        affected_files=files,
    )
