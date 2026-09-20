"""The investigation graph.

    pack_evidence -> propose -> validate -+-> finalize        (valid)
                       ^                 |
                       |                 +-> repair  -> validate   (validator found problems,
                       |                 |                          or the JSON was truncated)
                       +-- expand -------+                         (answered UNKNOWN)
                       |                 |
                       +-- shrink -------+                         (the call timed out)

With a toolbox, a gathering loop runs first and its results are appended to the evidence:

    pack_evidence -> decide -+-> act -> decide     (a tool, while the budget lasts)
                             +-> propose ...      (the model is ready, or the budget is out)

The validator is deterministic code, not another model call: it checks that quotes exist
verbatim in the evidence, that file paths were not invented, that the category is in the
taxonomy, and that the answer does not restate the instructions. That is what makes the
loop worth having - the model gets concrete, checkable feedback instead of "are you sure?".

A failed call is not automatically a failed case. Three causes are told apart because the
remedies differ: a truncated answer is repairable (say what was wrong and ask again), a
timeout is repairable by sending less evidence, and an unreachable server is not repairable
at all. The first dev and test runs lost one case each to the first two.

LLM calls are capped (default 3) because a local 7B model needs ~90 s per call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ci_triage.agent import evidence as ev
from ci_triage.agent.prompts import (
    ACTION_SCHEMA,
    MAX_QUOTE_CHARS,
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    build_decide_prompt,
    build_repair_prompt,
    build_user_prompt,
)
from ci_triage.agent.tools import TOOL_NAMES, Toolbox, ToolResult
from ci_triage.diagnosis import MAX_EXCERPT_CHARS, Diagnosis, Evidence
from ci_triage.llm import LLMClient, LLMError, LLMOutputError, LLMTimeout, Usage
from ci_triage.miner.schema import CaseView
from ci_triage.taxonomy import FailureCategory

logger = logging.getLogger(__name__)

MAX_LLM_CALLS = 3
MAX_EVIDENCE_ITEMS = 6
MAX_AFFECTED_FILES = 10
# Tool turns are budgeted separately from answer attempts: looking things up must not
# eat the repair attempts. Three is what the headroom analysis calls for - one read for
# a file the log already named, or a search plus a read when it did not.
MAX_TOOL_CALLS = 3
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
    # "timeout" | "output" | "transport" - decides whether asking again can help.
    error_kind: str
    # Set once the evidence has been shrunk after a timeout, so we never grow it again.
    shrunk: bool
    # Attempts that tried to produce a diagnosis; tool turns are counted separately
    # so that looking things up cannot consume the repair budget.
    answer_calls: int
    action: dict[str, Any]
    tool_results: list[ToolResult]
    # Set when the model asked for something it had already asked for: gathering
    # stops, because repeating a request cannot return anything new.
    gathering_stuck: bool


def _error_kind(exc: LLMError) -> str:
    if isinstance(exc, LLMTimeout):
        return "timeout"
    if isinstance(exc, LLMOutputError):
        return "output"
    return "transport"


def build_graph(
    client: LLMClient,
    max_llm_calls: int = MAX_LLM_CALLS,
    *,
    toolbox: Toolbox | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
):
    """Compile the investigation graph for one LLM client.

    Without a `toolbox` this is the Phase 3 graph unchanged, so its numbers stay
    reproducible. With one, a gathering loop runs first and its results are appended
    to the evidence, inside the untrusted block.
    """

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

    def _call(
        state: InvestigationState,
        prompt: str,
        node: str,
        *,
        schema: dict[str, Any] = OUTPUT_SCHEMA,
        key: str = "proposal",
    ) -> InvestigationState:
        started = time.perf_counter()
        is_answer = key == "proposal"
        try:
            reply, usage = client.complete_json(system=SYSTEM_PROMPT, user=prompt, schema=schema)
        except LLMError as exc:
            kind = _error_kind(exc)
            failed: InvestigationState = {
                "error": str(exc),
                "error_kind": kind,
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": record(
                    state, node, error=str(exc), error_kind=kind, raw=exc.raw_output[:500]
                ),
            }
            if is_answer:
                failed["answer_calls"] = state.get("answer_calls", 0) + 1
            return failed
        done: InvestigationState = {
            key: reply,
            # Clear any earlier failure: without this a stale error survives a
            # successful retry and validate() reports it as a fresh problem.
            "error": "",
            "error_kind": "",
            "llm_calls": state.get("llm_calls", 0) + 1,
            "usage": state.get("usage", Usage(0, 0, 0)) + usage,
            "trace": record(
                state,
                node,
                seconds=round(time.perf_counter() - started, 1),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                answer=reply.get("failure_type") or reply.get("action"),
            ),
        }
        if is_answer:
            done["answer_calls"] = state.get("answer_calls", 0) + 1
        return done

    def _full_evidence(state: InvestigationState) -> str:
        """Base evidence plus anything the tools returned, all inside one block."""
        sections = [result.as_prompt_section() for result in state.get("tool_results", [])]
        return ev.with_sections(state["evidence"], sections)

    def decide(state: InvestigationState) -> InvestigationState:
        """Ask for one tool call, or for the investigation to move on.

        Deliberately not given the full evidence: see `evidence.brief`.
        """
        results = state.get("tool_results", [])
        shown = ev.packed_paths(state["evidence"]) | {
            str(r.args.get("path")) for r in results if r.ok and r.args.get("path")
        }
        prompt = build_decide_prompt(
            ev.brief(state["case"], shown),
            [result.as_prompt_section() for result in results],
            max_tool_calls - len(results),
        )
        return _call(state, prompt, "decide", schema=ACTION_SCHEMA, key="action")

    def act(state: InvestigationState) -> InvestigationState:
        """Run the requested tool. Refusals come back as readable results, not crashes."""
        assert toolbox is not None
        action = state.get("action") or {}
        name = str(action.get("action", ""))
        args = {
            key: action[key]
            for key in ("path", "pattern", "start_line", "end_line")
            if action.get(key) not in (None, "")
        }
        previous = state.get("tool_results", [])
        # Asking for exactly what was already asked for returns exactly what came back:
        # no new information, one more slow model call. Observed on the live model,
        # which requested read_file with no path three times in a row and spent 232 s
        # being refused for the same reason. Same lesson as the repair loop (BUG-007).
        repeated = any(call.name == name and call.args == args for call in previous)
        result = toolbox.run(name, args)
        return {
            "tool_results": [*previous, result],
            "gathering_stuck": repeated,
            "trace": record(state, "act", repeated=repeated, **result.as_trace()),
        }

    def propose(state: InvestigationState) -> InvestigationState:
        return _call(state, build_user_prompt(_full_evidence(state)), "propose")

    def repair(state: InvestigationState) -> InvestigationState:
        prompt = build_repair_prompt(_full_evidence(state), state.get("problems", []))
        return _call(state, prompt, "repair")

    def expand(state: InvestigationState) -> InvestigationState:
        """Answer was UNKNOWN: re-pack with a bigger budget and ask once more."""
        pack = ev.pack(state["case"], budget_chars=ev.EXPANDED_BUDGET_CHARS)
        return {
            "evidence": pack.text,
            "budget": ev.EXPANDED_BUDGET_CHARS,
            "trace": record(state, "expand_evidence", chars=len(pack.text)),
        }

    def shrink(state: InvestigationState) -> InvestigationState:
        """The call timed out: re-pack smaller and try once. Never grows again."""
        pack = ev.pack(state["case"], budget_chars=ev.SHRUNK_BUDGET_CHARS)
        return {
            "evidence": pack.text,
            "budget": ev.SHRUNK_BUDGET_CHARS,
            "shrunk": True,
            "error": "",
            "error_kind": "",
            "trace": record(state, "shrink_evidence", chars=len(pack.text)),
        }

    def validate(state: InvestigationState) -> InvestigationState:
        if state.get("error"):
            problem = _problem_for(state)
            return {
                "problems": [problem],
                "previous_problems": state.get("problems", []),
                # Recorded like the normal path: a trace that silently skips a node
                # when the call failed hides exactly the runs worth reading.
                "trace": record(state, "validate", problems=[problem]),
            }
        problems = _validate(state["case"], _full_evidence(state), state.get("proposal") or {})
        return {
            "problems": problems,
            "previous_problems": state.get("problems", []),
            "trace": record(state, "validate", problems=problems),
        }

    def finalize(state: InvestigationState) -> InvestigationState:
        proposal = state.get("proposal")
        if proposal is None:
            return {"trace": record(state, "finalize", ok=False)}
        diagnosis = _to_diagnosis(
            state["case"], _full_evidence(state), proposal, state.get("tool_results", [])
        )
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

    def route_decide(state: InvestigationState) -> str:
        """Gathering loop: another tool call, or move to the diagnosis."""
        if state.get("error"):
            return "propose"  # a failed decide is not worth a retry; answer with what we have
        action = (state.get("action") or {}).get("action")
        return "act" if action in TOOL_NAMES else "propose"

    def route_after_act(state: InvestigationState) -> str:
        """Only ask again if there is something left to ask for, and if asking helps.

        Asking "what next?" with a budget of zero costs a full model call to be told
        what we already know - ~30-60 s of local inference per case for nothing. And a
        model that just repeated a request will repeat it again.
        """
        if state.get("gathering_stuck"):
            return "propose"
        return "decide" if len(state.get("tool_results", [])) < max_tool_calls else "propose"

    def route(state: InvestigationState) -> str:
        proposal = state.get("proposal") or {}
        calls_left = state.get("answer_calls", 0) < max_llm_calls
        problems = state.get("problems") or []
        repeated = problems == state.get("previous_problems")

        kind = state.get("error_kind") or ""
        if kind:
            # The model did not answer this turn. Whether to ask again depends on why.
            if kind == "timeout" and calls_left and not state.get("shrunk"):
                return "shrink"  # too much prompt for this machine: send less
            if kind == "output" and calls_left and not repeated:
                return "repair"  # it is alive and can be told what was wrong
            if kind == "output" and calls_left and not state.get("shrunk"):
                # Truncated twice despite being told why. The schema's maxLength is
                # advisory - Ollama's constrained decoding does not enforce it - so
                # asking again in the same words will not help. A smaller prompt does
                # shorten the answer, so spend the last attempt on that instead.
                return "shrink"
            # Out of options. Fall back on an earlier good proposal if we have one.
            return "finalize" if proposal else END

        if not proposal:
            return END
        if problems:
            # Stop if the repair produced exactly the same complaints: the model is not
            # going to fix it, and each local call costs ~90-180 s.
            if repeated:
                return "finalize"
            # Retry while we can; otherwise finalize best-effort - the mapping step drops
            # ungrounded quotes and invented paths anyway.
            return "repair" if calls_left else "finalize"
        if (
            proposal.get("failure_type") == FailureCategory.UNKNOWN.value
            and calls_left
            and state.get("budget", 0) < ev.EXPANDED_BUDGET_CHARS
            # Never grow the prompt again on a machine that already timed out on it.
            and not state.get("shrunk")
        ):
            return "expand"
        return "finalize"

    graph = StateGraph(InvestigationState)
    graph.add_node("pack_evidence", pack_evidence)
    graph.add_node("propose", propose)
    graph.add_node("validate", validate)
    graph.add_node("repair", repair)
    graph.add_node("expand", expand)
    graph.add_node("shrink", shrink)
    graph.add_node("finalize", finalize)
    graph.set_entry_point("pack_evidence")
    if toolbox is None:
        graph.add_edge("pack_evidence", "propose")
    else:
        graph.add_node("decide", decide)
        graph.add_node("act", act)
        graph.add_edge("pack_evidence", "decide")
        graph.add_conditional_edges("decide", route_decide, {"act": "act", "propose": "propose"})
        graph.add_conditional_edges(
            "act", route_after_act, {"decide": "decide", "propose": "propose"}
        )
    graph.add_edge("propose", "validate")
    graph.add_edge("repair", "validate")
    graph.add_edge("expand", "propose")
    graph.add_edge("shrink", "propose")
    graph.add_conditional_edges(
        "validate",
        route,
        {
            "repair": "repair",
            "expand": "expand",
            "shrink": "shrink",
            "finalize": "finalize",
            END: END,
        },
    )
    graph.add_edge("finalize", END)
    return graph.compile()


# ----------------------------------------------------------------------- validation


def _problem_for(state: InvestigationState) -> str:
    """Turn a failed call into something the model can act on.

    Handing the raw exception back ("Unterminated string starting at char 276") tells
    the model nothing it can do differently; naming the cause does.
    """
    if state.get("error_kind") == "output":
        return (
            "your previous answer was not valid JSON - it was cut off in the middle of a "
            f"value. Keep every quote under {MAX_QUOTE_CHARS} characters, quote one line "
            "rather than a whole traceback, and send the complete JSON object."
        )
    return str(state.get("error") or "the model did not answer")


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


def _locate(
    case: CaseView, quote: str, tool_results: list[ToolResult] | None = None
) -> tuple[str, str] | None:
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
    # A file the agent fetched itself is evidence like any other - without this, every
    # quote from a tool read would be dropped as ungrounded and the tools would be
    # unable to contribute evidence at all.
    for result in tool_results or []:
        if result.ok and quote in result.content:
            path = result.args.get("path")
            return "source_file", str(path) if path else f"{result.name} result"
    return None


def _to_diagnosis(
    case: CaseView,
    evidence_text: str,
    proposal: dict[str, Any],
    tool_results: list[ToolResult] | None = None,
) -> Diagnosis:
    items: list[Evidence] = []
    for entry in proposal.get("evidence") or []:
        quote = str(entry.get("quote", "")).strip()
        located = _locate(case, quote, tool_results) if quote else None
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
