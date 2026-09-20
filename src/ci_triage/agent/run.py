"""Agent entry point used by the evaluation harness: CaseView -> Diagnosis."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ci_triage.agent.graph import MAX_LLM_CALLS, build_graph
from ci_triage.agent.tools import Toolbox
from ci_triage.diagnosis import Diagnosis
from ci_triage.llm import LLMClient, LLMError, OllamaClient, Usage
from ci_triage.miner.schema import CaseView

logger = logging.getLogger(__name__)

TRACE_DIR_ENV = "CI_TRIAGE_TRACE_DIR"
DEFAULT_REPOS_DIR = Path("data/repos")


class Investigator:
    """Wraps the compiled graph so the harness sees a plain callable per system.

    With `repos_dir` set the agent gets repository tools, and the graph is rebuilt per
    case: a toolbox is pinned to one repository and one commit, which is what stops a
    tool from ever reading the commit that fixed the failure.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        max_llm_calls: int = MAX_LLM_CALLS,
        trace_dir: Path | None = None,
        repos_dir: Path | None = None,
    ):
        self.client = client
        self.repos_dir = repos_dir
        # The version is part of the name so a report can never be read as an earlier
        # version's numbers. v2: failed calls are recovered instead of losing the case.
        model = client.name.replace(":", "-").replace("/", "-")
        self.name = f"{'agent_tools_v1' if repos_dir else 'agent_v2'}_{model}"
        self.trace_dir = trace_dir
        self.max_llm_calls = max_llm_calls
        self._graph = None if repos_dir else build_graph(client, max_llm_calls=max_llm_calls)
        self.last_usage = Usage(0, 0, 0)
        self.last_tool_calls = 0

    def _graph_for(self, case: CaseView):
        if self._graph is not None:
            return self._graph
        assert self.repos_dir is not None
        toolbox = Toolbox.for_case(
            self.repos_dir, case.repo.full_name, case.input.failed_commit.sha
        )
        return build_graph(self.client, max_llm_calls=self.max_llm_calls, toolbox=toolbox)

    def analyze(self, case: CaseView) -> Diagnosis:
        final = self._graph_for(case).invoke({"case": case})
        self.last_tool_calls = len(final.get("tool_results", []))
        # Token counts only accumulate on success, but a call that timed out or came
        # back truncated still occupied the model. Report the calls actually made,
        # otherwise the cost column quietly under-reports exactly the slow cases.
        usage = final.get("usage", Usage(0, 0, 0))
        self.last_usage = Usage(
            usage.input_tokens, usage.output_tokens, final.get("llm_calls", usage.calls)
        )
        self._write_trace(case, final)
        diagnosis = final.get("diagnosis")
        if diagnosis is None:
            raise LLMError(final.get("error") or "agent produced no diagnosis")
        return diagnosis

    def _write_trace(self, case: CaseView, final: dict) -> None:
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        usage = final.get("usage", Usage(0, 0, 0))
        trace = {
            "case_id": case.case_id,
            "system": self.name,
            "model": self.client.name,
            "llm_calls": final.get("llm_calls", 0),
            "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
            "problems": final.get("problems", []),
            "error": final.get("error"),
            "steps": final.get("trace", []),
            "raw_proposal": final.get("proposal"),
            "tool_calls": [result.as_trace() for result in final.get("tool_results", [])],
        }
        path = self.trace_dir / f"{case.case_id}.json"
        path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")


def from_env(trace_dir: Path | None = None, *, tools: bool = False) -> Investigator:
    """Build the default investigator: local Ollama, trace dir from env if not given."""
    if trace_dir is None and os.environ.get(TRACE_DIR_ENV):
        trace_dir = Path(os.environ[TRACE_DIR_ENV])
    repos_dir = Path(os.environ.get("CI_TRIAGE_REPOS_DIR", DEFAULT_REPOS_DIR)) if tools else None
    return Investigator(OllamaClient.from_env(), trace_dir=trace_dir, repos_dir=repos_dir)
