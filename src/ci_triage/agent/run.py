"""Agent entry point used by the evaluation harness: CaseView -> Diagnosis."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ci_triage.agent.graph import MAX_LLM_CALLS, build_graph
from ci_triage.diagnosis import Diagnosis
from ci_triage.llm import LLMClient, LLMError, OllamaClient, Usage
from ci_triage.miner.schema import CaseView

logger = logging.getLogger(__name__)

TRACE_DIR_ENV = "CI_TRIAGE_TRACE_DIR"


class Investigator:
    """Wraps the compiled graph so the harness sees a plain callable per system."""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_llm_calls: int = MAX_LLM_CALLS,
        trace_dir: Path | None = None,
    ):
        self.client = client
        self.name = f"agent_v1_{client.name.replace(':', '-').replace('/', '-')}"
        self.trace_dir = trace_dir
        self._graph = build_graph(client, max_llm_calls=max_llm_calls)
        self.last_usage = Usage(0, 0, 0)

    def analyze(self, case: CaseView) -> Diagnosis:
        final = self._graph.invoke({"case": case})
        self.last_usage = final.get("usage", Usage(0, 0, 0))
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
        }
        path = self.trace_dir / f"{case.case_id}.json"
        path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")


def from_env(trace_dir: Path | None = None) -> Investigator:
    """Build the default investigator: local Ollama, trace dir from env if not given."""
    if trace_dir is None and os.environ.get(TRACE_DIR_ENV):
        trace_dir = Path(os.environ[TRACE_DIR_ENV])
    return Investigator(OllamaClient.from_env(), trace_dir=trace_dir)
