"""The LLM client's failure mapping.

The agent decides whether to retry from the *type* of the failure, so these
mappings are load-bearing: a timeout misfiled as a plain transport error means
the case is abandoned instead of retried with a smaller prompt, which is exactly
what lost one case in the first held-out run.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ci_triage.llm import (
    DEFAULT_MAX_TOKENS,
    LLMError,
    LLMOutputError,
    LLMTimeout,
    OllamaClient,
)

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def client_for(handler) -> OllamaClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    return OllamaClient(model="test-model", client=http)


def ask(client: OllamaClient):
    return client.complete_json(system="s", user="u", schema=SCHEMA)


def test_successful_call_returns_parsed_json_and_usage():
    def handler(request):
        payload = json.loads(request.content)
        assert payload["format"] == SCHEMA  # constrained decoding, not a polite request
        assert payload["options"]["num_predict"] == DEFAULT_MAX_TOKENS
        return httpx.Response(
            200,
            json={
                "message": {"content": '{"ok": true}'},
                "prompt_eval_count": 11,
                "eval_count": 3,
            },
        )

    parsed, usage = ask(client_for(handler))

    assert parsed == {"ok": True}
    assert (usage.input_tokens, usage.output_tokens, usage.calls) == (11, 3, 1)


def test_timeout_is_reported_as_a_timeout_not_a_generic_transport_error():
    def handler(request):
        raise httpx.ReadTimeout("read timed out", request=request)

    with pytest.raises(LLMTimeout):
        ask(client_for(handler))


def test_unreachable_server_is_a_plain_transport_error():
    """Must NOT be an LLMTimeout: retrying smaller cannot revive a dead server."""

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LLMError) as caught:
        ask(client_for(handler))
    assert not isinstance(caught.value, LLMTimeout)
    assert not isinstance(caught.value, LLMOutputError)


def test_truncated_json_is_an_output_error_and_keeps_what_arrived():
    """The starlette failure: a long quote ran into the token cap mid-string."""
    cut = '{"failure_type": "TEST_FAILURE", "root_cause": "a very long tracebac'

    def handler(request):
        return httpx.Response(200, json={"message": {"content": cut}})

    with pytest.raises(LLMOutputError) as caught:
        ask(client_for(handler))
    assert caught.value.raw_output == cut  # kept for the trace, so it is debuggable


def test_json_that_is_not_an_object_is_an_output_error():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "[1, 2, 3]"}})

    with pytest.raises(LLMOutputError):
        ask(client_for(handler))


def test_http_error_status_is_not_retryable_as_output_or_timeout():
    def handler(request):
        return httpx.Response(500, text="model not loaded")

    with pytest.raises(LLMError) as caught:
        ask(client_for(handler))
    assert type(caught.value) is LLMError
    assert "500" in str(caught.value)


def test_timeout_is_configurable_from_the_environment(monkeypatch):
    monkeypatch.setenv("CI_TRIAGE_TIMEOUT", "42")
    client = OllamaClient.from_env()
    assert client._http.timeout.read == 42.0
