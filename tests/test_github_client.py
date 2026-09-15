import logging
import time

import httpx
import pytest

from ci_triage.github_client import GitHubClient, GitHubError, NotFoundError


def make_client(handler, tmp_path=None, **kwargs):
    sleeps: list[float] = []
    client = GitHubClient(
        "test-token",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        **kwargs,
    )
    return client, sleeps


def test_paginate_stops_at_short_page():
    def handler(request):
        page = int(request.url.params["page"])
        count = 100 if page == 1 else 7
        return httpx.Response(200, json={"jobs": [{"id": i} for i in range(count)]})

    client, _ = make_client(handler)
    items = list(client.paginate("/repos/o/r/actions/runs/1/jobs", "jobs"))
    assert len(items) == 107
    assert client.request_count == 2


def test_server_error_is_retried_with_backoff():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(502)
        return httpx.Response(200, json={"ok": True})

    client, sleeps = make_client(handler)
    assert client.get_json("/x") == {"ok": True}
    assert sleeps == [1, 2]


def test_primary_rate_limit_waits_until_reset():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            reset = str(int(time.time()) + 30)
            return httpx.Response(
                403,
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset},
                json={"message": "API rate limit exceeded"},
            )
        return httpx.Response(200, json={"ok": True})

    client, sleeps = make_client(handler)
    assert client.get_json("/x") == {"ok": True}
    assert len(sleeps) == 1 and 25 <= sleeps[0] <= 32


def test_plain_403_is_not_retried():
    client, sleeps = make_client(lambda r: httpx.Response(403, json={"message": "forbidden"}))
    with pytest.raises(GitHubError) as err:
        client.get_json("/x")
    assert err.value.status == 403
    assert sleeps == []


def test_404_raises_not_found_without_retry():
    client, sleeps = make_client(lambda r: httpx.Response(404, json={"message": "Not Found"}))
    with pytest.raises(NotFoundError):
        client.get_json("/x")
    assert client.request_count == 1
    assert sleeps == []


def test_job_log_follows_redirect_without_leaking_token():
    seen_blob_headers = {}

    def handler(request):
        if request.url.host == "api.github.com":
            assert request.headers["authorization"] == "Bearer test-token"
            return httpx.Response(302, headers={"location": "https://blob.example/signed?sig=s"})
        seen_blob_headers.update(request.headers)
        return httpx.Response(200, text="2026-01-01T00:00:00.0Z hello\n")

    client, _ = make_client(handler)
    assert client.get_job_log("o/r", 5) == "2026-01-01T00:00:00.0Z hello\n"
    assert "authorization" not in seen_blob_headers


def test_signed_log_url_never_reaches_logs(caplog):
    from ci_triage.miner.__main__ import configure_logging

    configure_logging()
    caplog.set_level(logging.DEBUG)

    def handler(request):
        if request.url.host == "api.github.com":
            return httpx.Response(302, headers={"location": "https://blob.example/x?sig=SECRET"})
        return httpx.Response(200, text="log")

    client, _ = make_client(handler)
    client.get_job_log("o/r", 5)
    assert "signed blob redirect" in caplog.text  # our sanitized request line exists
    assert "SECRET" not in caplog.text
    assert "blob.example" not in caplog.text


def test_expired_job_log_returns_none():
    client, _ = make_client(lambda r: httpx.Response(410, json={"message": "Gone"}))
    assert client.get_job_log("o/r", 5) is None


def test_disk_cache_avoids_repeat_requests(tmp_path):
    client, _ = make_client(lambda r: httpx.Response(200, json={"v": 1}), tmp_path)
    assert client.get_json("/x", {"a": 1}) == {"v": 1}
    assert client.get_json("/x", {"a": 1}) == {"v": 1}
    assert client.request_count == 1
    assert client.cache_hits == 1


def test_from_env_requires_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        GitHubClient.from_env()
