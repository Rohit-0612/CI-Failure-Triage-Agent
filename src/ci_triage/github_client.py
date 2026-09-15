"""Thin, explicit GitHub REST client.

Design goals:
- Every HTTP call goes through one place (`_request`) so retries, rate limits and
  request counting are visible and testable.
- Responses are cached on disk so re-running the miner does not re-spend API quota.
- Nothing sensitive is logged: no token, and no signed log-download URLs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

API_URL = "https://api.github.com"
MAX_RATE_LIMIT_WAIT_SECONDS = 15 * 60


class GitHubError(Exception):
    def __init__(self, status: int, path: str, message: str = ""):
        super().__init__(f"GitHub API {status} for {path}: {message}".strip())
        self.status = status
        self.path = path


class NotFoundError(GitHubError):
    """404/410: resource missing or expired (e.g. logs past retention). Never retried."""


class GitHubClient:
    def __init__(
        self,
        token: str | None,
        cache_dir: Path | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 5,
        timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ci-triage-miner",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # Redirects are handled manually so the auth header is never sent to the
        # signed blob-storage URL that log downloads redirect to.
        self._http = httpx.Client(
            base_url=API_URL,
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )
        self._blob_http = httpx.Client(timeout=timeout, transport=transport)
        self._cache_dir = cache_dir
        self._max_retries = max_retries
        self._sleep = sleep
        self.request_count = 0
        self.cache_hits = 0

    @classmethod
    def from_env(cls, cache_dir: Path | None = None) -> GitHubClient:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN is not set. The Actions logs API requires authentication; "
                "copy .env.example to .env and set a read-only token."
            )
        return cls(token, cache_dir)

    def close(self) -> None:
        self._http.close()
        self._blob_http.close()

    # ------------------------------------------------------------------ public API

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        cache_key = self._cache_key("json", path, params)
        cached = self._cache_read(cache_key)
        if cached is not None:
            return json.loads(cached)
        response = self._request(self._http, path, params)
        self._cache_write(cache_key, response.text)
        return response.json()

    def paginate(
        self,
        path: str,
        item_key: str,
        params: dict[str, Any] | None = None,
        max_pages: int = 10,
    ) -> Iterator[dict[str, Any]]:
        """Yield items from a list endpoint that wraps results in `item_key`.

        Stops at the first short page, so no Link-header parsing is needed.
        """
        per_page = 100
        for page in range(1, max_pages + 1):
            body = self.get_json(path, {**(params or {}), "per_page": per_page, "page": page})
            items = body.get(item_key, [])
            yield from items
            if len(items) < per_page:
                return

    def get_job_log(self, repo: str, job_id: int) -> str | None:
        """Return the plain-text log for a job, or None if it is missing/expired."""
        path = f"/repos/{repo}/actions/jobs/{job_id}/logs"
        cache_key = self._cache_key("log", path, None)
        cached = self._cache_read(cache_key)
        if cached is not None:
            return cached
        try:
            redirect = self._request(self._http, path, None, allow_redirect=True)
            location = redirect.headers.get("location")
            if redirect.status_code == 200:
                text = redirect.text
            elif location:
                shown = f"{path} (signed blob redirect)"
                text = self._request(self._blob_http, location, None, log_path=shown).text
            else:
                raise GitHubError(redirect.status_code, path, "log redirect without location")
        except NotFoundError:
            logger.info("log unavailable (expired or missing) for %s", path)
            return None
        self._cache_write(cache_key, text)
        return text

    # ------------------------------------------------------------------ internals

    def _request(
        self,
        http: httpx.Client,
        url: str,
        params: dict[str, Any] | None,
        *,
        allow_redirect: bool = False,
        log_path: str | None = None,
    ) -> httpx.Response:
        """GET with retries for transient errors and rate limits.

        `log_path` replaces `url` in log lines and errors; used for signed URLs.
        """
        shown = log_path or url
        for attempt in range(self._max_retries + 1):
            started = time.monotonic()
            try:
                self.request_count += 1
                response = http.get(url, params=params)
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise GitHubError(0, shown, f"transport error: {exc}") from exc
                self._backoff(attempt, shown, "transport error")
                continue

            status = response.status_code
            # Our own request log: safe path only (never the signed URL or query secrets).
            logger.info(
                "GET %s -> %d in %.1fs (attempt %d)",
                shown,
                status,
                time.monotonic() - started,
                attempt + 1,
            )
            if status == 200 or (allow_redirect and status in (301, 302, 307)):
                return response
            if status in (404, 410):
                raise NotFoundError(status, shown)
            if status in (403, 429):
                wait = self._rate_limit_wait(response)
                if wait is None:
                    raise GitHubError(status, shown, _error_message(response))
                if attempt == self._max_retries:
                    raise GitHubError(status, shown, "rate limited; retries exhausted")
                logger.warning("rate limited on %s, sleeping %.0fs", shown, wait)
                self._sleep(wait)
                continue
            if status >= 500:
                if attempt == self._max_retries:
                    raise GitHubError(status, shown, "server error; retries exhausted")
                self._backoff(attempt, shown, f"HTTP {status}")
                continue
            raise GitHubError(status, shown, _error_message(response))
        raise AssertionError("unreachable")

    def _rate_limit_wait(self, response: httpx.Response) -> float | None:
        """Seconds to wait for a rate-limited response, or None if it is a real 403."""
        headers = response.headers
        if "retry-after" in headers:
            return float(headers["retry-after"])
        if headers.get("x-ratelimit-remaining") == "0" and "x-ratelimit-reset" in headers:
            wait = max(0.0, float(headers["x-ratelimit-reset"]) - time.time()) + 1
            if wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                raise GitHubError(response.status_code, "", f"rate limit resets in {wait:.0f}s")
            return wait
        if "secondary rate limit" in _error_message(response).lower():
            return 60.0
        return None

    def _backoff(self, attempt: int, shown: str, reason: str) -> None:
        wait = 2**attempt
        logger.warning("%s on %s, retry %d in %ds", reason, shown, attempt + 1, wait)
        self._sleep(wait)

    def _cache_key(self, kind: str, path: str, params: dict[str, Any] | None) -> str:
        raw = json.dumps([kind, path, sorted((params or {}).items())], default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_read(self, key: str) -> str | None:
        if self._cache_dir is None:
            return None
        file = self._cache_dir / key[:2] / key
        if file.exists():
            self.cache_hits += 1
            return file.read_text(encoding="utf-8")
        return None

    def _cache_write(self, key: str, text: str) -> None:
        if self._cache_dir is None:
            return
        file = self._cache_dir / key[:2] / key
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("message", ""))
    except ValueError:
        return ""
