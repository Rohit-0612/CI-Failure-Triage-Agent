"""The repository tools an investigating agent may call, and their limits.

Why tools exist here at all is measured, not assumed. `evaluation headroom` splits
localization misses by the information the model held: on the dev split 16.7% of
cases named the right file in the log but never received its content (the packer
sends `evidence.MAX_FILES` files and has to guess which), and another 11.9% never
saw the path at all. Those two buckets are what `read_file` and `search_code` are
for; nothing here helps the 4.8% where the model had the file and chose otherwise.

Three properties matter more than the tools themselves:

1. **Every tool is pinned to the failed commit, and the ref is not a parameter.**
   The model cannot ask for a branch, a tag, or another commit, because there is no
   argument for one. This is not tidiness: the clone also contains the commit that
   fixed the failure, and one tool call that reached it would leak the answer into
   the input and quietly destroy the benchmark.

2. **Tool arguments are untrusted.** They are written by a model that has just read
   attacker-controlled repository text, so paths are validated the way a web server
   validates them, and searches are fixed-string.

3. **Tool output is untrusted too.** It is repository text arriving through a channel
   the model chose, so it goes through the same `evidence.sanitize()` as everything
   else and is capped. Phase 3 decided what the model would read; from here the model
   decides, which is a new way in for injected text.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ci_triage.agent.evidence import sanitize
from ci_triage.git_local import GitError, GitRepo

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 2_500
# A file read gets more room than a listing or a search hit list: 2_500 chars is about
# 60 lines, and a 500-line source file read 60 lines at a time is not worth a model call.
MAX_READ_CHARS = 6_000
MAX_FILE_LINES = 200
MAX_LISTED_FILES = 60
MAX_PATH_LENGTH = 300
MIN_PATTERN_LENGTH = 3

# Segments that must never appear in a path a model asked for.
_FORBIDDEN_SEGMENTS = {"..", ".git"}
_WINDOWS_DRIVE = re.compile(r"^[a-zA-Z]:")

TOOL_NAMES = ("read_file", "search_code", "list_files")

TOOL_DESCRIPTIONS = {
    "read_file": (
        'read_file - read a source file as it was at the failed commit. Set "path" to '
        'the file, for example "src/pkg/thing.py", and "pattern" to "". Long files come '
        'back a window at a time; set "start_line" to read from a particular line, for '
        "example the line number search_code reported. "
        "Use this when the log names a file you have not been shown."
    ),
    "search_code": (
        "search_code - find where a piece of text appears in the repository at the "
        'failed commit. Set "pattern" to the plain text to look for (not a regular '
        'expression), and "path" to a directory to search in, or "" for the whole '
        "repository. Use this to find where a symbol from the error comes from."
    ),
    "list_files": (
        'list_files - list file paths at the failed commit. Set "path" to a directory, '
        'or "" for the whole repository, and "pattern" to "". Use this to find out how '
        "the project is laid out."
    ),
}


class ToolError(Exception):
    """A tool call that must be refused. The message is shown to the model."""


@dataclass(frozen=True)
class ToolResult:
    name: str
    args: dict[str, Any]
    ok: bool
    content: str
    seconds: float = 0.0

    def as_prompt_section(self) -> str:
        shown = {k: v for k, v in self.args.items() if v not in (None, "")}
        header = f"## result of {self.name}({shown})"
        return f"{header}\n{self.content}"

    def as_trace(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "args": self.args,
            "ok": self.ok,
            "chars": len(self.content),
            "seconds": round(self.seconds, 1),
        }


def safe_path(raw: Any, *, allow_empty: bool = False) -> str:
    """Validate a path a model supplied, or raise ToolError explaining why not.

    Refuses what a path-traversal check refuses - absolute paths, `..`, `.git` - and
    normalises Windows separators, because logs print those and the model copies them.
    """
    if raw in (None, ""):
        if allow_empty:
            return ""
        raise ToolError("a 'path' argument is required")
    if not isinstance(raw, str):
        raise ToolError("'path' must be a string")

    path = raw.strip().replace("\\", "/").strip("/")
    if not path:
        if allow_empty:
            return ""
        raise ToolError("'path' is empty")
    if len(path) > MAX_PATH_LENGTH:
        raise ToolError(f"'path' is longer than {MAX_PATH_LENGTH} characters")
    if raw.strip().startswith("/") or _WINDOWS_DRIVE.match(raw.strip()):
        raise ToolError("'path' must be relative to the repository root")
    segments = path.split("/")
    if any(segment in _FORBIDDEN_SEGMENTS for segment in segments):
        raise ToolError("'path' may not contain '..' or '.git'")
    if any(segment in ("", ".") for segment in segments):
        raise ToolError("'path' contains an empty or '.' segment")
    return path


def _start_line(args: dict[str, Any], total: int) -> int:
    """Where to start reading. Anything unusable starts at the top of the file.

    How far the window reaches is decided by the character budget, not by the model:
    asking it for both ends gives it two more fields to get wrong for no benefit.
    """
    value = args.get("start_line")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 1
    return max(1, min(int(value), max(total, 1)))


@dataclass
class Toolbox:
    """Runs allowlisted tools against one repository, pinned to one commit."""

    repo: GitRepo
    sha: str
    max_result_chars: int = MAX_RESULT_CHARS
    max_read_chars: int = MAX_READ_CHARS
    calls: list[ToolResult] = field(default_factory=list)
    _hydrated: bool = False

    @classmethod
    def for_case(cls, repos_dir: Path, full_name: str, sha: str) -> Toolbox:
        return cls(repo=GitRepo.open_or_init(repos_dir, full_name), sha=sha)

    def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Dispatch one call. A refusal is a result the model can read, not a crash:
        it can correct a bad path itself, and losing the case would cost far more."""
        started = time.perf_counter()
        try:
            if name not in TOOL_NAMES:
                raise ToolError(f"unknown tool {name!r}; available tools: {', '.join(TOOL_NAMES)}")
            body, ok = getattr(self, f"_{name}")(args), True
        except ToolError as exc:
            body, ok = f"error: {exc}", False
        except GitError as exc:
            # The repository may be missing or offline; the investigation continues
            # on the evidence already gathered.
            logger.info("tool %s failed: %s", name, exc)
            body, ok = f"error: repository is not available ({exc})", False

        result = ToolResult(
            name=name,
            args=args,
            ok=ok,
            content=self._bound(body, name),
            seconds=time.perf_counter() - started,
        )
        self.calls.append(result)
        return result

    def _bound(self, text: str, tool: str) -> str:
        """Sanitize and cap. Tool output is repository text like any other evidence.

        A file read is allowed more room than a listing: read_file already cut its own
        window to MAX_READ_CHARS, so a smaller cap here would silently re-cut it and
        contradict the line range it just reported.
        """
        limit = self.max_read_chars if tool == "read_file" else self.max_result_chars
        clean = sanitize(text)
        if len(clean) > limit:
            return clean[:limit] + "\n[... result truncated ...]"
        return clean

    def _hydrate(self) -> None:
        """Pull the commit's blobs once, so reads and searches stay local."""
        if not self._hydrated:
            self.repo.hydrate(self.sha)
            self._hydrated = True

    # ------------------------------------------------------------------ the tools

    def _read_file(self, args: dict[str, Any]) -> str:
        """A window of the file, honestly labelled.

        The window is cut by characters, not by the requested line count, and the header
        names the lines actually returned. The first version promised "lines 1-200 of
        500" and then let the char cap silently drop everything past line 62, so the
        model was told it had read a file it had barely opened.
        """
        path = safe_path(args.get("path"))
        self._hydrate()
        found = self.repo.read_file(self.sha, path, max_chars=400_000)
        if found is None:
            raise ToolError(
                f"{path!r} does not exist at the failed commit (or is not a text file). "
                "Use list_files or search_code to find the right path."
            )
        lines = found[0].splitlines()
        total = len(lines)
        start = _start_line(args, total)

        shown, used = [], 0
        for number in range(start, min(total, start + MAX_FILE_LINES) + 1):
            row = f"{number:>5}  {lines[number - 1]}"
            if used + len(row) + 1 > MAX_READ_CHARS and shown:
                break
            shown.append(row)
            used += len(row) + 1

        last = start + len(shown) - 1
        header = f"{path} lines {start}-{last} of {total}"
        if last < total:
            header += (
                f'; {total - last} more lines follow - read again with "start_line": {last + 1}'
            )
        return header + "\n" + "\n".join(shown)

    def _search_code(self, args: dict[str, Any]) -> str:
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or len(pattern.strip()) < MIN_PATTERN_LENGTH:
            raise ToolError(
                f"'pattern' must be plain text of at least {MIN_PATTERN_LENGTH} characters"
            )
        directory = safe_path(args.get("path"), allow_empty=True)
        self._hydrate()
        glob = f"{directory}/*" if directory else None
        rows = self.repo.grep(self.sha, pattern.strip(), path_glob=glob)
        if not rows:
            where = f" under {directory}" if directory else ""
            return f"no matches for {pattern.strip()!r}{where} at the failed commit"
        return f"{len(rows)} match(es) for {pattern.strip()!r}:\n" + "\n".join(rows)

    def _list_files(self, args: dict[str, Any]) -> str:
        directory = safe_path(args.get("path"), allow_empty=True)
        paths = self.repo.list_files(self.sha)
        if directory:
            prefix = f"{directory}/"
            paths = [p for p in paths if p.startswith(prefix)]
            if not paths:
                raise ToolError(f"no files under {directory!r} at the failed commit")
        shown = paths[:MAX_LISTED_FILES]
        header = f"{len(paths)} file(s)" + (f" under {directory}" if directory else "")
        if len(paths) > len(shown):
            header += f", first {len(shown)} shown"
        return f"{header}:\n" + "\n".join(shown)
