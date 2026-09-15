"""Local git access through a partial (blobless, shallow) bare clone per repository.

Why local git instead of the GitHub compare API: on PR branches the "fixed" commit
often replaces the failing one (amend / force-push). The compare API then diffs from
the merge base and reports the whole PR, while `git diff red green` shows the real
change between the two tested trees.

Only fixed argument lists are passed to git (no shell), and SHAs are validated first.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_FETCH_DEPTH = 50
GIT_TIMEOUT_SECONDS = 180

Relation = Literal["identical", "ahead", "amend", "diverged", "unknown"]


class GitError(Exception):
    pass


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    parents: tuple[str, ...]
    author_date: str
    message: str


def validate_sha(sha: str) -> str:
    if not SHA_RE.match(sha):
        raise ValueError(f"not a full 40-char lowercase SHA: {sha!r}")
    return sha


class GitRepo:
    def __init__(self, path: Path, remote_url: str):
        self.path = path
        self.remote_url = remote_url

    @classmethod
    def open_or_init(
        cls, repos_dir: Path, full_name: str, remote_url: str | None = None
    ) -> GitRepo:
        path = repos_dir / full_name.replace("/", "__")
        repo = cls(path, remote_url or f"https://github.com/{full_name}.git")
        if not (path / "HEAD").exists():
            path.mkdir(parents=True, exist_ok=True)
            repo._git("init", "--bare", "--quiet")
            # A named remote makes it the promisor for lazily-fetched blobs.
            repo._git("remote", "add", "origin", repo.remote_url)
        return repo

    # ------------------------------------------------------------------ fetching

    def fetch(self, shas: list[str], depth: int = DEFAULT_FETCH_DEPTH) -> set[str]:
        """Fetch commits by SHA (blobs are downloaded lazily). Returns SHAs now available.

        A batch fetch fails entirely if one SHA is unreachable (e.g. force-pushed away),
        so on failure each SHA is retried alone.
        """
        wanted = [validate_sha(s) for s in dict.fromkeys(shas)]
        missing = [s for s in wanted if not self.has_commit(s)]
        if missing:
            try:
                self._fetch(missing, depth)
            except GitError:
                for sha in missing:
                    try:
                        self._fetch([sha], depth)
                    except GitError as exc:
                        logger.info("could not fetch %s: %s", sha[:10], exc)
        return {s for s in wanted if self.has_commit(s)}

    def _fetch(self, shas: list[str], depth: int) -> None:
        self._git(
            "fetch",
            "--quiet",
            "--no-tags",
            "--filter=blob:none",
            f"--depth={depth}",
            "origin",
            *shas,
        )

    def has_commit(self, sha: str) -> bool:
        try:
            self._git("cat-file", "-e", f"{sha}^{{commit}}")
            return True
        except GitError:
            return False

    # ------------------------------------------------------------------ queries

    def commit_info(self, sha: str) -> CommitInfo:
        out = self._git("show", "-s", "--format=%H%x00%P%x00%aI%x00%B", validate_sha(sha))
        full, parents, date, message = out.split("\x00", 3)
        return CommitInfo(full, tuple(parents.split()), date, message.strip())

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        try:
            self._git("merge-base", "--is-ancestor", ancestor, descendant)
            return True
        except GitError:
            return False

    def relation(self, red: str, green: str) -> Relation:
        """How the first green commit relates to the last red commit.

        identical: same commit was re-tested (rerun / re-trigger) -> not a code fix.
        ahead:     green descends from red -> the commits in red..green are the fix window.
        amend:     same parents, different commit -> red was replaced (amend/force-push).
        diverged:  anything else (rebase onto a new base, unrelated history).
        unknown:   history not deep enough to decide.
        """
        if red == green:
            return "identical"
        if self.is_ancestor(red, green):
            return "ahead"
        red_parents = self.commit_info(red).parents
        green_parents = self.commit_info(green).parents
        if red_parents and red_parents == green_parents:
            return "amend"
        if not red_parents or not green_parents:
            return "unknown"  # shallow boundary: parents were cut off by --depth
        return "diverged"

    def commits_between(self, base: str, head: str) -> list[CommitInfo]:
        """Commits reachable from head but not base, oldest first."""
        out = self._git("rev-list", "--reverse", f"{base}..{head}")
        return [self.commit_info(sha) for sha in out.split()]

    def diff(self, base: str, head: str, max_chars: int) -> tuple[str, bool]:
        text = self._git("diff", "--no-color", "--no-ext-diff", base, head)
        if len(text) > max_chars:
            return text[:max_chars], True
        return text, False

    def changed_files(self, base: str, head: str) -> list[str]:
        out = self._git("diff", "--name-only", "--no-renames", base, head)
        return [line for line in out.splitlines() if line]

    def hunk_headers(self, base: str, head: str) -> list[str]:
        """`path @@ -a,b +c,d @@ context` lines: a compact, comparable fix location."""
        out = self._git("diff", "--no-color", "-U0", base, head)
        hunks, current = [], ""
        for line in out.splitlines():
            if line.startswith("+++ b/"):
                current = line[6:]
            elif line.startswith("@@") and current:
                hunks.append(f"{current} {line}")
        return hunks

    def list_files(self, sha: str) -> list[str]:
        """All file paths at a commit (trees are present in a blobless clone)."""
        out = self._git("ls-tree", "-r", "--name-only", validate_sha(sha))
        return [line for line in out.splitlines() if line]

    def read_file(self, sha: str, path: str, max_chars: int) -> tuple[str, bool] | None:
        """File content at a commit, or None if absent/binary."""
        try:
            raw = self._git_bytes("show", f"{validate_sha(sha)}:{path}")
        except GitError:
            return None
        if b"\x00" in raw[:8000]:
            return None
        text = raw.decode("utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars], True
        return text, False

    # ------------------------------------------------------------------ plumbing

    def _git(self, *args: str) -> str:
        return self._git_bytes(*args).decode("utf-8", errors="replace")

    def _git_bytes(self, *args: str) -> bytes:
        cmd = ["git", "-C", str(self.path), *args]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"git {args[0]} timed out") from exc
        if proc.returncode != 0:
            raise GitError(proc.stderr.decode("utf-8", errors="replace").strip()[:500])
        return proc.stdout
