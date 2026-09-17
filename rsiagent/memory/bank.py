"""The persistent memory bank.

Memory is "a collection of actor-authored files, with no required schema, file
count, or length" (Appendix A.2).  This module implements the ownership rules
that surround it, which are the part that actually matters:

* **Only completed actor learning updates are promoted.**  Work-phase memory
  edits are discarded before learning begins.
* **Branches are isolated.**  Every project in a wave opens its own session on
  the *same* pre-wave snapshot and cannot see sibling work.
* **Updates are serial and cumulative.**  Once a wave's verdicts are grounded,
  sessions commit one at a time; each commit sees the previous ones.
* **The curriculum agent gets a disposable copy.**  Its local edits are never
  synchronized back.
* **Frozen memory is read-only**, and its tree hash is recorded so that Phase 3
  can prove the evaluation ran against exactly the memory exploration produced.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..errors import MemoryIntegrityError

MEMORY_INDEX = "INDEX.md"
"""A conventional entry point, not a required one.

The paper's case studies show the actor maintaining an index of its own design
(Appendix E.1: "the index itself also learns from the branch structure"), so
this name is a seed, never a schema.
"""

_IGNORED = {"__pycache__", ".DS_Store", ".git"}


@dataclass(frozen=True)
class MemoryStats:
    """File count, byte total, and tree hash — the paper's Figure A1 axes."""

    file_count: int
    total_bytes: int
    tree_hash: str

    def as_dict(self) -> dict[str, object]:
        return {
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "tree_hash": self.tree_hash,
        }


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not any(part in _IGNORED for part in p.parts)
    )


def compute_stats(root: Path) -> MemoryStats:
    """Deterministic file count, byte total, and tree hash for a directory."""
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in _iter_files(root):
        data = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
        digest.update(b"\n")
        count += 1
        total += len(data)
    return MemoryStats(file_count=count, total_bytes=total, tree_hash=digest.hexdigest())


def render_memory(root: Path, *, limit_bytes: int = 120_000) -> str:
    """Render a memory tree for a prompt, newest-relevant-first is *not* assumed.

    The paper gives the actor raw ownership of representation and retrieval; a
    plain concatenation is the honest default, and an actor that wants an index
    writes one.
    """
    files = _iter_files(root)
    if not files:
        return "(memory is empty)"

    chunks: list[str] = []
    used = 0
    for path in files:
        rel = path.relative_to(root).as_posix()
        body = path.read_text(encoding="utf-8", errors="replace")
        chunk = f"### {rel}\n{body.rstrip()}\n"
        if used + len(chunk) > limit_bytes:
            chunks.append(f"### {rel}\n[omitted: memory budget reached]\n")
            continue
        chunks.append(chunk)
        used += len(chunk)
    return "\n".join(chunks)


class MemorySnapshot:
    """An immutable, materialized copy of the bank."""

    def __init__(self, path: Path, stats: MemoryStats, *, label: str = "") -> None:
        self.path = path
        self.stats = stats
        self.label = label

    @property
    def tree_hash(self) -> str:
        return self.stats.tree_hash

    @property
    def is_empty(self) -> bool:
        return self.stats.file_count == 0

    def read_all(self) -> dict[str, str]:
        return {
            p.relative_to(self.path).as_posix(): p.read_text(encoding="utf-8", errors="replace")
            for p in _iter_files(self.path)
        }

    def render(self, *, limit_bytes: int = 120_000) -> str:
        return render_memory(self.path, limit_bytes=limit_bytes)

    def materialize(self, dest: Path) -> Path:
        """Copy the snapshot into ``dest`` so a branch can work on it."""
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.path, dest, ignore=shutil.ignore_patterns(*_IGNORED))
        return dest

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MemorySnapshot(label={self.label!r}, files={self.stats.file_count}, "
            f"bytes={self.stats.total_bytes}, hash={self.tree_hash[:12]})"
        )


class MemorySession:
    """A working copy with promotion rights.

    One session is opened per actor context that is allowed to learn: one per
    BRS branch at consolidation time, and one per DRS target attempt or practice
    project.  Nothing this session writes is visible to anyone else until
    :meth:`commit` runs, which is what makes "Work-phase memory edits are
    discarded before learning" a structural property rather than a convention.
    """

    def __init__(self, bank: "MemoryBank", path: Path, base: MemorySnapshot | None) -> None:
        self._bank = bank
        self.path = path
        self.base = base
        self._closed = False

    # -- reads ------------------------------------------------------------
    def read_all(self) -> dict[str, str]:
        return {
            p.relative_to(self.path).as_posix(): p.read_text(encoding="utf-8", errors="replace")
            for p in _iter_files(self.path)
        }

    def render(self, *, limit_bytes: int = 120_000) -> str:
        return render_memory(self.path, limit_bytes=limit_bytes)

    def list_files(self) -> list[str]:
        return [p.relative_to(self.path).as_posix() for p in _iter_files(self.path)]

    def read(self, relpath: str) -> str:
        target = self.path / relpath
        if not target.exists():
            raise MemoryIntegrityError(f"no such memory file: {relpath}")
        return target.read_text(encoding="utf-8", errors="replace")

    def stats(self) -> MemoryStats:
        return compute_stats(self.path)

    # -- writes (rejected once the session is closed) ---------------------
    def _guard(self) -> None:
        if self._closed:
            raise MemoryIntegrityError("memory session is closed; edits are no longer accepted")
        if not self._bank.writeback:
            raise MemoryIntegrityError(
                "frozen memory: writeback is disabled, so no edits may be made"
            )

    def write(self, relpath: str, content: str) -> None:
        self._guard()
        target = self.path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def delete(self, relpath: str) -> None:
        self._guard()
        target = self.path / relpath
        if target.exists():
            target.unlink()

    # -- promotion --------------------------------------------------------
    @property
    def changed(self) -> bool:
        """Whether this session differs from the snapshot it opened on."""
        if self.base is None:
            return compute_stats(self.path).file_count > 0
        return compute_stats(self.path) != self.base.stats

    def commit(self) -> MemoryStats:
        """Promote this session into the canonical bank.  Irreversible."""
        self._guard()
        stats = self._bank.promote(self.path)
        self._closed = True
        return stats

    def discard(self) -> None:
        """Drop the working copy without promoting anything."""
        self._closed = True
        shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> "MemorySession":
        return self

    def __exit__(self, *exc: object) -> None:
        if not self._closed:
            self.discard()


class MemoryBank:
    """Canonical, on-disk memory with promotion and freezing."""

    def __init__(self, root: str | Path, *, writeback: bool = True) -> None:
        self.root = Path(root).expanduser().resolve()
        self.writeback = writeback
        self._sessions_dir = self.root.parent / f"{self.root.name}.sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    # -- reads ------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return compute_stats(self.root).file_count == 0

    def stats(self) -> MemoryStats:
        return compute_stats(self.root)

    def tree_hash(self) -> str:
        return compute_stats(self.root).tree_hash

    def files(self) -> dict[str, bytes]:
        return {
            p.relative_to(self.root).as_posix(): p.read_bytes() for p in _iter_files(self.root)
        }

    def render(self, *, limit_bytes: int = 120_000) -> str:
        return render_memory(self.root, limit_bytes=limit_bytes)

    # -- snapshots and sessions -------------------------------------------
    def snapshot(self, *, label: str = "snapshot") -> MemorySnapshot:
        """Freeze the current canonical state for branches to share."""
        dest = self._sessions_dir / f"snap-{uuid.uuid4().hex[:10]}"
        dest.mkdir(parents=True, exist_ok=True)
        for path in _iter_files(self.root):
            rel = path.relative_to(self.root)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        return MemorySnapshot(dest, compute_stats(dest), label=label)

    def session(self, base: MemorySnapshot | None = None) -> MemorySession:
        """Open a promotable working copy, optionally on a snapshot's contents."""
        workdir = self._sessions_dir / f"session-{uuid.uuid4().hex[:10]}"
        if base is not None:
            base.materialize(workdir)
        else:
            workdir.mkdir(parents=True, exist_ok=True)
            for path in _iter_files(self.root):
                rel = path.relative_to(self.root)
                target = workdir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        return MemorySession(self, workdir, base)

    def promote(self, source: Path) -> MemoryStats:
        """Atomically replace canonical memory with ``source``."""
        if not self.writeback:
            raise MemoryIntegrityError("frozen memory: promotion is disabled")
        staging = self.root.parent / f"{self.root.name}.staging-{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=True, exist_ok=True)
        for path in _iter_files(source):
            rel = path.relative_to(source)
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

        backup = self.root.parent / f"{self.root.name}.previous"
        shutil.rmtree(backup, ignore_errors=True)
        if self.root.exists():
            self.root.rename(backup)
        staging.rename(self.root)
        shutil.rmtree(backup, ignore_errors=True)
        return compute_stats(self.root)

    # -- views ------------------------------------------------------------
    def readonly_view(self) -> MemorySnapshot:
        """A disposable copy for the curriculum agent.

        "The curriculum agent may inspect a disposable copy to guide
        exploration, but its local edits are not synchronized back"
        (Appendix A.2).  Returning a snapshot rather than a session is how that
        is enforced: there is no ``commit`` to call.
        """
        return self.snapshot(label="curriculum-view")

    def freeze_to(self, dest: Path) -> MemorySnapshot:
        """Copy the bank to ``dest`` and mark it immutable."""
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.root, dest, ignore=shutil.ignore_patterns(*_IGNORED))
        return MemorySnapshot(dest, compute_stats(dest), label="frozen")

    def assert_frozen(self, frozen: MemorySnapshot) -> None:
        """Verify a frozen snapshot is byte-identical to its recorded hash."""
        current = compute_stats(frozen.path)
        if current.tree_hash != frozen.stats.tree_hash:
            raise MemoryIntegrityError(
                "frozen memory changed during evaluation: "
                f"{frozen.stats.tree_hash[:12]} -> {current.tree_hash[:12]}"
            )

    def cleanup_sessions(self) -> None:
        shutil.rmtree(self._sessions_dir, ignore_errors=True)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
