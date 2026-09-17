"""Append-only run journal.

The paper reconstructs its memory-growth figures "from the saved journals", and
records memory tree hashes so a frozen evaluation can be shown to have run
against exactly the memory exploration produced.  This module is the same idea
in miniature: one JSONL line per event, plus a self-contained summary.

The journal is a *host artifact*.  Nothing in it is ever fed back into an agent
prompt, which is what keeps official scores and audit records outside the
learning loop.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..memory.bank import MemoryStats


@dataclass
class JournalEvent:
    event: str
    phase: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=False)


def read_journal(path: str | Path) -> list[JournalEvent]:
    """Reconstruct a run's events from its JSONL file.

    The file is the authoritative record: a journal reopened after the process
    that wrote it has exited must yield the same series, because that is how the
    paper reconstructs its memory-growth figures "from the saved journals".
    """
    target = Path(path).expanduser()
    if not target.exists():
        return []
    events: list[JournalEvent] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        blob = json.loads(line)
        events.append(
            JournalEvent(
                event=blob.get("event", ""),
                phase=blob.get("phase", ""),
                data=blob.get("data", {}) or {},
                ts=blob.get("ts", 0.0),
            )
        )
    return events


def read_memory_series(path: str | Path) -> list[dict[str, Any]]:
    """The memory-checkpoint series from a journal file.

    The event's phase is folded back into each record, so consumers get one flat
    row per checkpoint without having to re-correlate against the event stream.
    """
    return [
        {**event.data, "phase": event.phase}
        for event in read_journal(path)
        if event.event == "memory_checkpoint"
    ]


class Journal:
    """Append-only JSONL writer.

    ``on_event`` is an optional observer called with each :class:`JournalEvent`
    as it is written.  It exists so a caller can narrate a run live without
    re-reading the file, and it must not raise — journaling is bookkeeping, and
    a broken observer should never take down a lineage.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        on_event: Callable[[JournalEvent], None] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._seq = 0
        self._on_event = on_event

    # -- writing ----------------------------------------------------------
    def log(self, event: str, *, phase: str = "", **data: Any) -> JournalEvent:
        self._seq += 1
        entry = JournalEvent(event=event, phase=phase, data=data)
        self._handle.write(entry.to_json() + "\n")
        self._handle.flush()
        if self._on_event is not None:
            # Observers are best-effort: journaling is bookkeeping, and a broken
            # narrator must never take down a lineage.
            with contextlib.suppress(Exception):
                self._on_event(entry)
        return entry

    def log_memory(
        self,
        stats: MemoryStats,
        *,
        phase: str,
        label: str,
        kind: str = "commit",
        verdict: str | None = None,
    ) -> None:
        """Record a memory checkpoint.

        ``kind`` is one of ``brs_project``, ``drs_checkpoint``, ``freeze``,
        ``snapshot`` — the units Figure A1 distinguishes.  A red cross in that
        figure is a failed outcome that still produced an update, which is why
        ``verdict`` is recorded alongside.
        """
        self.log(
            "memory_checkpoint",
            phase=phase,
            seq=self._seq,
            label=label,
            kind=kind,
            verdict=verdict,
            **stats.as_dict(),
        )

    def memory_growth(self) -> list[dict[str, Any]]:
        """Every memory checkpoint this run recorded, read back from the file."""
        return read_memory_series(self.path)

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullJournal(Journal):
    """A journal that writes nowhere.  For tests and dry runs."""

    def __init__(self) -> None:  # noqa: D107
        self.path = Path("/dev/null")
        self._seq = 0

        class _Sink:
            closed = False

            def write(self, _: str) -> int:
                return 0

            def flush(self) -> None:
                pass

        self._handle = _Sink()  # type: ignore[assignment]

    def close(self) -> None:
        pass
