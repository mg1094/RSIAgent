"""Environment interface.

The paper drives a QEMU-backed Ubuntu guest.  Nothing in the RSI protocol
depends on that: the protocol needs an environment it can *reset*, *execute
programs in*, *snapshot*, and *restore*.  Modelling those four verbs as a
protocol is what lets the same lifecycle run against a laptop directory, a
container, or a VM.

Two invariants from the paper are expressed in this interface:

1. **Code is the control channel.**  The actor never clicks; it submits whole
   programs, and gets back combined stdout+stderr plus an exit code
   (Appendix A.1, "Program-Based Actions").
2. **Verification is checkpoint-protected.**  The verifier may run arbitrary
   probes, and the harness restores the pre-inspection checkpoint before the
   candidate is graded, "so none of your in-VM effects enter the scored state"
   (Prompt P2).
"""

from __future__ import annotations

import abc
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ProgramKind = Literal["python", "bash"]


@dataclass
class ExecResult:
    """Outcome of one submitted program."""

    program: str
    kind: ProgramKind
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_s: float = 0.0
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Combined stdout+stderr — what the actor actually receives."""
        parts = [self.stdout]
        if self.stderr:
            parts.append(self.stderr)
        return "".join(parts)

    def render(self, limit: int = 12_000) -> str:
        """Prompt-ready rendering, mirroring the runtime's return payload."""
        body = self.output
        if len(body) > limit:
            head = body[: limit // 2]
            tail = body[-limit // 2 :]
            body = f"{head}\n... [{len(body) - limit} chars elided] ...\n{tail}"
        status = "TIMEOUT" if self.timed_out else f"exit={self.exit_code}"
        return f"$ ({self.kind}) [{status}, {self.duration_s:.2f}s]\n{body.rstrip()}"


@dataclass
class Observation:
    """A ``look`` result: what the actor would see if it had eyes."""

    kind: Literal["text", "image"] = "text"
    text: str = ""
    image_path: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Environment(abc.ABC):
    """A resettable, snapshottable execution surface."""

    name: str = "environment"

    # -- lifecycle --------------------------------------------------------
    @abc.abstractmethod
    def reset(self) -> None:
        """Return to the sealed baseline state.

        The paper resets the environment between independent attempts: "Each
        evaluation starts with a reset interaction history and task
        environment."  Interaction history is the *caller's* concern; this
        method only handles the world.
        """

    @abc.abstractmethod
    def execute(
        self,
        program: str,
        *,
        kind: ProgramKind = "python",
        timeout: float | None = None,
    ) -> ExecResult:
        """Run one program and return its combined output and exit status."""

    # -- inspection -------------------------------------------------------
    @abc.abstractmethod
    def observe(self, hint: str = "") -> Observation:
        """Return what is visible without running a program."""

    @abc.abstractmethod
    def list_files(self, subdir: str = "") -> list[str]:
        """Relative paths of regular files visible to the actor."""

    @abc.abstractmethod
    def read_bytes(self, relpath: str) -> bytes:
        """Read one file from the environment."""

    def read_text(self, relpath: str, encoding: str = "utf-8", errors: str = "replace") -> str:
        return self.read_bytes(relpath).decode(encoding, errors)

    @abc.abstractmethod
    def exists(self, relpath: str) -> bool:
        """Whether a relative path exists in the environment."""

    def size_bytes(self, relpath: str) -> int:
        return len(self.read_bytes(relpath))

    # -- fixtures ---------------------------------------------------------
    @abc.abstractmethod
    def write_fixture(self, relpath: str, content: str | bytes) -> None:
        """Place an input fixture.  Only valid before the baseline is sealed."""

    # -- checkpoints ------------------------------------------------------
    @abc.abstractmethod
    def snapshot(self, label: str) -> str:
        """Freeze current state and return a checkpoint id."""

    @abc.abstractmethod
    def restore(self, checkpoint_id: str) -> None:
        """Roll the environment back to a checkpoint."""

    @abc.abstractmethod
    def discard(self, checkpoint_id: str) -> None:
        """Release a checkpoint's storage."""

    # -- convenience ------------------------------------------------------
    def shell(self, command: str, *, timeout: float | None = None) -> ExecResult:
        return self.execute(command, kind="bash", timeout=timeout)

    def describe(self) -> str:
        """Short human-readable summary for journals and prompts."""
        files = self.list_files()
        preview = "\n".join(f"  {shlex.quote(f)}" for f in files[:40])
        more = f"\n  ... and {len(files) - 40} more" if len(files) > 40 else ""
        return f"{self.name}: {len(files)} files\n{preview}{more}"
