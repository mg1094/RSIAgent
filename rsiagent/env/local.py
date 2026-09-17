"""A directory-backed environment that runs programs as subprocesses.

This is the drop-in replacement for the paper's QEMU guest when you want the
protocol to run on a workstation.  It is a *convenience* environment, not a
security boundary.

.. warning::
   :meth:`LocalWorkspaceEnvironment.execute` runs model-authored Python and
   shell programs **on the host**, with the privileges of this process.  The
   paper's reference setup isolates the actor inside a QEMU VM and the verifier
   inside a restorable checkpoint of that VM.  If you point this environment at
   anything you do not control, put it in a container or a disposable VM first:
   the RSI protocol will happily let a model write and execute arbitrary code.

The workspace layout::

    <root>/                 the actor's world — sealed baseline, resettable
    <root>.baseline/        the sealed copy `reset()` restores from
    <root>.checkpoints/     snapshot storage, kept outside the actor's view
    <root>.private/         actor-private scratch the verifier never sees

Everything the actor can see lives under ``root``.  Memory is stored elsewhere
by :mod:`rsiagent.memory` and is therefore invisible to the verifier, which is
how the paper's "actor-private memory, reasoning, and execution logs are
hidden" boundary is enforced structurally rather than by prompt.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

from ..errors import CheckpointError, EnvironmentError_
from .base import Environment, ExecResult, Observation, ProgramKind

_IGNORED_NAMES = {"__pycache__", ".DS_Store", ".git", ".venv", "node_modules"}


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy ``src`` onto ``dst``, replacing it entirely."""
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(
        src,
        dst,
        symlinks=True,
        ignore=shutil.ignore_patterns(*_IGNORED_NAMES),
    )


class LocalWorkspaceEnvironment(Environment):
    """Resettable workspace with subprocess program execution."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str = "local-workspace",
        program_timeout_s: float = 600.0,
        allow_execution: bool = True,
        env: dict[str, str] | None = None,
        observe_depth: int = 2,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.name = name
        self.program_timeout_s = program_timeout_s
        self.allow_execution = allow_execution
        self.observe_depth = observe_depth

        self.baseline_dir = self.root.parent / f"{self.root.name}.baseline"
        self.checkpoint_dir = self.root.parent / f"{self.root.name}.checkpoints"
        self.private_dir = self.root.parent / f"{self.root.name}.private"

        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.private_dir.mkdir(parents=True, exist_ok=True)

        self._sealed = self.baseline_dir.exists()
        self._step = 0

        # A deliberately small environment: no inherited secrets, no surprises.
        self._base_env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(self.private_dir),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(self.private_dir / "tmp"),
            "RSIAGENT_WORKSPACE": str(self.root),
        }
        if env:
            self._base_env.update(env)
        (self.private_dir / "tmp").mkdir(parents=True, exist_ok=True)

    # -- fixtures & lifecycle --------------------------------------------
    def write_fixture(self, relpath: str, content: str | bytes) -> None:
        if self._sealed:
            raise EnvironmentError_(
                "fixtures must be written before seal_baseline(); "
                "call reset() only after the baseline exists"
            )
        target = self.root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8") if isinstance(content, str) else content
        target.write_bytes(data)

    def prepare(self) -> None:
        """Empty the workspace and unseal it, so fixtures may be installed.

        Distinct from :meth:`reset`, which restores the *sealed* baseline.  Use
        this when provisioning a task for the first time.
        """
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self._sealed = False
        self._step = 0

    def seal_baseline(self) -> Path:
        """Freeze the current workspace as the state ``reset()`` restores."""
        _copy_tree(self.root, self.baseline_dir)
        self._sealed = True
        return self.baseline_dir

    def reset(self) -> None:
        if not self.baseline_dir.exists():
            self.seal_baseline()
            return
        _copy_tree(self.baseline_dir, self.root)
        self._step = 0

    @property
    def sealed(self) -> bool:
        return self._sealed

    # -- execution --------------------------------------------------------
    def execute(
        self,
        program: str,
        *,
        kind: ProgramKind = "python",
        timeout: float | None = None,
    ) -> ExecResult:
        if not self.allow_execution:
            raise EnvironmentError_(
                "program execution is disabled for this environment (allow_execution=False)"
            )
        timeout = timeout or self.program_timeout_s
        self._step += 1

        suffix = ".py" if kind == "python" else ".sh"
        script = self.private_dir / "programs" / f"step_{self._step:04d}{suffix}"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(program, encoding="utf-8")

        if kind == "python":
            argv = [sys.executable, "-I", "-B", str(script)]
        elif kind == "bash":
            argv = ["/bin/bash", "--noprofile", "--norc", str(script)]
        else:
            raise EnvironmentError_(f"unsupported program kind: {kind!r}")

        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=self.root,
                env=self._base_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
                start_new_session=True,  # so a timeout can kill the whole group
            )
            duration = time.monotonic() - started
            return ExecResult(
                program=program,
                kind=kind,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                exit_code=completed.returncode,
                duration_s=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            return ExecResult(
                program=program,
                kind=kind,
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr) + f"\n[timeout after {timeout:.0f}s]",
                exit_code=124,
                duration_s=duration,
                timed_out=True,
            )

    # -- inspection -------------------------------------------------------
    def list_files(self, subdir: str = "") -> list[str]:
        base = (self.root / subdir) if subdir else self.root
        if not base.exists():
            return []
        out: list[str] = []
        for path in sorted(base.rglob("*")):
            if any(part in _IGNORED_NAMES for part in path.parts):
                continue
            if path.is_file():
                out.append(str(path.relative_to(self.root)))
        return out

    def read_bytes(self, relpath: str) -> bytes:
        target = self.root / relpath
        if not target.exists():
            raise EnvironmentError_(f"no such file in environment: {relpath}")
        return target.read_bytes()

    def exists(self, relpath: str) -> bool:
        return (self.root / relpath).exists()

    def observe(self, hint: str = "") -> Observation:
        """A text stand-in for a screenshot.

        A real GUI environment returns pixels here.  This environment has no
        screen, so it reports the file tree and any files the hint names, which
        is enough for the actor to orient itself.
        """
        lines = [f"workspace: {self.root.name}", "files:"]
        for relpath in self.list_files():
            size = (self.root / relpath).stat().st_size
            lines.append(f"  {relpath}  ({size} bytes)")

        if hint:
            for token in _candidate_paths(hint):
                if self.exists(token):
                    text = self.read_text(token)
                    if len(text) > 4000:
                        text = text[:2000] + "\n... [truncated] ...\n" + text[-2000:]
                    lines.append(f"\n--- {token} ---\n{text}")

        return Observation(kind="text", text="\n".join(lines))

    # -- checkpoints ------------------------------------------------------
    def snapshot(self, label: str = "checkpoint") -> str:
        checkpoint_id = f"{label}-{uuid.uuid4().hex[:12]}"
        target = self.checkpoint_dir / checkpoint_id
        try:
            _copy_tree(self.root, target)
        except OSError as exc:  # pragma: no cover - disk-full and friends
            raise CheckpointError(f"snapshot failed: {exc}") from exc
        return checkpoint_id

    def restore(self, checkpoint_id: str) -> None:
        source = self.checkpoint_dir / checkpoint_id
        if not source.exists():
            raise CheckpointError(f"unknown checkpoint: {checkpoint_id}")
        _copy_tree(source, self.root)

    def discard(self, checkpoint_id: str) -> None:
        shutil.rmtree(self.checkpoint_dir / checkpoint_id, ignore_errors=True)

    # -- extra ------------------------------------------------------------
    def write_private(self, relpath: str, content: str | bytes) -> Path:
        """Write into actor-private storage.  Never visible to the verifier."""
        target = self.private_dir / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8") if isinstance(content, str) else content
        target.write_bytes(data)
        return target

    def cleanup_checkpoints(self) -> None:
        shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def _decode(blob: bytes | str | None) -> str:
    if blob is None:
        return ""
    if isinstance(blob, bytes):
        return blob.decode("utf-8", "replace")
    return blob


def _candidate_paths(hint: str) -> Iterable[str]:
    """Pull plausible relative file paths out of a free-text hint."""
    for raw in hint.replace(",", " ").split():
        token = raw.strip("'\"`()[]")
        if token and not token.startswith("-") and ("/" in token or "." in token):
            yield token.lstrip("/")
