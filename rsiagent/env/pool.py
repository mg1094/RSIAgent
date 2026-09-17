"""Environment factories for isolated branches.

"Every project in a wave begins with the same immutable pre-wave memory snapshot
and cannot observe sibling work or outcomes" (Appendix A.2).  Isolation between
siblings is therefore a requirement, not an optimisation — and the cheapest
honest way to get it is one environment per branch rather than one shared
environment with careful bookkeeping.

A factory returns a *fresh* environment per label, so the runner can execute a
wave concurrently and provision target attempts on demand.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from .base import Environment
from .local import LocalWorkspaceEnvironment

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


@runtime_checkable
class EnvironmentFactory(Protocol):
    """Builds an isolated environment for one branch or attempt."""

    def __call__(self, label: str) -> Environment: ...


def safe_label(label: str) -> str:
    """Make a label usable as a directory name."""
    cleaned = _SAFE.sub("-", label).strip("-")
    return cleaned or "branch"


class LocalEnvironmentPool:
    """Hands out isolated :class:`LocalWorkspaceEnvironment` instances.

    Layout::

        <root>/<sanitised-label>/workspace/     the actor's world
        <root>/<sanitised-label>/workspace.baseline/
        <root>/<sanitised-label>/workspace.checkpoints/
        <root>/<sanitised-label>/workspace.private/
    """

    def __init__(
        self,
        root: str | Path,
        *,
        program_timeout_s: float = 600.0,
        allow_execution: bool = True,
        env: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.program_timeout_s = program_timeout_s
        self.allow_execution = allow_execution
        self.env = env

    def __call__(self, label: str) -> Environment:
        branch_dir = self.root / safe_label(label)
        branch_dir.mkdir(parents=True, exist_ok=True)
        return LocalWorkspaceEnvironment(
            branch_dir / "workspace",
            name=safe_label(label),
            program_timeout_s=self.program_timeout_s,
            allow_execution=self.allow_execution,
            env=self.env,
        )

    def cleanup(self, keep_labels: set[str] | None = None) -> None:
        """Remove branch directories, optionally keeping some."""
        import shutil

        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            if keep_labels and child.name in keep_labels:
                continue
            shutil.rmtree(child, ignore_errors=True)


class SingleEnvironmentFactory:
    """Always returns the same environment.  For tests and single-branch runs."""

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def __call__(self, label: str) -> Environment:
        return self.environment
