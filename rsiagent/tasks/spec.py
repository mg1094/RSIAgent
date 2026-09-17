"""Task, project, and score types.

Two boundaries live here.

**Practice vs. target.**  A practice project is authored by the curriculum
agent, judged PASS/FAIL, and exists to produce transferable knowledge.  The
target task is fixed from outside, judged PASS/FAIL/UNVERIFIED, and is the thing
that eventually gets scored.

**Agents vs. evaluator.**  The evaluator that produces the official score is
*sealed*: "The official evaluator is invoked after the actor–verifier loop and
does not supply scores or hidden checks to the learning agents."  Keeping
:class:`TaskScore` out of every prompt path is what makes that true — the score
is recorded in the journal and never rendered into a context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..env.base import Environment


@dataclass(frozen=True)
class Project:
    """An exploration project authored by the curriculum agent."""

    id: str
    instruction: str
    fixtures: dict[str, str] = field(default_factory=dict)
    wave: int = 0

    def label(self) -> str:
        return f"wave{self.wave}:{self.id}" if self.wave else self.id


@dataclass(frozen=True)
class TaskQuery:
    """The target task.

    ``instruction`` is authoritative for both the actor and the verifier, and is
    the only part of the task the curriculum agent sees.  Any rubric, hidden
    check, or expected value lives in the evaluator and never reaches an agent.
    """

    id: str
    instruction: str
    fixtures: dict[str, str] = field(default_factory=dict)

    def label(self) -> str:
        return self.id


@dataclass
class TaskScore:
    """The official evaluator's output."""

    partial: float
    """Normalized task score in [0, 1].  The paper's ``Partial`` metric."""

    components: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    error: str = ""

    @property
    def binary(self) -> bool:
        """Full credit.  The paper's ``Binary`` metric is 1{s_i = 1}."""
        return self.partial >= 1.0

    @property
    def scored(self) -> bool:
        """Whether this is a real score rather than an infrastructure failure."""
        return not self.error

    def as_dict(self) -> dict[str, object]:
        return {
            "partial": round(self.partial, 6),
            "binary": self.binary,
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "notes": self.notes,
            "error": self.error,
        }


@runtime_checkable
class Evaluator(Protocol):
    """The sealed official evaluator."""

    name: str

    def score(self, environment: Environment) -> TaskScore:
        """Score the candidate currently in ``environment``."""


class UnscoredEvaluator:
    """Records that no official score exists.

    Used for stage ablations and offline demos where no benchmark grader is
    available.  An infrastructure failure is *unscored*, never an official zero.
    """

    name = "unscored"

    def __init__(self, reason: str = "no official evaluator configured") -> None:
        self.reason = reason

    def score(self, environment: Environment) -> TaskScore:
        return TaskScore(partial=0.0, notes=self.reason, error=self.reason)


class ScriptedEvaluator:
    """A local evaluator built from named check functions.

    Each check returns a value in [0, 1] or a bool.  Partial credit is the mean
    over checks with equal weight, which mirrors the paper's "each task has
    equal weight" convention one level down.
    """

    name = "scripted-rubric"

    def __init__(self, checks: dict[str, object], *, notes: str = "") -> None:
        self.checks = checks
        self.notes = notes

    def score(self, environment: Environment) -> TaskScore:
        components: dict[str, float] = {}
        for name, check in self.checks.items():
            try:
                value = check(environment)  # type: ignore[operator]
            except Exception as exc:  # a broken check is a broken run, not a zero
                return TaskScore(
                    partial=0.0,
                    components=components,
                    error=f"check {name!r} raised {type(exc).__name__}: {exc}",
                )
            if isinstance(value, bool):
                components[name] = 1.0 if value else 0.0
            else:
                components[name] = max(0.0, min(1.0, float(value)))
        partial = sum(components.values()) / len(components) if components else 0.0
        return TaskScore(partial=partial, components=components, notes=self.notes)


def provision(environment: Environment, fixtures: dict[str, str], *, reset: bool = True) -> None:
    """Install fixtures into a fresh environment and seal the baseline.

    ``prepare`` empties the workspace without restoring the old baseline, which
    matters because :meth:`Environment.reset` would otherwise re-seal an empty
    tree before the fixtures land.
    """
    if reset:
        prepare = getattr(environment, "prepare", None)
        if callable(prepare):
            prepare()
        else:
            environment.reset()
    for relpath, content in fixtures.items():
        environment.write_fixture(relpath, content)
    seal = getattr(environment, "seal_baseline", None)
    if callable(seal):
        seal()


def load_fixture_dir(directory: str | Path) -> dict[str, str]:
    """Read every file under ``directory`` into a fixture mapping."""
    base = Path(directory).expanduser().resolve()
    if not base.exists():
        return {}
    return {
        path.relative_to(base).as_posix(): path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }
