"""Experience-owned learning: distillation, reconciliation, diagnosis.

Appendix A.2 specifies two learning steps that run **in the actor context that
produced the experience**, followed by a separate handoff to the curriculum
agent::

    After a grounded PASS or FAIL, its existing context receives the complete
    verifier report and enters two learning steps. Distillation identifies
    useful procedures, constraints, and failure lessons; reconciliation checks
    the resulting memory against older entries, revising contradictions and
    qualifying unsupported conclusions.

Running learning here — rather than in a fresh context — is what lets the model
re-read its own trajectory.  It is also why the actor that failed a project can
still contribute valuable evidence: it remembers *why*.

The diagnosis is the privacy boundary.  It carries conclusions and hypotheses,
never the reasoning transcript: "provide only the concise conclusions and
hypotheses you choose to communicate."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import UnverifiedOutcome
from ..memory.bank import MemorySession, MemoryStats
from ..prompts import learning as learning_prompts
from ..parsing import parse_diagnosis, parse_memory_edits
from ..status import Verdict
from ..agents.context import AgentContext


@dataclass
class LearningResult:
    """What one consolidation produced."""

    verdict: Verdict
    diagnosis: str = ""
    stats_before: MemoryStats | None = None
    stats_after: MemoryStats | None = None
    files_written: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    committed: bool = False

    @property
    def memory_changed(self) -> bool:
        if self.stats_before is None or self.stats_after is None:
            return False
        return self.stats_before.tree_hash != self.stats_after.tree_hash

    def render(self) -> str:
        before = self.stats_before.file_count if self.stats_before else 0
        after = self.stats_after.file_count if self.stats_after else 0
        delta = after - before
        return (
            f"learning[{self.verdict.value}]: {len(self.files_written)} writes, "
            f"{len(self.files_deleted)} deletes, files {before} -> {after} ({delta:+d}), "
            f"committed={self.committed}"
        )


def consolidate(
    context: AgentContext,
    session: MemorySession,
    *,
    verdict: Verdict,
    findings: str,
    commit: bool = True,
    enable_diagnosis: bool = True,
) -> LearningResult:
    """Run distillation, reconciliation, and diagnosis; optionally promote.

    Raises :class:`~rsiagent.errors.UnverifiedOutcome` when the verdict is
    UNVERIFIED: "an unresolved outcome does not become a successful or failed
    learning example."
    """
    if not verdict.grounded:
        raise UnverifiedOutcome(
            "refusing to consolidate an unresolved outcome", findings=findings
        )

    result = LearningResult(verdict=verdict, stats_before=session.stats())

    # --- Step 1: distillation, in the context that did the work ----------
    draft_reply = context.ask(
        learning_prompts.distillation(verdict.value, findings, session.render())
    )
    _apply_edits(session, draft_reply, result)

    # --- Step 2: reconciliation, against the whole bank ------------------
    reconcile_reply = context.ask(learning_prompts.reconciliation(session.render()))
    _apply_edits(session, reconcile_reply, result)

    # --- Handoff: a diagnosis the curriculum agent can act on ------------
    if enable_diagnosis:
        result.diagnosis = parse_diagnosis(context.ask(learning_prompts.DIAGNOSIS))

    result.stats_after = session.stats()
    if commit:
        result.stats_after = session.commit()
        result.committed = True
    return result


def _apply_edits(session: MemorySession, reply: str, result: LearningResult) -> None:
    for edit in parse_memory_edits(reply):
        if edit.op == "write":
            session.write(edit.path, edit.content)
            result.files_written.append(edit.path)
        else:
            session.delete(edit.path)
            result.files_deleted.append(edit.path)
