"""The curriculum agent: what to explore next, and when to stop.

The curriculum agent is the only role whose decisions *shape the trajectory* of
exploration rather than judge or execute it.  Its two authorities are separated
cleanly in the paper, and both are respected here:

* It selects experiences and issues "continuation or stopping decisions".
* It does **not** grade candidates and does **not** author memory.  It reads a
  disposable copy, and its local edits are never synchronized back — which is
  enforced structurally by handing it a
  :class:`~rsiagent.memory.bank.MemorySnapshot` (no ``commit`` method exists)
  rather than a session.

Its context persists within one exploration lineage, so it can reconsider its
own hypotheses across waves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..parsing import ParsedHandoff, parse_handoff
from ..prompts import curriculum as curriculum_prompts
from ..status import Decision
from .context import AgentContext


@dataclass
class OutcomeSummary:
    """One completed project or attempt, as the curriculum agent sees it."""

    label: str
    verdict: str
    instruction: str
    findings: str = ""
    phase: str = "phase1_brs"
    practice: bool = False

    def render(self) -> str:
        return curriculum_prompts.render_outcome(
            label=self.label,
            verdict=self.verdict,
            instruction=self.instruction,
            findings=self.findings,
        )


@dataclass
class Handoff:
    """A parsed curriculum handoff plus bookkeeping."""

    decision: Decision
    rationale: str = ""
    projects: list = field(default_factory=list)
    raw: str = ""
    exhausted: bool = False

    @property
    def stops(self) -> bool:
        return self.decision in (Decision.SATURATED, Decision.STALLED)


class CurriculumAgent:
    """Selects exploration work and decides readiness."""

    def __init__(self, *, max_wave_projects: int = 8) -> None:
        self.max_wave_projects = max_wave_projects

    # -- Phase 1 ----------------------------------------------------------
    def author_wave(
        self,
        context: AgentContext,
        *,
        target_query: str,
        memory_view: str,
        outcomes: list[OutcomeSummary],
        first_wave: bool,
        max_projects: int | None = None,
    ) -> Handoff:
        """Ask for the next wave of broad exploration projects.

        "Budget checks happen at complete-wave boundaries; the Curriculum Agent
        can declare saturation earlier."  This is a *semantic* decision, so the
        caller passes the wave budget in its own prompt context and only
        enforces the numeric boundary between waves.
        """
        budget = max_projects or self.max_wave_projects
        rendered_outcomes = "\n".join(o.render() for o in outcomes) if outcomes else "(none yet)"
        prompt = curriculum_prompts.brs_context(
            target_query=target_query,
            memory=memory_view or "(memory is empty)",
            outcomes=rendered_outcomes,
            max_projects=budget,
            first_wave=first_wave,
        )
        return self._parse(context.ask(prompt), expect_phase1=True)

    # -- Phase 2 ----------------------------------------------------------
    def review(
        self,
        context: AgentContext,
        *,
        target_query: str,
        outcome: OutcomeSummary,
        diagnosis: str,
        memory_view: str,
        practice_count: int,
    ) -> Handoff:
        """Decide whether more practice beats returning to the target.

        Called after *both* PASS and FAIL, and after each practice project.
        Under ``curriculum_review`` this is what makes a target PASS
        non-terminal.
        """
        prompt = curriculum_prompts.drs_context(
            target_query=target_query,
            outcome=outcome.render(),
            diagnosis=diagnosis or "(no diagnosis supplied)",
            memory=memory_view or "(memory is empty)",
            practice_count=practice_count,
        )
        return self._parse(context.ask(prompt), expect_phase1=False)

    # -- parsing ----------------------------------------------------------
    def _parse(self, reply: str, *, expect_phase1: bool) -> Handoff:
        parsed: ParsedHandoff = parse_handoff(reply)
        decision = Decision(parsed.decision)

        if expect_phase1:
            # Phase-2 vocabulary leaking into Phase 1 means the same thing under
            # a different name; coerce rather than fail a long run over wording.
            if decision is Decision.READY_FOR_TARGET:
                decision = Decision.SATURATED
            elif decision is Decision.PROJECT:
                decision = Decision.PROJECTS
        else:
            if decision is Decision.PROJECTS:
                decision = Decision.PROJECT
            elif decision is Decision.SATURATED:
                # "Exploration is saturated" and "ready for the target" are the
                # same instruction in Phase 2: stop practising, go run it.  The
                # alternative — treating it as unhandled — would crash the
                # runner on a decision the model is entitled to make.
                decision = Decision.READY_FOR_TARGET

        return Handoff(
            decision=decision,
            rationale=parsed.rationale,
            projects=parsed.projects,
            raw=reply,
        )
