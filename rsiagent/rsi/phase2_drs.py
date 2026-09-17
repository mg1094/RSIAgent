"""Stage 2 — Deep Recursive Self-exploration.

This is Algorithm A1, steps 3–6, and the ordering is easy to get subtly wrong.
The reference procedure, verbatim:

    3. DRS: attempt q after an environment reset and obtain a grounded target
       verdict. The same actor context consolidates that outcome into M.
    4. If the policy is verifier_pass and the target passed, proceed to Step 7.
       Otherwise, ask the curriculum agent to select the next experience.
    5. Execute each selected practice project sequentially: actor execution,
       verifier judgment, actor memory update, then curriculum review.
    6. If the preceding target passed and no new practice was selected, finish
       DRS. Otherwise, return to Step 3.

Two consequences are implemented literally below, because they are the parts
people drop:

* **A target PASS does not end DRS by default.**  Under ``curriculum_review``
  the curriculum agent still reviews the outcome, and if it selects practice,
  the target must be re-attempted afterwards.  Tracking
  ``practice_since_target`` is what distinguishes "passed, nothing left to
  learn" from "passed, then learned something that invalidates the pass".
* **STALLED is not convergence.**  It buys one final target attempt and is
  recorded as :attr:`TerminalStatus.STALLED`, never as
  :attr:`TerminalStatus.COMPLETED`.

The curriculum agent's readiness decision is also *not* a correctness verdict.
It is consulted in exactly one place — whether to keep practising.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..agents.actor import ActorAgent
from ..agents.context import AgentContext
from ..agents.curriculum import CurriculumAgent, OutcomeSummary
from ..agents.verifier import VerifierAgent
from ..config import RunConfig
from ..env.base import Environment
from ..errors import InfrastructureFailure, UnverifiedOutcome
from ..learning.consolidation import LearningResult, consolidate
from ..memory.bank import MemoryBank, MemorySession
from ..runtime.harness import AttemptResult, TaskHarness
from ..runtime.journal import Journal
from ..status import Decision, Phase, TerminalStatus, Verdict
from ..tasks.spec import Project, TaskQuery, provision


@dataclass
class TargetCycle:
    """One target attempt plus the review and practice it triggered."""

    index: int
    attempt: AttemptResult
    learning: LearningResult | None = None
    review_decision: Decision | None = None
    review_rationale: str = ""
    practice: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        return self.attempt.verdict


@dataclass
class Phase2Result:
    cycles: list[TargetCycle] = field(default_factory=list)
    practice_projects: list[str] = field(default_factory=list)
    status: TerminalStatus = TerminalStatus.COMPLETED
    final_verdict: Verdict | None = None
    final_attempt: AttemptResult | None = None
    rationale: str = ""

    @property
    def target_passed(self) -> bool:
        return self.final_verdict is Verdict.PASS


class DeepRefiner:
    """Target-conditioned sequential refinement."""

    def __init__(
        self,
        *,
        config: RunConfig,
        actor: ActorAgent,
        verifier: VerifierAgent,
        curriculum: CurriculumAgent,
        memory: MemoryBank,
        journal: Journal,
        target_environment: Environment,
        practice_environment_factory,
        actor_context_factory,
        verifier_context_factory,
        practice_verifier_context_factory,
        curriculum_context: AgentContext,
    ) -> None:
        self.config = config
        self.actor = actor
        self.verifier = verifier
        self.curriculum = curriculum
        self.memory = memory
        self.journal = journal
        self.target_environment = target_environment
        self.practice_environment_factory = practice_environment_factory
        self.actor_context_factory = actor_context_factory
        self.verifier_context_factory = verifier_context_factory
        self.practice_verifier_context_factory = practice_verifier_context_factory
        self.curriculum_context = curriculum_context
        self.target_harness = TaskHarness(
            actor, verifier, max_revisions=config.exploration.max_target_revisions
        )
        self.practice_harness = TaskHarness(
            actor, verifier, max_revisions=config.exploration.max_practice_revisions
        )

    # -- driver -----------------------------------------------------------
    def run(self, task: TaskQuery, *, target_query: str) -> Phase2Result:
        result = Phase2Result()
        exploration = self.config.exploration

        target_actor = self.actor_context_factory()
        target_verifier = self.verifier_context_factory()

        attempt = self._attempt_target(
            task, target_actor, target_verifier, cycle_index=0, memory_text=self.memory.render()
        )
        learning = self._consolidate_target(attempt, target_actor, cycle_index=0)
        cycle = TargetCycle(index=0, attempt=attempt, learning=learning)
        result.cycles.append(cycle)
        result.final_attempt = attempt
        result.final_verdict = attempt.verdict

        if not attempt.grounded:
            result.status = TerminalStatus.UNVERIFIED
            result.rationale = "target attempt could not be verified"
            return self._finish(result)

        if exploration.stop_policy == "verifier_pass" and attempt.passed:
            result.status = TerminalStatus.COMPLETED
            result.rationale = "verifier_pass: grounded target PASS"
            return self._finish(result)

        practice_since_target = False
        practice_count = 0
        cycle_index = 0

        # What the next curriculum review will look at.  "Execute each selected
        # practice project sequentially: actor execution, verifier judgment,
        # actor memory update, then curriculum review" — so after a practice
        # project the *practice's* grounded outcome is the one under review, not
        # the older target attempt.
        last_attempt: AttemptResult = attempt
        last_learning: LearningResult | None = learning
        last_label = "target"

        while True:
            review = self._review(
                target_query=target_query,
                attempt=last_attempt,
                diagnosis=last_learning.diagnosis if last_learning else "",
                practice_count=practice_count,
                label=last_label,
            )
            cycle.review_decision = review.decision
            cycle.review_rationale = review.rationale

            # --- Step 6: passed, and nothing new was selected ---------------
            if review.decision is Decision.READY_FOR_TARGET:
                if attempt.passed and not practice_since_target:
                    result.status = TerminalStatus.COMPLETED
                    result.rationale = review.rationale or "curriculum ready; target already passed"
                    return self._finish(result)
                # Either the target failed, or it passed before new practice
                # changed the memory it passed under.  Return to Step 3.
                cycle_index += 1
                if cycle_index >= exploration.drs_max_target_cycles:
                    result.status = TerminalStatus.BUDGET_EXHAUSTED
                    result.rationale = (
                        f"reached drs_max_target_cycles={exploration.drs_max_target_cycles}"
                    )
                    return self._finish(result)
                attempt = self._attempt_target(
                    task,
                    target_actor,
                    target_verifier,
                    cycle_index=cycle_index,
                    memory_text=self.memory.render(),
                )
                learning = self._consolidate_target(attempt, target_actor, cycle_index)
                cycle = TargetCycle(index=cycle_index, attempt=attempt, learning=learning)
                result.cycles.append(cycle)
                result.final_attempt = attempt
                result.final_verdict = attempt.verdict
                practice_since_target = False
                last_attempt, last_learning, last_label = attempt, learning, "target"
                if not attempt.grounded:
                    result.status = TerminalStatus.UNVERIFIED
                    result.rationale = "target attempt could not be verified"
                    return self._finish(result)
                continue

            # --- STALLED buys one final target attempt ----------------------
            if review.decision is Decision.STALLED:
                final = self._attempt_target(
                    task,
                    target_actor,
                    target_verifier,
                    cycle_index=cycle_index + 1,
                    memory_text=self.memory.render(),
                )
                self._consolidate_target(final, target_actor, cycle_index + 1)
                result.final_attempt = final
                result.final_verdict = final.verdict
                result.status = TerminalStatus.STALLED
                result.rationale = review.rationale or "curriculum stalled"
                return self._finish(result)

            # --- PROJECT: one practice project, then review again -----------
            if (
                exploration.drs_max_practice_projects is not None
                and practice_count >= exploration.drs_max_practice_projects
            ):
                result.status = TerminalStatus.BUDGET_EXHAUSTED
                result.rationale = (
                    f"reached drs_max_practice_projects={exploration.drs_max_practice_projects}"
                )
                return self._finish(result)

            if not review.projects:  # pragma: no cover - parser rejects this
                raise InfrastructureFailure(
                    "phase2",
                    f"curriculum decision {review.decision.value} carried no project",
                )

            project = Project(
                id=review.projects[0].id,
                instruction=review.projects[0].instruction,
                fixtures=review.projects[0].fixtures,
            )
            practice_attempt, practice_learning = self._run_practice(project, practice_count)
            practice_count += 1
            practice_since_target = True
            cycle.practice.append(project.id)
            result.practice_projects.append(project.id)
            if practice_attempt is not None:
                last_attempt, last_learning, last_label = (
                    practice_attempt,
                    practice_learning,
                    f"practice:{project.id}",
                )

    def _finish(self, result: Phase2Result) -> Phase2Result:
        self.journal.log(
            "phase_complete",
            phase=Phase.DRS.value,
            status=result.status.value,
            cycles=len(result.cycles),
            practice=len(result.practice_projects),
            final_verdict=result.final_verdict.value if result.final_verdict else None,
            rationale=result.rationale,
            memory=self.memory.stats().as_dict(),
        )
        return result

    # -- target -----------------------------------------------------------
    def _attempt_target(
        self,
        task: TaskQuery,
        actor_context: AgentContext,
        verifier_context: AgentContext,
        *,
        cycle_index: int,
        memory_text: str,
    ) -> AttemptResult:
        """Attempt the target "in a reset environment using the updated memory"."""
        self.target_environment.reset()
        self.journal.log(
            "target_attempt_start",
            phase=Phase.DRS.value,
            cycle=cycle_index,
            memory_hash=self.memory.tree_hash(),
        )
        try:
            attempt = self.target_harness.attempt(
                self.target_environment,
                actor_context=actor_context,
                verifier_context=verifier_context,
                instruction=task.instruction,
                memory_text=memory_text,
                practice=False,
                task_id=task.id,
                iteration_limit=self.config.limits.target_iterations,
                watchdog_s=self.config.limits.target_watchdog_s,
                program_timeout_s=self.config.limits.program_timeout_s,
            )
        except InfrastructureFailure as exc:
            self.journal.log(
                "target_attempt_failed", phase=Phase.DRS.value, cycle=cycle_index, error=str(exc)
            )
            raise

        self.journal.log(
            "target_verdict",
            phase=Phase.DRS.value,
            cycle=cycle_index,
            verdict=attempt.verdict.value,
            revisions=attempt.revisions,
            terminal=attempt.terminal,
            findings=attempt.final_findings[:4000],
        )
        return attempt

    def _consolidate_target(
        self, attempt: AttemptResult, actor_context: AgentContext, cycle_index: int
    ) -> LearningResult | None:
        """Both PASS and FAIL ground learning; UNVERIFIED grounds nothing."""
        if not attempt.grounded:
            self.journal.log(
                "target_learning_skipped",
                phase=Phase.DRS.value,
                cycle=cycle_index,
                reason="unverified outcome is not a learning example",
            )
            return None

        session = self.memory.session()
        note = attempt.delivery_note()
        if note:
            session_note = f"## Note on how this attempt ended\n\n{note}\n\n"
        else:
            session_note = ""
        if session_note:
            actor_context.append("user", session_note)

        try:
            learning = consolidate(
                actor_context,
                session,
                verdict=attempt.verdict,
                findings=attempt.final_findings,
            )
        except UnverifiedOutcome:
            session.discard()
            return None

        self.journal.log_memory(
            learning.stats_after,
            phase=Phase.DRS.value,
            label=f"target-cycle{cycle_index}",
            kind="drs_checkpoint",
            verdict=attempt.verdict.value,
        )
        self.journal.log(
            "memory_commit",
            phase=Phase.DRS.value,
            cycle=cycle_index,
            unit="target",
            verdict=attempt.verdict.value,
            writes=learning.files_written,
            deletes=learning.files_deleted,
            diagnosis=learning.diagnosis[:2000],
            **(learning.stats_after.as_dict() if learning.stats_after else {}),
        )
        return learning

    # -- practice ---------------------------------------------------------
    def _run_practice(
        self, project: Project, index: int
    ) -> tuple[AttemptResult | None, LearningResult | None]:
        """Actor execution, verifier judgment, memory update."""
        environment = self.practice_environment_factory(f"drs-practice-{index}-{project.id}")
        provision(environment, project.fixtures, reset=True)
        actor_context = self.actor_context_factory()
        verifier_context = self.practice_verifier_context_factory()

        self.journal.log(
            "practice_start",
            phase=Phase.DRS.value,
            project=project.id,
            instruction=project.instruction,
        )
        try:
            attempt = self.practice_harness.attempt(
                environment,
                actor_context=actor_context,
                verifier_context=verifier_context,
                instruction=project.instruction,
                memory_text=self.memory.render(),
                practice=True,
                project_id=project.id,
                fixtures=project.fixtures,
                iteration_limit=self.config.limits.practice_iterations,
                watchdog_s=self.config.limits.practice_watchdog_s,
                program_timeout_s=self.config.limits.program_timeout_s,
            )
        except InfrastructureFailure as exc:
            self.journal.log(
                "practice_failed", phase=Phase.DRS.value, project=project.id, error=str(exc)
            )
            return None, None

        self.journal.log(
            "practice_verdict",
            phase=Phase.DRS.value,
            project=project.id,
            verdict=attempt.verdict.value,
            findings=attempt.final_findings[:2000],
        )
        if not attempt.grounded:
            return attempt, None

        session: MemorySession = self.memory.session()
        try:
            learning = consolidate(
                actor_context, session, verdict=attempt.verdict, findings=attempt.final_findings
            )
        except UnverifiedOutcome:
            session.discard()
            return attempt, None

        self.journal.log_memory(
            learning.stats_after,
            phase=Phase.DRS.value,
            label=project.id,
            kind="drs_practice",
            verdict=attempt.verdict.value,
        )
        self.journal.log(
            "memory_commit",
            phase=Phase.DRS.value,
            unit="practice",
            project=project.id,
            verdict=attempt.verdict.value,
            writes=learning.files_written,
            deletes=learning.files_deleted,
            diagnosis=learning.diagnosis[:2000],
            **(learning.stats_after.as_dict() if learning.stats_after else {}),
        )
        return attempt, learning

    # -- curriculum -------------------------------------------------------
    def _review(
        self,
        *,
        target_query: str,
        attempt: AttemptResult,
        diagnosis: str,
        practice_count: int,
        label: str = "target",
    ):
        outcome = OutcomeSummary(
            label=label,
            verdict=attempt.verdict.value,
            instruction=attempt.instruction,
            findings=attempt.final_findings,
            phase=Phase.DRS.value,
            practice=label != "target",
        )
        handoff = self.curriculum.review(
            self.curriculum_context,
            target_query=target_query,
            outcome=outcome,
            diagnosis=diagnosis,
            memory_view=self.memory.readonly_view().render(),
            practice_count=practice_count,
        )
        self.journal.log(
            "curriculum_review",
            phase=Phase.DRS.value,
            decision=handoff.decision.value,
            rationale=handoff.rationale,
            projects=[p.id for p in handoff.projects],
            practice_count=practice_count,
        )
        return handoff
