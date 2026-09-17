"""Stage 1 — Broad Recursive Self-exploration.

BRS is where the paper's concurrency rules live, and they are unusual enough to
state precisely:

* The curriculum agent authors a **wave** of independent projects.
* Every project in the wave starts from the **same immutable pre-wave memory
  snapshot** and "cannot observe sibling work or outcomes".
* Work and verification happen **in parallel**, in isolated environments.
* Memory commits happen **serially, afterwards**, "in the order authored by the
  curriculum agent", and "each update operates on the latest canonical memory,
  including preceding updates from the same wave".
* The next wave is selected **only after these commits finish**, "allowing the
  curriculum agent to respond to both verified outcomes and the knowledge
  actually retained."
* The budget "is checked between completed waves, without interrupting an
  ongoing wave" — so the realized project count can exceed the nominal budget.

The split between parallel acquisition and sequential, cumulative consolidation
is the whole design.  It buys throughput without letting two branches write
memory from a stale read of it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ..agents.actor import ActorAgent
from ..agents.context import AgentContext
from ..agents.curriculum import CurriculumAgent, Handoff, OutcomeSummary
from ..agents.verifier import VerifierAgent
from ..config import RunConfig
from ..env.base import Environment
from ..env.pool import EnvironmentFactory
from ..errors import InfrastructureFailure, UnverifiedOutcome
from ..learning.consolidation import LearningResult, consolidate
from ..memory.bank import MemoryBank, MemorySession, MemorySnapshot
from ..runtime.harness import AttemptResult, TaskHarness
from ..runtime.journal import Journal
from ..status import Decision, Phase, TerminalStatus, Verdict
from ..tasks.spec import Project, provision


@dataclass
class Branch:
    """One project's full lineage: isolate, work, verify, then learn."""

    project: Project
    environment: Environment
    actor_context: AgentContext
    verifier_context: AgentContext
    work_session: MemorySession
    attempt: AttemptResult | None = None
    learning: LearningResult | None = None
    error: str = ""

    @property
    def verdict(self) -> Verdict | None:
        return self.attempt.verdict if self.attempt else None

    @property
    def grounded(self) -> bool:
        return bool(self.attempt and self.attempt.grounded)


@dataclass
class WaveResult:
    """One completed wave."""

    index: int
    decision: Decision
    rationale: str
    branches: list[Branch] = field(default_factory=list)
    committed: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""
    memory_before: MemorySnapshot | None = None
    memory_after_hash: str = ""

    @property
    def projects(self) -> int:
        return len(self.branches)

    def outcomes(self) -> list[OutcomeSummary]:
        summaries = []
        for branch in self.branches:
            if branch.attempt is None:
                continue
            summaries.append(
                OutcomeSummary(
                    label=branch.project.label(),
                    verdict=branch.attempt.verdict.value,
                    instruction=branch.project.instruction,
                    findings=branch.attempt.final_findings,
                    phase=Phase.BRS.value,
                    practice=True,
                )
            )
        return summaries


@dataclass
class Phase1Result:
    waves: list[WaveResult] = field(default_factory=list)
    status: TerminalStatus = TerminalStatus.SATURATED
    total_projects: int = 0
    rationale: str = ""

    def outcomes(self) -> list[OutcomeSummary]:
        collected: list[OutcomeSummary] = []
        for wave in self.waves:
            collected.extend(wave.outcomes())
        return collected


class BroadExplorer:
    """Runs Phase 1 to saturation, budget exhaustion, or stall."""

    def __init__(
        self,
        *,
        config: RunConfig,
        actor: ActorAgent,
        verifier: VerifierAgent,
        curriculum: CurriculumAgent,
        memory: MemoryBank,
        journal: Journal,
        env_factory: EnvironmentFactory,
        actor_context_factory,
        verifier_context_factory,
        curriculum_context: AgentContext,
    ) -> None:
        self.config = config
        self.actor = actor
        self.verifier = verifier
        self.curriculum = curriculum
        self.memory = memory
        self.journal = journal
        self.env_factory = env_factory
        self.actor_context_factory = actor_context_factory
        self.verifier_context_factory = verifier_context_factory
        self.curriculum_context = curriculum_context
        self.harness = TaskHarness(
            actor,
            verifier,
            max_revisions=config.exploration.max_practice_revisions,
        )

    # -- driver -----------------------------------------------------------
    def run(self, target_query: str) -> Phase1Result:
        exploration = self.config.exploration
        result = Phase1Result()
        outcomes: list[OutcomeSummary] = []
        authored = 0
        wave_index = 0

        while True:
            if wave_index >= exploration.brs_max_waves:
                result.status = TerminalStatus.BUDGET_EXHAUSTED
                result.rationale = f"reached brs_max_waves={exploration.brs_max_waves}"
                break

            # "The budget is checked between completed waves."  A wave already
            # underway is never truncated, so this test sits at the top.
            if wave_index > 0 and authored >= exploration.brs_project_budget:
                result.status = TerminalStatus.BUDGET_EXHAUSTED
                result.rationale = (
                    f"reached nominal project budget ({exploration.brs_project_budget}) "
                    f"after {authored} projects"
                )
                break

            handoff = self.curriculum.author_wave(
                self.curriculum_context,
                target_query=target_query,
                memory_view=self.memory.readonly_view().render(),
                outcomes=outcomes,
                first_wave=(wave_index == 0),
                max_projects=exploration.brs_wave_width,
            )

            self.journal.log(
                "curriculum_handoff",
                phase=Phase.BRS.value,
                wave=wave_index,
                decision=handoff.decision.value,
                rationale=handoff.rationale,
                projects=[p.id for p in handoff.projects],
            )

            if handoff.decision is Decision.SATURATED:
                result.status = TerminalStatus.SATURATED
                result.rationale = handoff.rationale
                break
            if handoff.decision is Decision.STALLED:
                result.status = TerminalStatus.STALLED
                result.rationale = handoff.rationale
                break

            projects = [
                Project(id=p.id, instruction=p.instruction, fixtures=p.fixtures, wave=wave_index)
                for p in handoff.projects
            ]
            wave = self.run_wave(wave_index, projects, handoff)
            result.waves.append(wave)
            authored += wave.projects
            result.total_projects = authored

            if wave.blocked:
                result.status = TerminalStatus.UNVERIFIED
                result.rationale = wave.block_reason
                break

            outcomes.extend(wave.outcomes())
            wave_index += 1

        self.journal.log(
            "phase_complete",
            phase=Phase.BRS.value,
            status=result.status.value,
            waves=len(result.waves),
            projects=result.total_projects,
            rationale=result.rationale,
            memory=self.memory.stats().as_dict(),
        )
        return result

    # -- one wave ---------------------------------------------------------
    def run_wave(self, index: int, projects: list[Project], handoff: Handoff) -> WaveResult:
        pre_wave = self.memory.snapshot(label=f"wave{index}")
        wave = WaveResult(
            index=index,
            decision=handoff.decision,
            rationale=handoff.rationale,
            memory_before=pre_wave,
        )
        self.journal.log(
            "wave_start",
            phase=Phase.BRS.value,
            wave=index,
            projects=[p.id for p in projects],
            pre_wave_hash=pre_wave.tree_hash,
            pre_wave_files=pre_wave.stats.file_count,
        )

        branches = [self._prepare_branch(index, project, pre_wave) for project in projects]
        wave.branches = branches

        self._acquire(branches)

        unresolved = [b for b in branches if not b.grounded]
        if unresolved and self.config.exploration.brs_on_blocked == "halt":
            wave.blocked = True
            wave.block_reason = "wave blocked by unresolved branch(es): " + ", ".join(
                f"{b.project.id}({b.error or (b.verdict.value if b.verdict else 'no verdict')})"
                for b in unresolved
            )
            self.journal.log(
                "wave_blocked",
                phase=Phase.BRS.value,
                wave=index,
                reason=wave.block_reason,
            )
            for branch in branches:
                branch.work_session.discard()
            return wave

        # "Once all project verdicts are available, the original actor contexts
        # resume one at a time in the order authored by the curriculum agent."
        for branch in branches:
            if not branch.grounded:
                branch.work_session.discard()
                self.journal.log(
                    "branch_dropped",
                    phase=Phase.BRS.value,
                    wave=index,
                    project=branch.project.id,
                    reason=branch.error or "ungrounded verdict",
                )
                continue
            self._consolidate(branch, index)
            wave.committed.append(branch.project.id)

        wave.memory_after_hash = self.memory.tree_hash()
        self.journal.log(
            "wave_committed",
            phase=Phase.BRS.value,
            wave=index,
            committed=wave.committed,
            memory=self.memory.stats().as_dict(),
        )
        return wave

    def _prepare_branch(self, index: int, project: Project, pre_wave: MemorySnapshot) -> Branch:
        label = f"w{index}-{project.id}"
        environment = self.env_factory(label)
        provision(environment, project.fixtures, reset=True)
        return Branch(
            project=project,
            environment=environment,
            actor_context=self.actor_context_factory(),
            verifier_context=self.verifier_context_factory(),
            # The same immutable snapshot for every sibling, materialised into a
            # private working copy so nothing a branch writes is visible.
            work_session=self.memory.session(pre_wave),
        )

    def _acquire(self, branches: list[Branch]) -> None:
        """Execute and verify the wave, up to ``brs_concurrency`` at a time."""
        if not branches:
            return
        width = min(self.config.exploration.brs_concurrency, len(branches))

        if width == 1:
            for branch in branches:
                self._run_branch(branch)
            return

        with ThreadPoolExecutor(max_workers=width, thread_name_prefix="brs") as pool:
            futures = {pool.submit(self._run_branch, b): b for b in branches}
            for future in as_completed(futures):
                branch = futures[future]
                try:
                    future.result()
                except Exception as exc:  # already recorded on the branch
                    if not branch.error:
                        branch.error = f"{type(exc).__name__}: {exc}"

    def _run_branch(self, branch: Branch) -> None:
        """Work and verify one project.  Never raises for agent-level faults."""
        self.journal.log(
            "project_start",
            phase=Phase.BRS.value,
            project=branch.project.id,
            wave=branch.project.wave,
            instruction=branch.project.instruction,
        )
        try:
            branch.attempt = self.harness.attempt(
                branch.environment,
                actor_context=branch.actor_context,
                verifier_context=branch.verifier_context,
                instruction=branch.project.instruction,
                memory_text=branch.work_session.render(),
                practice=True,
                project_id=branch.project.id,
                fixtures=branch.project.fixtures,
                iteration_limit=self.config.limits.practice_iterations,
                watchdog_s=self.config.limits.practice_watchdog_s,
                program_timeout_s=self.config.limits.program_timeout_s,
                environment_note=_project_note(branch.project),
            )
        except InfrastructureFailure as exc:
            branch.error = f"infrastructure: {exc.reason}"
        except Exception as exc:
            branch.error = f"{type(exc).__name__}: {exc}"

        self.journal.log(
            "project_verdict",
            phase=Phase.BRS.value,
            project=branch.project.id,
            verdict=branch.verdict.value if branch.verdict else None,
            revisions=branch.attempt.revisions if branch.attempt else None,
            programs=branch.attempt.programs_run if branch.attempt else 0,
            findings=(branch.attempt.final_findings[:2000] if branch.attempt else ""),
            error=branch.error,
        )

    def _consolidate(self, branch: Branch, wave_index: int) -> None:
        """Distill and reconcile one branch into the *current* canonical memory."""
        assert branch.attempt is not None
        branch.work_session.discard()  # work-phase edits are dropped here

        # Opened on the live bank, so this branch sees every earlier commit from
        # its own wave.
        session = self.memory.session()
        branch.actor_context.append(
            "user",
            "## Memory has advanced\n\n"
            "Other projects finished while you worked. The memory below is the "
            "current canonical state, including updates from them; it may differ "
            "from the snapshot you started with. Reconcile against this, not "
            "against your starting copy.\n\n" + session.render(),
        )
        try:
            branch.learning = consolidate(
                branch.actor_context,
                session,
                verdict=branch.attempt.verdict,
                findings=branch.attempt.final_findings,
            )
        except UnverifiedOutcome as exc:
            branch.error = f"unverified: {exc.reason}"
            session.discard()
            return

        stats = branch.learning.stats_after
        self.journal.log_memory(
            stats,
            phase=Phase.BRS.value,
            label=branch.project.id,
            kind="brs_project",
            verdict=branch.attempt.verdict.value,
        )
        self.journal.log(
            "memory_commit",
            phase=Phase.BRS.value,
            wave=wave_index,
            project=branch.project.id,
            verdict=branch.attempt.verdict.value,
            writes=branch.learning.files_written,
            deletes=branch.learning.files_deleted,
            diagnosis=branch.learning.diagnosis[:2000],
            **stats.as_dict(),
        )


def _project_note(project: Project) -> str:
    if not project.fixtures:
        return ""
    listing = "\n".join(f"  - {name}" for name in sorted(project.fixtures))
    return f"## Inputs for this project\n\n{listing}"
