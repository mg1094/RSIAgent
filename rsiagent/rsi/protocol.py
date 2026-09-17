"""The reference RSI lifecycle — paper Algorithm A1.

    1. Initialize canonical memory M <- M0.
    2. BRS: repeat complete waves until a recorded curriculum stop or the
       complete-wave budget boundary.
    3. DRS: attempt q after an environment reset and obtain a grounded target
       verdict.  The same actor context consolidates that outcome into M.
    4. If the policy is verifier_pass and the target passed, proceed to Step 7.
       Otherwise, ask the curriculum agent to select the next experience.
    5. Execute each selected practice project sequentially: actor execution,
       verifier judgment, actor memory update, then curriculum review.
    6. If the preceding target passed and no new practice was selected, finish
       DRS.  Otherwise, return to Step 3.  A curriculum STALLED decision instead
       allows one final target attempt and learning update before recording the
       actual terminal verdict and ending DRS.
    7. Evaluation: freeze M, reset the environment, and execute q with the
       actor-verifier harness and no learning.

    Guard: unresolved verification or infrastructure errors suspend advancement;
    they are not converted into task verdicts.  A completed lifecycle need not
    have a successful target verdict.

:class:`RSIRunner` is that procedure, with the phase bodies delegated to
:mod:`rsiagent.rsi.phase1_brs`, :mod:`rsiagent.rsi.phase2_drs`, and
:mod:`rsiagent.rsi.phase3_eval`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..agents.actor import ActorAgent
from ..agents.context import ContextFactory
from ..agents.curriculum import CurriculumAgent
from ..agents.verifier import VerifierAgent
from ..config import RunConfig
from ..env.base import Environment
from ..env.pool import EnvironmentFactory, LocalEnvironmentPool
from ..errors import InfrastructureFailure
from ..llm.factory import RoleClients
from ..memory.bank import MemoryBank, MemorySnapshot, MemoryStats
from ..prompts import actor as actor_prompts
from ..prompts import verifier as verifier_prompts
from ..runtime.journal import Journal, NullJournal
from ..status import Phase, TerminalStatus, Verdict
from ..tasks.spec import Evaluator, TaskQuery, TaskScore, UnscoredEvaluator, provision
from .phase1_brs import BroadExplorer, Phase1Result
from .phase2_drs import DeepRefiner, Phase2Result
from .phase3_eval import FrozenMemoryEvaluator, Phase3Result


@dataclass
class RSIResult:
    """Everything one lineage produced."""

    task: TaskQuery
    status: TerminalStatus
    phase1: Phase1Result | None = None
    phase2: Phase2Result | None = None
    phase3: Phase3Result | None = None
    memory: MemoryStats | None = None
    frozen: MemorySnapshot | None = None
    rationale: str = ""
    role_swapped: bool = False

    @property
    def score(self) -> TaskScore | None:
        return self.phase3.score if self.phase3 else None

    @property
    def partial(self) -> float:
        return self.score.partial if self.score else 0.0

    @property
    def binary(self) -> bool:
        return bool(self.score and self.score.binary)

    @property
    def target_verdict(self) -> Verdict | None:
        return self.phase2.final_verdict if self.phase2 else None

    def summary(self) -> str:
        lines = [
            f"task            {self.task.id}",
            f"status          {self.status.value}",
            f"memory          {self.memory.file_count if self.memory else 0} files, "
            f"{self.memory.total_bytes if self.memory else 0} bytes, "
            f"{(self.memory.tree_hash[:12] if self.memory else '-')}",
        ]
        if self.phase1:
            lines.append(
                f"phase1 (BRS)    {len(self.phase1.waves)} waves, "
                f"{self.phase1.total_projects} projects, {self.phase1.status.value}"
            )
        if self.phase2:
            lines.append(
                f"phase2 (DRS)    {len(self.phase2.cycles)} target cycles, "
                f"{len(self.phase2.practice_projects)} practice projects, "
                f"final verdict "
                f"{self.phase2.final_verdict.value if self.phase2.final_verdict else '-'}"
            )
        if self.phase3:
            score = self.phase3.score
            lines.append(
                f"phase3 (eval)   partial={score.partial:.4f} binary={score.binary} "
                f"agent_verdict={self.phase3.verdict.value if self.phase3.verdict else '-'}"
                f"{'' if self.phase3.memory_intact else '  MEMORY MODIFIED'}"
            )
        if self.rationale:
            lines.append(f"rationale       {self.rationale}")
        return "\n".join(lines)


class RSIRunner:
    """Wires the three roles, the memory bank, and the phases together."""

    def __init__(
        self,
        config: RunConfig,
        task: TaskQuery,
        clients: RoleClients,
        *,
        env_factory: EnvironmentFactory | None = None,
        evaluator: Evaluator | None = None,
        journal: Journal | None = None,
        memory: MemoryBank | None = None,
        swapped_clients: RoleClients | None = None,
        user_channel=None,
    ) -> None:
        self.config = config
        self.task = task
        self.clients = clients
        self.evaluator = evaluator or UnscoredEvaluator()
        self.journal = journal or NullJournal()
        self.memory = memory or MemoryBank(config.memory_path, writeback=config.memory_writeback)
        self.swapped_clients = swapped_clients
        self.user_channel = user_channel

        self.env_factory = env_factory or LocalEnvironmentPool(
            config.run_dir / "envs",
            program_timeout_s=config.limits.program_timeout_s,
        )
        self.actor = ActorAgent(user_channel=user_channel)
        self.verifier = VerifierAgent(program_timeout_s=config.limits.program_timeout_s)
        self.curriculum = CurriculumAgent(max_wave_projects=config.exploration.brs_wave_width)

    # -- context factories ------------------------------------------------
    def _actor_factory(self, clients: RoleClients) -> ContextFactory:
        return ContextFactory(
            clients.actor,
            actor_prompts.ACTOR_SYSTEM,
            role="actor",
            temperature=self.config.actor.temperature,
            top_p=self.config.actor.top_p,
            max_tokens=self.config.actor.max_tokens,
        )

    def _target_verifier_factory(self, clients: RoleClients) -> ContextFactory:
        return ContextFactory(
            clients.verifier,
            verifier_prompts.VERIFIER_SYSTEM,
            role="verifier",
            temperature=self.config.verifier.temperature,
            top_p=self.config.verifier.top_p,
            max_tokens=self.config.verifier.max_tokens,
        )

    def _practice_verifier_factory(self, clients: RoleClients) -> ContextFactory:
        return ContextFactory(
            clients.verifier,
            verifier_prompts.PRACTICE_VERIFIER_SYSTEM,
            role="verifier-practice",
            temperature=self.config.verifier.temperature,
            top_p=self.config.verifier.top_p,
            max_tokens=self.config.verifier.max_tokens,
        )

    def _curriculum_factory(self, clients: RoleClients) -> ContextFactory:
        return ContextFactory(
            clients.curriculum,
            "(curriculum system prompt is supplied per call)",
            role="curriculum",
            temperature=self.config.curriculum.temperature,
            top_p=self.config.curriculum.top_p,
            max_tokens=self.config.curriculum.max_tokens,
        )

    # -- driver -----------------------------------------------------------
    def run(self) -> RSIResult:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        target_query = self.task.instruction

        self.journal.log(
            "run_start",
            task=self.task.id,
            instruction=target_query,
            config=self.config.to_dict(),
        )
        provision(self.env_factory("target"), self.task.fixtures, reset=True)

        result = RSIResult(task=self.task, status=TerminalStatus.COMPLETED)

        stages = self.config.stages
        try:
            if "brs" in stages:
                result.phase1 = self._run_phase1(target_query)
                if result.phase1.status in (
                    TerminalStatus.UNVERIFIED,
                    TerminalStatus.STALLED,
                ):
                    # The guard: an unresolved branch suspends advancement.  We
                    # still evaluate, because a frozen memory is a valid object
                    # to evaluate even when exploration ended early.
                    result.rationale = f"phase1 ended {result.phase1.status.value}"

            if "drs" in stages:
                result.phase2 = self._run_phase2(target_query)

                if (
                    result.phase2.status is TerminalStatus.STALLED
                    and self.config.limits.allow_stall_role_switch
                    and self.swapped_clients is not None
                ):
                    # "The target harness also permits one stall-triggered role
                    # switch ... when execution budget remains."
                    self.journal.log(
                        "role_switch", phase=Phase.DRS.value, reason="stalled target attempt"
                    )
                    result.phase2 = self._run_phase2(target_query, swap=True)
                    result.role_swapped = True

            result.phase3 = self._run_phase3()

        except InfrastructureFailure as exc:
            result.status = TerminalStatus.INFRASTRUCTURE_FAILURE
            result.rationale = f"{exc.stage}: {exc.reason}"
            self.journal.log(
                "run_failed", status=result.status.value, error=str(exc), stage=exc.stage
            )
            return self._finalize(result)

        result.status = self._terminal_status(result)
        if not result.rationale:
            result.rationale = (
                result.phase2.rationale if result.phase2 else ""
            ) or (result.phase1.rationale if result.phase1 else "")
        return self._finalize(result)

    def _finalize(self, result: RSIResult) -> RSIResult:
        result.memory = self.memory.stats()
        result.frozen = result.phase3.frozen if result.phase3 else None
        self.journal.log(
            "run_complete",
            status=result.status.value,
            memory=result.memory.as_dict(),
            score=result.score.as_dict() if result.score else None,
            target_verdict=(
                result.target_verdict.value if result.target_verdict else None
            ),
            rationale=result.rationale,
        )
        return result

    @staticmethod
    def _terminal_status(result: RSIResult) -> TerminalStatus:
        """The lineage's status is the *worst* news any phase reported.

        Infrastructure and unresolved-verification outcomes dominate, so a run
        that could not be graded is never reported as a clean completion.
        """
        if result.phase3 is None:
            return TerminalStatus.ABORTED
        if result.phase3.status is TerminalStatus.INFRASTRUCTURE_FAILURE:
            return TerminalStatus.INFRASTRUCTURE_FAILURE
        if result.phase3.attempt is not None and not result.phase3.attempt.grounded:
            return TerminalStatus.UNVERIFIED
        if result.phase2 and result.phase2.status is TerminalStatus.STALLED:
            return TerminalStatus.STALLED
        if result.phase1 and result.phase1.status is TerminalStatus.STALLED:
            return TerminalStatus.STALLED
        if result.phase1 and result.phase1.status is TerminalStatus.UNVERIFIED:
            return TerminalStatus.UNVERIFIED
        return TerminalStatus.COMPLETED

    # -- phases -----------------------------------------------------------
    def _run_phase1(self, target_query: str) -> Phase1Result:
        self.journal.log("phase_start", phase=Phase.BRS.value)
        clients = self.clients
        explorer = BroadExplorer(
            config=self.config,
            actor=self.actor,
            verifier=self.verifier,
            curriculum=self.curriculum,
            memory=self.memory,
            journal=self.journal,
            env_factory=self.env_factory,
            actor_context_factory=self._actor_factory(clients),
            verifier_context_factory=self._practice_verifier_factory(clients),
            curriculum_context=self._curriculum_factory(clients)(),
        )
        return explorer.run(target_query)

    def _run_phase2(self, target_query: str, *, swap: bool = False) -> Phase2Result:
        self.journal.log("phase_start", phase=Phase.DRS.value, swapped=swap)
        clients = self.swapped_clients if (swap and self.swapped_clients) else self.clients
        actor_clients = clients
        if swap and self.swapped_clients is not None:
            # Swap the roles, not just the models: the verifier model becomes the
            # actor and vice versa.
            actor_clients = RoleClients(
                actor=self.swapped_clients.verifier,
                verifier=self.swapped_clients.actor,
                curriculum=clients.curriculum,
                observer=clients.observer,
            )

        adapter = ActorAgent(user_channel=self.user_channel)
        target_env = self.env_factory("target")
        provision(target_env, self.task.fixtures, reset=True)

        refiner = DeepRefiner(
            config=self.config,
            actor=adapter,
            verifier=self.verifier,
            curriculum=self.curriculum,
            memory=self.memory,
            journal=self.journal,
            target_environment=target_env,
            practice_environment_factory=self.env_factory,
            actor_context_factory=self._actor_factory(actor_clients),
            verifier_context_factory=self._target_verifier_factory(clients),
            practice_verifier_context_factory=self._practice_verifier_factory(clients),
            curriculum_context=self._curriculum_factory(clients)(),
        )
        return refiner.run(self.task, target_query=target_query)

    def _run_phase3(self) -> Phase3Result:
        self.journal.log("phase_start", phase=Phase.EVAL.value)
        eval_env = self.env_factory("evaluation")
        provision(eval_env, self.task.fixtures, reset=True)
        evaluator = FrozenMemoryEvaluator(
            config=self.config,
            actor=self.actor,
            verifier=self.verifier,
            memory=self.memory,
            journal=self.journal,
            environment=eval_env,
            evaluator=self.evaluator,
            actor_context_factory=self._actor_factory(self.clients),
            verifier_context_factory=self._target_verifier_factory(self.clients),
        )
        return evaluator.run(self.task)

    def harness_attempt(
        self,
        environment: Environment,
        actor_factory: ContextFactory,
        verifier_factory: ContextFactory,
        task: TaskQuery,
    ):
        """One actor-verifier attempt with no memory and no curriculum.

        Shared by the ``w/o RSI`` baseline, which is "the same code-as-policy
        actor-verifier harness" with exploration and persistent memory disabled.
        """
        from ..runtime.harness import TaskHarness

        harness = TaskHarness(
            self.actor,
            self.verifier,
            max_revisions=self.config.exploration.max_target_revisions,
        )
        return harness.attempt(
            environment,
            actor_context=actor_factory(),
            verifier_context=verifier_factory(),
            instruction=task.instruction,
            memory_text=None,
            practice=False,
            task_id=task.id,
            iteration_limit=self.config.limits.target_iterations,
            watchdog_s=self.config.limits.target_watchdog_s,
            program_timeout_s=self.config.limits.program_timeout_s,
        )


@dataclass
class BaselineResult:
    """The ``RSIAgent (w/o RSI)`` condition: same harness, no memory."""

    task: TaskQuery
    score: TaskScore | None = None
    verdict: Verdict | None = None
    status: TerminalStatus = TerminalStatus.COMPLETED
    rationale: str = ""
    frozen: MemorySnapshot | None = None

    @property
    def partial(self) -> float:
        return self.score.partial if self.score else 0.0

    @property
    def binary(self) -> bool:
        return bool(self.score and self.score.binary)

    def summary(self) -> str:
        score = self.score
        return (
            f"task            {self.task.id}\n"
            f"status          {self.status.value}\n"
            f"baseline        partial={score.partial:.4f} binary={score.binary}\n"
            f"agent_verdict   {self.verdict.value if self.verdict else '-'}"
        )


def run_baseline(
    config: RunConfig,
    task: TaskQuery,
    clients: RoleClients,
    *,
    env_factory: EnvironmentFactory | None = None,
    evaluator: Evaluator | None = None,
    journal: Journal | None = None,
) -> BaselineResult:
    """Execute the target with no exploration and no persistent memory."""
    journal = journal or NullJournal()
    baseline_config = dataclasses.replace(config, enable_memory=False, memory_writeback=False)
    runner = RSIRunner(
        baseline_config,
        task,
        clients,
        env_factory=env_factory,
        evaluator=evaluator,
        journal=journal,
        memory=MemoryBank(baseline_config.memory_path, writeback=False),
    )

    actor_factory = runner._actor_factory(clients)
    verifier_factory = runner._target_verifier_factory(clients)
    environment = runner.env_factory("baseline")
    provision(environment, task.fixtures, reset=True)

    journal.log("baseline_start", task=task.id, instruction=task.instruction)
    result = BaselineResult(task=task)
    try:
        attempt = runner.harness_attempt(environment, actor_factory, verifier_factory, task)
    except InfrastructureFailure as exc:
        result.status = TerminalStatus.INFRASTRUCTURE_FAILURE
        result.rationale = f"{exc.stage}: {exc.reason}"
        return result

    result.verdict = attempt.verdict
    try:
        result.score = (evaluator or UnscoredEvaluator()).score(environment)
    except Exception as exc:
        result.score = TaskScore(partial=0.0, error=f"{type(exc).__name__}: {exc}")
    journal.log(
        "baseline_score",
        evaluator=(evaluator or UnscoredEvaluator()).name,
        agent_verdict=attempt.verdict.value,
        **result.score.as_dict(),
    )
    return result
