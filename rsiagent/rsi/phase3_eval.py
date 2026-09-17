"""Stage 3 — Test-time memory reuse under a sealed evaluator.

The three things that make this phase meaningful rather than decorative:

* **The curriculum agent and memory updates are disabled.**  "At this stage, the
  curriculum agent and all memory updates are disabled."  A single
  ``writeback=False`` bank and the absence of a curriculum context enforce it.
* **The same harness.**  "The actor–verifier harness then attempts the target
  after an environment reset."  Evaluation is not a different agent — it is the
  same agent with learning switched off.  That is what makes "RSI improves the
  agent" a claim about memory rather than about a second system.
* **The evaluator is sealed and runs last.**  "The official evaluator is invoked
  after the actor–verifier loop and does not supply scores or hidden checks to
  the learning agents."  Scores are written to the journal, never to a context.

The frozen memory's file-tree hash is recorded before and re-checked after, so a
run can demonstrate it evaluated against exactly the memory exploration
produced.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agents.actor import ActorAgent
from ..agents.verifier import VerifierAgent
from ..config import RunConfig
from ..env.base import Environment
from ..errors import InfrastructureFailure, MemoryIntegrityError
from ..memory.bank import MemoryBank, MemorySnapshot
from ..runtime.harness import AttemptResult, TaskHarness
from ..runtime.journal import Journal
from ..status import Phase, TerminalStatus, Verdict
from ..tasks.spec import Evaluator, TaskQuery, TaskScore


@dataclass
class Phase3Result:
    attempt: AttemptResult | None = None
    score: TaskScore | None = None
    frozen: MemorySnapshot | None = None
    memory_intact: bool = True
    status: TerminalStatus = TerminalStatus.COMPLETED
    rationale: str = ""
    integrity_error: str = ""

    @property
    def partial(self) -> float:
        return self.score.partial if self.score else 0.0

    @property
    def binary(self) -> bool:
        return bool(self.score and self.score.binary)

    @property
    def verdict(self) -> Verdict | None:
        return self.attempt.verdict if self.attempt else None


class FrozenMemoryEvaluator:
    """Runs a task against frozen memory with learning disabled."""

    def __init__(
        self,
        *,
        config: RunConfig,
        actor: ActorAgent,
        verifier: VerifierAgent,
        memory: MemoryBank,
        journal: Journal,
        environment: Environment,
        evaluator: Evaluator,
        actor_context_factory,
        verifier_context_factory,
    ) -> None:
        self.config = config
        self.actor = actor
        self.verifier = verifier
        self.memory = memory
        self.journal = journal
        self.environment = environment
        self.evaluator = evaluator
        self.actor_context_factory = actor_context_factory
        self.verifier_context_factory = verifier_context_factory
        self.harness = TaskHarness(
            actor, verifier, max_revisions=config.exploration.max_target_revisions
        )

    def run(self, task: TaskQuery) -> Phase3Result:
        result = Phase3Result()

        frozen = self.memory.freeze_to(self.config.frozen_memory_path)
        result.frozen = frozen
        self.journal.log(
            "memory_frozen",
            phase=Phase.EVAL.value,
            path=str(frozen.path),
            **frozen.stats.as_dict(),
        )
        self.journal.log_memory(
            frozen.stats, phase=Phase.EVAL.value, label="frozen", kind="freeze"
        )

        # A read-only bank over the frozen bytes.  Any write attempt raises
        # rather than silently succeeding, so "no writeback" is testable.
        frozen_bank = MemoryBank(frozen.path, writeback=False)
        memory_text = frozen_bank.render()

        self.environment.reset()
        actor_context = self.actor_context_factory()
        verifier_context = self.verifier_context_factory()

        try:
            attempt = self.harness.attempt(
                self.environment,
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
            result.status = TerminalStatus.INFRASTRUCTURE_FAILURE
            result.rationale = f"evaluation attempt failed: {exc.reason}"
            self.journal.log(
                "evaluation_failed", phase=Phase.EVAL.value, error=exc.reason
            )
            return result

        result.attempt = attempt
        self.journal.log(
            "evaluation_verdict",
            phase=Phase.EVAL.value,
            verdict=attempt.verdict.value,
            revisions=attempt.revisions,
            terminal=attempt.terminal,
            findings=attempt.final_findings[:4000],
        )

        # --- sealed official scoring, outside every agent context -----------
        try:
            score = self.evaluator.score(self.environment)
        except Exception as exc:  # a grader crash is unscored, not a zero
            score = TaskScore(partial=0.0, error=f"{type(exc).__name__}: {exc}")
        result.score = score
        self.journal.log(
            "official_score",
            phase=Phase.EVAL.value,
            evaluator=self.evaluator.name,
            agent_verdict=attempt.verdict.value,
            **score.as_dict(),
        )

        # --- integrity check -------------------------------------------------
        try:
            self.memory.assert_frozen(frozen)
        except MemoryIntegrityError as exc:
            result.memory_intact = False
            result.integrity_error = str(exc)
            result.status = TerminalStatus.INFRASTRUCTURE_FAILURE
            result.rationale = "frozen memory was modified during evaluation"
            self.journal.log(
                "memory_integrity_violation", phase=Phase.EVAL.value, error=str(exc)
            )
            return result

        result.status = TerminalStatus.COMPLETED
        if not attempt.grounded:
            result.rationale = "evaluation completed with an unresolved agent verdict"
        elif attempt.verdict is Verdict.FAIL:
            result.rationale = "evaluation completed; agent verdict FAIL"
        else:
            result.rationale = "evaluation completed; agent verdict PASS"
        return result
